import copy
import datetime
import gc
import json
import os
import time
from pathlib import Path

import httpx
import hydra
import optuna
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI
from optuna.samplers import TPESampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from evaluate import (
    combine_scores_per_concept,
    eval_steering,
    log_results_to_wandb,
    plot_steering,
    run_eval,
)
from hypersteer import get_model
from hypersteer.data import get_steering_dataset_factory
from hypersteer.utils.configs import (
    ExperimentConfig,
    InferenceConfig,
    config_to_pydantic,
)
from hypersteer.utils.constants import *  # noqa: F403
from hypersteer.utils.constants import (
    CONFIG_FILE,
    HYPERNETWORK_MODELS,
    STEERING_EXCLUDE_MODELS,
)
from hypersteer.utils.dry_run import patch_client
from hypersteer.utils.helpers import (
    barrier,
    combine_all_results,
    configure_tokenizer_model,
    destroy_process_group,
    dump_json,
    get_and_set_device,
    get_cache_key,
    get_logger,
    get_rank,
    get_world_size,
)
from hypersteer.utils.model_utils import get_prefix_length, get_suffix_length

# Initialize the logger
logger = get_logger(__name__)


# optimization to avoid repeated model and tokenizer loading
global_model_instance = None
global_tokenizer = None
global_model_name = None


def get_or_load_model(model_name, device, use_bf16=False, special_tokens=None):
    """Helper function to get existing model or load it if needed"""
    global global_model_instance, global_tokenizer, global_model_name

    # If we already have the model loaded and it's the right one, return it
    if global_model_instance is not None and global_model_name == model_name:
        return global_model_instance, global_tokenizer

    # Otherwise, load the model and tokenizer
    logger.info(f"Loading model {model_name} onto device {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=1024, use_fast=False
    )
    tokenizer.padding_side = "right"

    model_instance = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else None,
        device_map=device,
    )
    model_instance = model_instance.eval()

    # Configure tokenizer and model
    configure_tokenizer_model(model_instance, tokenizer, special_tokens=special_tokens)

    # Store in globals
    global_model_instance = model_instance
    global_tokenizer = tokenizer
    global_model_name = model_name

    return model_instance, tokenizer


def load_config(config_path):
    """
    Load metadata from a JSON lines file.
    """
    if not os.path.exists(Path(config_path) / CONFIG_FILE):
        return None
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d


def load_dataset_for_inference(args):
    """
    Load HuggingFace dataset for inference and extract concept information.
    """
    from hypersteer.data import get_training_dataset

    # Load the dataset using the same function as training
    dataset = get_training_dataset(
        dataset_type="axbench",
        dataset_name=args.dataset.hf_dataset_name,
        data_files=args.dataset.hf_data_files,
        split=args.dataset.hf_split,
        cache_dir=args.dataset.cache_dir,
        select_concept_ids=args.dataset.select_concept_ids,
        max_concepts=args.dataset.max_concepts,
    )

    # Extract unique concept information from the dataset
    df = dataset.to_pandas()
    concept_info = []

    # Get unique concepts with their IDs
    unique_concepts = df.groupby("concept_id").first()

    for concept_id, row in unique_concepts.iterrows():
        if concept_id >= 0:  # Skip negative concept IDs
            concept_info.append(
                {
                    "concept_id": concept_id,
                    "concept": row.get("input_concept", f"concept_{concept_id}"),
                    "ref": f"https://neuronpedia.org/api/feature/{concept_id}",  # Default SAE link format
                    "concept_genres_map": {
                        row.get("input_concept", f"concept_{concept_id}"): ["text"]
                    },
                }
            )

    return concept_info


def create_data_steering(
    dataset_factory,
    concept_info,
    concept_id,
    num_of_examples,
    steering_factors,
    steering_datasets,
    args: InferenceConfig,
):
    # Find concept info for this concept_id
    concept_data = next(
        (c for c in concept_info if c["concept_id"] == concept_id), None
    )
    if concept_data is None:
        raise ValueError(f"Concept ID {concept_id} not found in dataset")

    concept = concept_data["concept"]
    sae_link = concept_data["ref"]
    sae_id = int(sae_link.split("/")[-1])

    current_df = dataset_factory.create_eval_df(
        [concept],
        num_of_examples,
        steering_factors,
        steering_datasets,
        concept_id=concept_id,
        steering_model_name=args.steering_model_name,
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id

    return current_df, (concept_id, sae_link, sae_id)


def prepare_df(current_df, tokenizer, is_chat_model, model_name):
    suffix_length, _ = get_suffix_length(tokenizer)
    if is_chat_model:
        if model_name == "meta-llama/Llama-3.1-8B-Instruct":

            def apply_chat_template(row):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[
                    1:-suffix_length
                ]
                return tokenizer.decode(tokens)

            current_df["input"] = current_df.apply(apply_chat_template, axis=1)
        else:

            def apply_chat_template(row):
                messages = [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[
                    1:-suffix_length
                ]
                return tokenizer.decode(tokens)

            current_df["input"] = current_df.apply(apply_chat_template, axis=1)
    return current_df


def infer_steering(
    args: ExperimentConfig,
    rank,
    world_size,
    device,
    logger,
    infer_run="inference",
):
    train_dir = args.dataset.train_dir
    dump_dir = args.dataset.dump_dir
    num_of_examples = args.inference.steering_num_of_examples
    concept_info = load_dataset_for_inference(args)
    steering_factors = args.inference.steering_factors
    steering_datasets = args.inference.steering_datasets

    # Get list of all concept_ids
    concept_ids = [info["concept_id"] for info in concept_info]

    if args.dataset.select_concept_ids:
        concept_ids = [
            concept_id
            for concept_id in concept_ids
            if concept_id in args.dataset.select_concept_ids
        ]
        if len(concept_ids) != len(args.dataset.select_concept_ids):
            raise ValueError(
                f"Selected concept IDs {args.dataset.select_concept_ids} not found in dataset"
            )

        logger.debug(f"Selected concept IDs: {concept_ids}")

    # For now, only rank 0 processes all concepts (proper distributed inference will be implemented later)
    if rank == 0:
        my_concept_ids = concept_ids
    else:
        my_concept_ids = []  # Other ranks do nothing for now

    if len(my_concept_ids) == 0:
        logger.info("No concepts to process. Exiting.")
        return

    # Create a new OpenAI client.
    lm_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )
    if int(os.environ.get("OPENAI_DRY_RUN", "1")) == 1:
        lm_client = patch_client(lm_client)

    # Initialize the dataset factory with the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        args.inference.steering_model_name, use_fast=False, model_max_length=1024
    )
    tokenizer.padding_side = "right"
    # Check if we're using PromptSteering or if the model has synergy enabled
    model_name = args.model.model_name
    if model_name == "PromptSteering":
        has_prompt_steering = True
    else:
        has_prompt_steering = getattr(args.model, "use_synergy", False)

    # Use the new dataset factory abstraction
    dataset_factory = get_steering_dataset_factory(
        "axbench",  # Default to axbench for now, could be configurable
        tokenizer=tokenizer,
        dump_dir=dump_dir,
        master_data_dir=args.inference.master_data_dir,
        lm_client=lm_client,
        lm_model=args.inference.lm_model,
        has_prompt_steering=has_prompt_steering,
    )

    is_chat_model = True if args.inference.model_name in CHAT_MODELS else False  # noqa: F405
    prefix_length = 1  # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.info(f"Chat model prefix length: {prefix_length}")

    # Load model instance onto device
    if args.inference.use_bf16:
        logger.info(f"Using bfloat16 for model {args.inference.model_name}")

    model_instance, tokenizer = get_or_load_model(
        args.inference.model_name,
        device,
        use_bf16=args.inference.use_bf16,
        special_tokens=args.train.special_tokens
        if hasattr(args.train, "special_tokens")
        else None,
    )

    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    cache_dir = Path(args.dataset.cache_dir or "assets/data/axbench/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(
        args.inference, my_concept_ids, concept_info, is_latent=False
    )
    cache_file = os.path.join(cache_dir, f"steering_data_cache_{cache_key}.parquet")

    # Try to load from cache first
    if os.path.exists(cache_file) and args.inference.use_cache:
        logger.info(f"Loading steering data from cache {cache_file}")
        cached_df = pd.read_parquet(cache_file)
        data_per_concept = {}
        # Reconstruct the data_per_concept dictionary from the cached DataFrame
        for concept_id in my_concept_ids:
            concept_data = cached_df[cached_df["concept_id"] == concept_id]
            if not concept_data.empty:
                sae_link = concept_data["sae_link"].iloc[0]
                sae_id = concept_data["sae_id"].iloc[0]
                data_per_concept[concept_id] = (
                    concept_data.drop(["sae_link", "sae_id"], axis=1),
                    sae_link,
                    sae_id,
                )
    else:
        logger.info("Generating steering data and caching results")
        data_per_concept = {}
        all_dfs = []

        for concept_id in my_concept_ids:
            current_df, (_, sae_link, sae_id) = create_data_steering(
                dataset_factory,
                concept_info,
                concept_id,
                num_of_examples,
                steering_factors,
                steering_datasets,
                args.inference,
            )
            # Add sae info to DataFrame for caching
            current_df["sae_link"] = sae_link
            current_df["sae_id"] = sae_id
            all_dfs.append(current_df)

            # Store in memory format
            data_per_concept[concept_id] = (
                current_df.drop(["sae_link", "sae_id"], axis=1),
                sae_link,
                sae_id,
            )

        # Cache the results
        if all_dfs and args.inference.use_cache:
            cached_df = pd.concat(all_dfs, ignore_index=True)
            cached_df.to_parquet(cache_file)
            logger.info(f"Cached steering data to {cache_file}")

    # Determine which models to run inference on
    models_to_run = []
    if args.inference.models:
        # Use the models list if provided
        models_to_run = args.inference.models
    else:
        # Fall back to single model_name for backward compatibility
        models_to_run = [args.model.model_name]

    logger.info(f"Running inference on models: {models_to_run}")

    # Check if any models are excluded from steering
    valid_models = [
        model for model in models_to_run if model not in STEERING_EXCLUDE_MODELS
    ]
    excluded_models = [
        model for model in models_to_run if model in STEERING_EXCLUDE_MODELS
    ]

    if excluded_models:
        logger.warning(f"Models {excluded_models} are excluded from steering, skipping")

    if not valid_models:
        logger.warning("No valid models to run inference on, exiting")
        return

    # Create combined DataFrame for all concepts
    logger.debug("Preparing combined DataFrame for batch inference")
    combined_dfs = []
    for concept_id in my_concept_ids:
        concept_df = data_per_concept[concept_id][0]
        combined_dfs.append(concept_df)
    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Run inference for each model using full batch inference
    for model_name in valid_models:
        logger.info(f"Running batch inference for model: {model_name}")

        # Both PromptSteering and Hypersteer models support full batch inference
        if model_name in HYPERNETWORK_MODELS or model_name == "PromptSteering":
            logger.info(f"Loading {model_name} on {device} for full batch inference.")

            model_config = args.model

            benchmark_model = get_model(
                model_name,
                model=model_instance,
                tokenizer=tokenizer,
                low_rank_dimension=len(concept_info),
                device=device,
                training_args=model_config,
                concept_ids=my_concept_ids,
            )
            benchmark_model.load(
                dump_dir=train_dir,
                mode="steering",
            )
            benchmark_model.to(device)
            if hasattr(benchmark_model, "ax"):
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)

            # Run full batch inference on all concepts at once
            logger.info(
                f"Running full batch inference on {len(my_concept_ids)} concepts"
            )
            results = benchmark_model.predict_steer(
                combined_df,
                batch_size=args.inference.steering_batch_size,
                prefix_length=prefix_length,
                dump_dir=Path(args.dataset.dump_dir) / infer_run,
                concept_id=my_concept_ids,  # Pass all concept IDs for batch processing
            )

            # Store the results in combined_df
            for k, v in results.items():
                combined_df[f"{model_name}_{k}"] = v

            del benchmark_model
            torch.cuda.empty_cache()
        else:
            # Fallback for models that don't support batch inference (should be rare)
            logger.warning(
                f"Model {model_name} does not support batch inference, using individual concept processing."
            )

            for concept_id in my_concept_ids:
                current_df, sae_link, sae_id = data_per_concept[concept_id]
                model_config = args.model

                benchmark_model = get_model(
                    model_name,
                    model=model_instance,
                    tokenizer=tokenizer,
                    training_args=model_config,
                    low_rank_dimension=len(concept_info),
                    device=device,
                )
                benchmark_model.load(
                    dump_dir=train_dir,
                    sae_path=concept_info[0]["ref"],
                    mode="steering",
                    intervention_type=args.inference.steering_intervention_type,
                    concept_id=concept_id,
                    concept_ids=my_concept_ids,
                )
                benchmark_model.to(device)
                if hasattr(benchmark_model, "ax") and args.inference.use_bf16:
                    benchmark_model.ax.eval()
                    benchmark_model.ax.to(torch.bfloat16)

                logger.info(
                    f"Inference steering with {model_name} on {device} for concept {concept_id}."
                )

                # Run prediction
                results = benchmark_model.predict_steer(
                    current_df,
                    concept_id=concept_id,
                    sae_link=sae_link,
                    sae_id=sae_id,
                    dump_dir=Path(args.dataset.dump_dir) / infer_run,
                    batch_size=args.inference.steering_batch_size,
                    eval_output_length=args.inference.steering_output_length,
                    temperature=args.inference.temperature,
                    prefix_length=prefix_length,
                    positions=model_config.intervention_positions
                    if model_name not in {"PromptSteering", "GemmaScopeSAE"}
                    else None,
                    use_synergy=getattr(model_config, "use_synergy", False),
                    disable_neuronpedia_max_act=args.inference.disable_neuronpedia_max_act,
                )
                # Store the results in current_df and update combined_df
                for k, v in results.items():
                    current_df[f"{model_name}_{k}"] = v

                # Update the corresponding rows in combined_df
                concept_mask = combined_df["concept_id"] == concept_id
                for k, v in results.items():
                    combined_df.loc[concept_mask, f"{model_name}_{k}"] = v

                del benchmark_model
                torch.cuda.empty_cache()

    # Save results only on rank 0 (assuming distributed inference will be implemented properly in the future)
    if rank == 0:
        combined_df = combined_df.sort_values(
            by=["concept_id", "input_id", "factor"]
        ).reset_index(drop=True)
        combined_df.to_parquet(
            Path(dump_dir) / infer_run / "steering_data.parquet", engine="pyarrow"
        )
        logger.info(
            f"Saved steering inference results to {Path(dump_dir) / infer_run / 'steering_data.parquet'}"
        )

    # Synchronize all processes
    barrier()


def select_steering_factors(
    args: ExperimentConfig, rank, world_size, device, logger, infer_run="inference"
):
    """Use Optuna with TPE to select the best steering factor across all concepts."""

    logger.info("=" * 80)
    logger.info("Starting steering factor selection with Optuna TPE optimizer")
    logger.info("=" * 80)

    dump_dir = Path(args.dataset.dump_dir)
    concept_info = load_dataset_for_inference(args)

    # Make eval run dir
    if args.evaluate.run_distinct_evals:
        eval_run = f"evaluate_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    else:
        eval_run = "evaluate"
    (dump_dir / eval_run).mkdir(parents=True, exist_ok=True)

    # Get list of all concept_ids
    concept_ids = [info["concept_id"] for info in concept_info]

    # Filter out concept ids if specified
    if args.dataset.select_concept_ids:
        concept_ids = [
            concept_id
            for concept_id in concept_ids
            if concept_id in args.dataset.select_concept_ids
        ]
        if len(concept_ids) != len(args.dataset.select_concept_ids):
            raise ValueError(
                f"Selected concept IDs {args.dataset.select_concept_ids} not found in dataset"
            )

        logger.debug(f"Selected concept IDs: {concept_ids}")

    # Define optimization parameters from config
    factor_selection_config = args.inference.factor_selection
    n_trials = factor_selection_config.n_calls  # Reuse n_calls as n_trials
    factor_min = factor_selection_config.factor_min
    factor_max = factor_selection_config.factor_max
    metric_name = factor_selection_config.metric
    report_to = args.evaluate.report_to
    # Set up additional parameters with sensible defaults if not present in config
    use_discrete_space = factor_selection_config.discrete_space
    discrete_steps = factor_selection_config.discrete_steps
    use_log_scale = factor_selection_config.log_scale
    n_startup_trials = (
        factor_selection_config.n_startup_trials
    )  # Random trials at start
    capture_all_metrics = factor_selection_config.capture_all_metrics

    # Generate discrete values if using discrete space
    discrete_values = None
    if use_discrete_space:
        discrete_values = [
            factor_min + (factor_max - factor_min) * i / discrete_steps
            for i in range(discrete_steps + 1)
        ]
        logger.info(f"Using discrete factor space with values: {discrete_values}")

    trial_results_dict = {}
    trial_times = []

    # Define the objective function that evaluates all concepts
    def objective(trial):
        trial_start_time = time.time()
        # Sample steering factor based on configuration
        if use_discrete_space:
            factor = trial.suggest_categorical("factor", discrete_values)
        else:
            # Use continuous space with optional log scaling
            factor = trial.suggest_float(
                "factor", factor_min, factor_max, log=use_log_scale
            )

        logger.info(
            f"Evaluating steering factor: {factor} for all concepts (trial {trial.number + 1}/{n_trials})"
        )

        trial_args = copy.deepcopy(args)

        # Update config to use just this single factor
        trial_args.inference.steering_factors = [factor]
        trial_args.evaluate.report_to = None
        # No need to set models since we use the single model configuration

        local_trial_infer_run = (
            Path(dump_dir) / infer_run / f"inference_trial_{trial.number}"
        )
        local_trial_eval_run = Path(dump_dir) / infer_run / f"eval_trial_{trial.number}"
        local_trial_infer_run.mkdir(parents=True, exist_ok=True)
        local_trial_eval_run.mkdir(parents=True, exist_ok=True)

        # Run inference with this factor
        infer_steering(
            trial_args,
            rank,
            world_size,
            device,
            logger,
            infer_run=local_trial_infer_run,
        )

        # Run evaluation across all concepts
        eval_results = eval_steering(
            trial_args,
            select_concept_ids=concept_ids if concept_ids else None,
            return_results=True,
            eval_run=local_trial_eval_run,
            infer_run=local_trial_infer_run,
        )
        trial_num = trial.number
        trial_results_dict[trial_num] = {
            "factor": trial.params["factor"],
            "results": copy.deepcopy(eval_results),  # Deep copy to be safe
        }

        # Aggregate scores across all concepts
        for data in eval_results:
            data["results"]["LMJudgeEvaluator"] = combine_scores_per_concept(data)

        # Determine which models to use for factor selection
        factor_selection_models = []
        if trial_args.inference.factor_selection.models:
            factor_selection_models = trial_args.inference.factor_selection.models
        else:
            # Fall back to single model for backward compatibility
            factor_selection_models = [trial_args.inference.factor_selection.model]

        # Extract the aggregated scores for all concepts and compute the mean across all models
        total_score = 0.0
        total_count = 0

        for model_name in factor_selection_models:
            model_score = 0.0
            model_count = 0
            for result in eval_results:
                if (
                    model_name in result["results"]["LMJudgeEvaluator"]
                    and metric_name in result["results"]["LMJudgeEvaluator"][model_name]
                ):
                    score_val = result["results"]["LMJudgeEvaluator"][model_name][
                        metric_name
                    ][0]
                    model_score += score_val
                    model_count += 1

            if model_count > 0:
                total_score += model_score
                total_count += model_count

        mean_score = total_score / total_count if total_count > 0 else 0.0

        # Store all metrics to understand tradeoffs
        if capture_all_metrics:
            metrics_dict = {}
            for metric in [
                "relevance_concept_ratings",
                "fluency_ratings",
                "relevance_instruction_ratings",
                "lm_judge_rating",
            ]:
                metric_sum = 0.0
                metric_count = 0
                for result in eval_results:
                    if (
                        model_name in result["results"]["LMJudgeEvaluator"]
                        and metric in result["results"]["LMJudgeEvaluator"][model_name]
                    ):
                        metric_val = result["results"]["LMJudgeEvaluator"][model_name][
                            metric
                        ][0]
                        metric_sum += metric_val
                        metric_count += 1

                metrics_dict[metric] = (
                    metric_sum / metric_count if metric_count > 0 else 0.0
                )

            trial.set_user_attr("all_metrics", metrics_dict)
            logger.info(f"Factor: {factor}, All metrics: {metrics_dict}")

        logger.info(f"Factor: {factor}, Mean {metric_name}: {mean_score}")

        # Calculate and log the trial execution time
        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time
        trial_times.append(trial_duration)

        # Format as hours:minutes:seconds
        hours, remainder = divmod(trial_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        # Also store the timing information in the trial
        trial.set_user_attr("execution_time", trial_duration)
        trial.set_user_attr("execution_time_formatted", time_str)

        # Show average time and estimated remaining time
        avg_time = sum(trial_times) / len(trial_times)
        remaining_trials = n_trials - (trial.number + 1)
        est_remaining_time = avg_time * remaining_trials

        # Format estimated remaining time
        hours, remainder = divmod(est_remaining_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        est_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        logger.info(f"Factor: {factor}, Mean {metric_name}: {mean_score}")
        logger.info(f"Trial {trial.number + 1} execution time: {time_str}")
        logger.info(f"Average trial time: {avg_time:.2f} seconds")
        logger.info(
            f"Estimated remaining time: {est_time_str} ({remaining_trials} trials left)"
        )

        # Return positive score for optimization (Optuna maximizes by default)
        return mean_score

    sampler = TPESampler(
        seed=args.inference.seed,
        n_startup_trials=n_startup_trials,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
    total_start_time = time.time()
    logger.info(
        f"Starting optimization with {n_trials} trials at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    study.optimize(objective, n_trials=n_trials)

    # Log total execution time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    total_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    logger.info(f"Total optimization time: {total_time_str}")

    # Extract results
    best_trial = study.best_trial
    best_factor = best_trial.params["factor"]
    best_score = best_trial.value

    # Create detailed results report
    results = {
        "best_factor": best_factor,
        "best_score": best_score,
        "total_execution_time_seconds": total_duration,
        "total_execution_time_formatted": total_time_str,
        "all_trials": [
            {
                "trial": trial.number,
                "factor": trial.params["factor"],
                "score": trial.value,
                "execution_time_seconds": trial.user_attrs.get("execution_time", None),
                "execution_time_formatted": trial.user_attrs.get(
                    "execution_time_formatted", None
                ),
            }
            for trial in study.trials
        ],
    }

    # Create detailed results report
    results = {
        "best_factor": best_factor,
        "best_score": best_score,
        "all_trials": [
            {
                "trial": trial.number,
                "factor": trial.params["factor"],
                "score": trial.value,
            }
            for trial in study.trials
        ],
    }

    # Add detailed metrics if captured
    if capture_all_metrics:
        for i, trial in enumerate(study.trials):
            if "all_metrics" in trial.user_attrs:
                results["all_trials"][i]["detailed_metrics"] = trial.user_attrs[
                    "all_metrics"
                ]

    logger.info("=" * 80)
    logger.info(f"Best factor across all concepts: {best_factor} (score: {best_score})")
    logger.info(
        f"Trial history: {[(t.params['steering_factor'], t.value) for t in study.trials]}"
    )

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create plot directory
        plot_dir = dump_dir / infer_run / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)

        # Add execution time plot
        plt.figure(figsize=(10, 6))
        trial_numbers = [t.number for t in study.trials]
        exec_times = [t.user_attrs.get("execution_time", 0) for t in study.trials]
        plt.bar(trial_numbers, exec_times)
        plt.xlabel("Trial number")
        plt.ylabel("Execution time (seconds)")
        plt.title("Trial Execution Times")
        plt.grid(True, alpha=0.3)
        plt.savefig(str(plot_dir / "trial_execution_times.png"))
        plt.close()

        # Plot optimization history - Fix: directly use plt to create this plot
        plt.figure(figsize=(10, 6))
        trials_data = np.array([[t.number, t.value] for t in study.trials])
        if len(trials_data) > 0:
            plt.plot(trials_data[:, 0], trials_data[:, 1], "-o")
            plt.xlabel("Trial number")
            plt.ylabel(f"{metric_name} score (higher is better)")
            plt.title("Optimization History")
            plt.grid(True, alpha=0.3)
        plt.savefig(str(plot_dir / "optimization_history.png"))
        plt.close()

        # Custom scatter plot of steering factor vs score
        plt.figure(figsize=(10, 6))
        factors = [t.params["factor"] for t in study.trials]
        scores = [t.value for t in study.trials]
        plt.scatter(factors, scores)
        plt.xlabel("Steering Factor")
        plt.ylabel(f"{metric_name} (higher is better)")
        plt.title("Steering Factor vs Score")
        plt.grid(True, alpha=0.3)
        plt.savefig(str(plot_dir / "factor_vs_score.png"))
        plt.close()

        # If capturing all metrics, plot trade-off curves
        if (
            capture_all_metrics
            and len(study.trials) > 0
            and "all_metrics" in study.trials[0].user_attrs
        ):
            # Get the actual metric keys from the first trial
            available_metrics = list(study.trials[0].user_attrs["all_metrics"].keys())
            logger.info(f"Available metrics for visualization: {available_metrics}")

            # Plot individual metric trends
            for metric in available_metrics:
                if metric == metric_name:  # Skip the primary metric (already plotted)
                    continue
                metric_values = [
                    t.user_attrs["all_metrics"].get(metric, 0) for t in study.trials
                ]

                # Check if metric values are non-zero (otherwise skip this plot)
                if np.max(np.abs(metric_values)) < 1e-6:
                    logger.warning(f"Skipping plot for {metric} - all values are zero")
                    continue

                plt.figure(figsize=(10, 6))
                # Sort by factor for line plot
                sorted_data = sorted(zip(factors, metric_values))
                sorted_factors, sorted_metrics = zip(*sorted_data)
                # Create both scatter and line plot
                plt.scatter(factors, metric_values, label="Trial results")
                plt.plot(
                    sorted_factors, sorted_metrics, "r--", alpha=0.5, label="Trend"
                )
                plt.xlabel("Steering Factor")
                plt.ylabel(f"{metric}")
                plt.title(f"Steering Factor vs {metric}")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig(str(plot_dir / f"factor_vs_{metric}.png"))
                plt.close()

            # Create a 3D scatter plot showing the trade-off between all three metrics
            if len(available_metrics) >= 3:
                # Use only metrics with non-zero values
                valid_metrics = []
                for metric in available_metrics:
                    values = [
                        t.user_attrs["all_metrics"].get(metric, 0) for t in study.trials
                    ]
                    if np.max(np.abs(values)) > 1e-6:
                        valid_metrics.append(metric)

                if len(valid_metrics) >= 3:
                    # Use the first three valid metrics for the 3D plot
                    metrics_for_3d = valid_metrics[:3]

                    # Make sure the metrics actually have values (debug for troubleshooting)
                    logger.info(f"Using metrics for 3D plot: {metrics_for_3d}")
                    for metric in metrics_for_3d:
                        metric_values = [
                            t.user_attrs["all_metrics"].get(metric, 0)
                            for t in study.trials
                        ]
                        logger.info(
                            f"Metric {metric} values: min={min(metric_values)}, max={max(metric_values)}"
                        )

                    # Extract metric values
                    x_values = [
                        t.user_attrs["all_metrics"].get(metrics_for_3d[0], 0)
                        for t in study.trials
                    ]
                    y_values = [
                        t.user_attrs["all_metrics"].get(metrics_for_3d[1], 0)
                        for t in study.trials
                    ]
                    z_values = [
                        t.user_attrs["all_metrics"].get(metrics_for_3d[2], 0)
                        for t in study.trials
                    ]

                    # Create single figure for 3D plot
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection="3d")
                    scatter = ax.scatter(
                        x_values,
                        y_values,
                        z_values,
                        c=factors,
                        cmap="viridis",
                        s=100,
                        alpha=0.7,
                    )

                    # Add factor annotations to points
                    for i, (f, x, y, z) in enumerate(
                        zip(factors, x_values, y_values, z_values)
                    ):
                        ax.text(x, y, z, f"{f:.2f}", fontsize=8)

                    fig.colorbar(scatter, label="Steering Factor")
                    ax.set_xlabel(metrics_for_3d[0])
                    ax.set_ylabel(metrics_for_3d[1])
                    ax.set_zlabel(metrics_for_3d[2])
                    ax.set_title(f"Trade-off between {', '.join(metrics_for_3d)}")
                    plt.savefig(str(plot_dir / "3d_metric_tradeoff.png"))
                    plt.close()

                # Create pairwise 2D plots for all combinations of valid metrics
                for i in range(len(valid_metrics)):
                    for j in range(i + 1, len(valid_metrics)):
                        metric_i = valid_metrics[i]
                        metric_j = valid_metrics[j]

                        x_values = [
                            t.user_attrs["all_metrics"].get(metric_i, 0)
                            for t in study.trials
                        ]
                        y_values = [
                            t.user_attrs["all_metrics"].get(metric_j, 0)
                            for t in study.trials
                        ]

                        plt.figure(figsize=(12, 8))
                        plt.scatter(
                            x_values,
                            y_values,
                            c=factors,
                            cmap="viridis",
                            s=100,
                            alpha=0.7,
                        )

                        # Add factor annotations
                        for k, (f, x, y) in enumerate(zip(factors, x_values, y_values)):
                            plt.annotate(f"{f:.2f}", (x, y), fontsize=8)

                        plt.colorbar(label="Steering Factor")
                        plt.xlabel(metric_i)
                        plt.ylabel(metric_j)
                        plt.title(f"Trade-off between {metric_i} and {metric_j}")
                        plt.grid(True, alpha=0.3)
                        plt.savefig(
                            str(plot_dir / f"{metric_i}_vs_{metric_j}_tradeoff.png")
                        )
                        plt.close()

    except Exception as e:
        logger.warning(f"Failed to create optimization plots: {e}")
        import traceback

        logger.warning(traceback.format_exc())  # More detailed error info

    # Save results
    results_path = dump_dir / eval_run / "factor_selection_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    all_results = []
    for trial in study.trials:
        trial_num = trial.number
        if trial_num in trial_results_dict:
            # Get our saved results
            factor = trial_results_dict[trial_num]["factor"]
            trial_results = trial_results_dict[trial_num]["results"]
            # Add factor to each result
            for result in trial_results:
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["factor"] = factor
            all_results.append(trial_results)

    combined_results = combine_all_results(all_results)

    # Dump combined results as jsonl
    results_jsonl_path = dump_dir / eval_run / "combined_results.jsonl"
    with open(results_jsonl_path, "w") as f:
        for result in combined_results:
            json.dump(result, f)
            f.write("\n")
    logger.info(f"Combined results saved to {results_jsonl_path}")

    # Initialize wandb if report_to is wandb
    if report_to and "wandb" in report_to and not wandb.run:
        wandb.init(
            # We stored run name in here
            # Load run_name from config.yaml in dump_dir
            name=args.wandb.run_name,
            project=args.wandb.project,
            entity=args.wandb.entity,
            resume="must",
        )

    plot_steering(
        combined_results,
        dump_dir / eval_run,
        report_to,
        mode="steering",
    )

    if args.evaluate.report_to == "wandb":
        log_results_to_wandb(
            dump_dir, eval_run=eval_run, infer_run=infer_run, concept_info=concept_info
        )

    # Log hyperparameter optimization plots and tradeoff information
    if wandb.run and args.evaluate.report_to == "wandb":
        # Log the optimization history
        optimization_history = {
            "hyperopt/trial_number": [],
            "hyperopt/factor": [],
            "hyperopt/score": [],
        }

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                optimization_history["hyperopt/trial_number"].append(trial.number)
                optimization_history["hyperopt/factor"].append(trial.params["factor"])
                optimization_history["hyperopt/score"].append(trial.value)

        # Log as a table
        if optimization_history["hyperopt/trial_number"]:
            optimization_df = pd.DataFrame(optimization_history)
            wandb.log(
                {
                    "hyperopt/optimization_history": wandb.Table(
                        dataframe=optimization_df
                    )
                }
            )

            # Create and log optimization plots
            try:
                # Log the optimization history plot
                optuna_fig = optuna.visualization.plot_optimization_history(study)
                wandb.log(
                    {"hyperopt/optimization_history_plot": wandb.Image(optuna_fig)}
                )
                # Log the parameter importance plot
                param_importance_fig = optuna.visualization.plot_param_importances(
                    study
                )
                wandb.log(
                    {"hyperopt/param_importance": wandb.Image(param_importance_fig)}
                )
                # Log the contour plot if there are multiple parameters
                if len(study.best_trial.params) > 1:
                    contour_fig = optuna.visualization.plot_contour(study)
                    wandb.log({"hyperopt/contour_plot": wandb.Image(contour_fig)})
                # Log the slice plot
                slice_fig = optuna.visualization.plot_slice(study)
                wandb.log({"hyperopt/slice_plot": wandb.Image(slice_fig)})
                # Log the parallel coordinate plot
                parallel_fig = optuna.visualization.plot_parallel_coordinate(study)
                wandb.log({"hyperopt/parallel_coordinate": wandb.Image(parallel_fig)})
            except Exception as e:
                logger.warning(f"Failed to create optimization plots: {e}")
            # Log the best trial information
            wandb.log(
                {
                    "hyperopt/best_factor": best_factor,
                    "hyperopt/best_score": best_score,
                    "hyperopt/n_trials": n_trials,
                }
            )

    # Log metadata about the factor selection run to help with traceability
    metadata_path = dump_dir / eval_run / "eval_metadata.json"
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "infer_run": str(infer_run),
        "eval_run": str(eval_run),
        "best_factor": best_factor,
        "best_score": best_score,
        "n_trials": n_trials,
        "factor_min": factor_min,
        "factor_max": factor_max,
        "metric_name": metric_name,
        "model_used": args.inference.factor_selection.model,
        "discrete_space": use_discrete_space,
        "discrete_steps": discrete_steps if use_discrete_space else None,
        "log_scale": use_log_scale,
        "total_execution_time": total_duration,
        "concept_ids": concept_ids if concept_ids else "all",
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Factor selection metadata saved to {metadata_path}")

    logger.info(f"Factor selection results saved to {results_path}")
    logger.info("=" * 80)
    return results


def run_inference(args: ExperimentConfig):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Dataset loading is now handled directly through HuggingFace dataset configuration
    # No need for separate metadata directory handling

    # Set up dump dir
    original_dump_dir = Path(args.dataset.dump_dir)

    # Determine inference run directory logic
    if args.inference.infer_run:
        # If a custom infer_run value is provided, always use it
        infer_run = args.inference.infer_run
    elif args.inference.run_distinct_infers:
        # Otherwise, if distinct infers is requested, use a timestamped run
        infer_run = f"inference_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
    else:
        # Otherwise, use the default "inference" directory
        infer_run = "inference"

    inference_dump_dir = Path(original_dump_dir) / infer_run

    # Only skip if NOT overwriting and the directory exists,
    # but if running distinct infers, we never skip (since the dir is always new)
    if (
        not args.inference.run_distinct_infers
        and not args.inference.overwrite_existing_infer
        and inference_dump_dir.exists()
        and any(inference_dump_dir.iterdir())
    ):
        logger.warning(
            f"Infer dump dir {inference_dump_dir} already exists and is nonempty. Skipping."
        )
        return

    inference_dump_dir.mkdir(parents=True, exist_ok=True)

    args.dataset.train_dir = original_dump_dir / "train"
    logger.info(
        f"Inferencing with following configuration:\n{dump_json(args.inference.model_dump(), indent=4)}"
    )
    set_seed(args.dataset.seed)

    # Initialize the process group
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
    except Exception as e:
        logger.error(f"Failed to initialize distributed process group: {e}")

    # Get the rank and world_size from environment variables
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set the device for this process
    device = get_and_set_device(local_rank)

    # Define common arguments for all inference functions
    _common_args = [args, rank, world_size, device, logger]
    _common_kwargs = {"infer_run": infer_run}

    if args.inference.factor_selection.enable:
        select_steering_factors(*_common_args, **_common_kwargs)
    else:
        infer_steering(*_common_args, **_common_kwargs)

    # Finalize the process group
    destroy_process_group()

    return infer_run


def clear_global_model():
    """
    Clear global model and tokenizer instances to free up memory.
    """
    global global_model_instance, global_tokenizer, global_model_name

    # Delete the model instance and tokenizer
    global_model_instance = None
    global_tokenizer = None
    global_model_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Use experiment config if it exists, otherwise use the main config
    config = cfg.experiment if hasattr(cfg, "experiment") else cfg

    # Handle pretrained config merging if needed
    if hasattr(config, "dataset") and config.dataset.dump_dir:
        pretrained_cfg_path = Path(config.dataset.dump_dir) / "config.yaml"
        if pretrained_cfg_path.exists():
            logger.info(f"Loading pretrained config from {pretrained_cfg_path}")
            pretrained_cfg = OmegaConf.load(pretrained_cfg_path)
            config = OmegaConf.merge(pretrained_cfg, config)

    config = config_to_pydantic(config, ExperimentConfig)
    infer_run = run_inference(config)
    if config.inference.run_eval:
        run_eval(config, infer_run)
    clear_global_model()


if __name__ == "__main__":
    main()
