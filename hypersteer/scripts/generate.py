# adapted mostly from axbench/scripts/generate.py

import atexit
import csv
import glob
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import httpx
import hydra
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import AsyncOpenAI
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from hypersteer.data.base import get_dataset_factory
from hypersteer.utils.configs import (
    GenerateConfig,
    config_to_pydantic,
)
from hypersteer.utils.constants import CHAT_MODELS
from hypersteer.utils.helpers import get_current_device

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)

model_name_map = {
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "generate_state.pkl"
METADATA_FILE = "metadata.jsonl"


def load_concepts(dump_dir):
    sae_concepts = []
    if ".txt" in dump_dir:
        with open(dump_dir) as file:
            concepts = [line.strip() for line in file.readlines()]
        if concepts[0].startswith("http://") or concepts[0].startswith("https://"):
            logger.warning("Detect external links. Pull concept info from the link.")
            for concept in concepts:
                if "www.neuronpedia.org" not in concept:
                    raise ValueError(f"Pulling from {concept} is not supported.")
                sae_path = concept.split("https://www.neuronpedia.org/")[-1]
                sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
                headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
                response = requests.get(sae_url, headers=headers).json()
                explanation = response["explanations"][0]["description"]
                sae_concepts += [explanation.strip()]
            return sae_concepts, concepts
        return concepts, ["null"] * len(concepts)
    elif ".csv" in dump_dir:
        # for csv, then the format is <concept>,<url>
        # no http connection is needed
        concepts = []
        with open(dump_dir) as file:
            reader = csv.reader(file)
            for row in reader:
                sae_concepts += [row[0]]
                concepts += [row[1]]
        return sae_concepts, concepts
    elif ".json" in dump_dir:
        concepts = []
        # this must be a neuropedia export.
        with open(dump_dir) as file:
            json_concepts = json.load(file)
        seen_index = set()
        for concept in json_concepts:
            model = concept["modelId"]
            sae_model = concept["layer"]
            subspace_id = concept["index"]
            if subspace_id in seen_index:
                continue  # if there are multiple descriptions, we only take the first one.
            seen_index.add(subspace_id)
            sae_concepts += [concept["description"].strip()]
            concepts += [
                f"https://www.neuronpedia.org/{model}/{sae_model}/{subspace_id}"
            ]
        return sae_concepts, concepts
    else:
        raise ValueError(f"Unsupported file type: {dump_dir}.")


def save_df_to_parquet_safely(df, final_path):
    import os
    import tempfile

    # Create temporary file in the same directory as the target
    dirname = os.path.dirname(os.path.abspath(final_path))
    with tempfile.NamedTemporaryFile(
        delete=False, dir=dirname, suffix=".parquet.tmp"
    ) as tmp:
        temp_path = tmp.name
        try:
            # Write to temporary file first
            df.to_parquet(temp_path, index=False)
            # Ensure data is written to disk
            os.fsync(tmp.fileno())
        except Exception as e:
            os.unlink(temp_path)  # Clean up temp file
            raise e
    try:
        # Atomic rename operation
        os.rename(temp_path, final_path)
    except Exception as e:
        os.unlink(temp_path)  # Clean up temp file
        raise e


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    with open(Path(metadata_path) / METADATA_FILE) as f:
        for line in f:
            data = json.loads(line)
            concept, ref = data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": data["concept_id"],
            }
            metadata += [flatten_data]  # Return the metadata as is
    return metadata


def save(
    dump_dir,
    state,
    concept_id,
    concept,
    concept_genres_map,
    ref,
    partition,
    current_df,
    dataset_factory,
):
    """
    Save the current state, metadata, and DataFrame using Parquet format.
    """
    # Save state
    state_path = os.path.join(dump_dir, STATE_FILE)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    # Save metadata
    metadata_path = os.path.join(dump_dir, METADATA_FILE)
    metadata_entry = {
        "concept_id": concept_id,
        "concept": concept,
        "ref": ref,
        "concept_genres_map": concept_genres_map,
    }
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata_entry) + "\n")

    # Save DataFrame using Parquet
    rotation_freq = 500
    file_index = concept_id // rotation_freq
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        # first time cache, we need to add global negative examples.
        if concept_id == 0:
            combined_df = pd.concat(
                [dataset_factory.negative_df, current_df], ignore_index=True
            )
        else:
            combined_df = current_df
    save_df_to_parquet_safely(combined_df, df_path)


def load_state(dump_dir):
    """
    Load the state from a file if it exists.

    Args:
        dump_dir (str): The directory to load the state file from.

    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    state_path = os.path.join(Path(dump_dir), STATE_FILE)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            return state
    return None


def generate_training(args, generate_args):
    dump_dir = args.dump_dir
    dump_dir = Path(dump_dir) / "generate"
    dump_dir.mkdir(parents=True, exist_ok=True)

    concept_path = generate_args.concept_path
    num_of_examples = generate_args.num_of_examples
    max_concepts = generate_args.max_concepts

    # Load and optionally shuffle concepts
    set_seed(generate_args.seed)
    all_concepts, all_refs = load_concepts(concept_path)

    # Limit the number of concepts if specified
    if max_concepts is not None:
        combined = list(zip(all_concepts, all_refs))
        random.shuffle(combined)
        all_concepts, all_refs = zip(*combined)
        all_concepts = list(all_concepts)[:max_concepts]
        all_refs = list(all_refs)[:max_concepts]

    concepts = list(zip(all_concepts, all_refs))

    # Load the state if it exists.
    state = load_state(dump_dir) if not generate_args.ignore_generate_state else None
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    if start_concept_id >= len(concepts):
        logger.warning("Datasets for all concepts have been generated. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Load lm and tokenizer.
    model_name = model_name_map[all_refs[0].split("/")[3]]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    is_chat_model = True if model_name in CHAT_MODELS else False
    include_system_prompt = (
        True if model_name == "meta-llama/Llama-3.1-8B-Instruct" else False
    )
    model = model.to(get_current_device())

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        # raw llama3
        logger.info("Adding special padding token to tokenizer")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    # Init the dataset factory.
    dataset_factory = get_dataset_factory(
        "axbench",
        model,
        client,
        tokenizer,
        generate_args.dataset_category,
        num_of_examples,
        generate_args.output_length,
        dump_dir,
        use_cache=generate_args.lm_use_cache,
        master_data_dir=generate_args.master_data_dir,
        seed=generate_args.seed,
        lm_model=generate_args.lm_model,
        start_concept_id=start_concept_id,
        is_chat_model=is_chat_model,
        include_system_prompt=include_system_prompt,
        batch_size=generate_args.batch_size,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    progress_bar = tqdm(
        range(start_concept_id, len(concepts)), desc="Processing concept"
    )
    only_one_concept = True if len(concepts) == 1 else False
    data_concept_id = start_concept_id
    for concept_id in progress_bar:
        concept, ref = concepts[concept_id]
        logger.info(f"Generating training data for concept: {concept}")

        # prepare concept related data.
        concept_genres_map = dataset_factory.prepare_genre_concepts([concept])
        # generate with retry mechanism.
        # try:
        current_df = dataset_factory.create_train_df(
            concept,
            num_of_examples,
            concept_genres_map,
            output_length=generate_args.output_length,
            current_concept_id=data_concept_id,
            only_one_concept=only_one_concept,
            is_generation=True,
        )
        current_df["concept_id"] = data_concept_id
        # except Exception as e:
        #     logger.warning(f"Failed to create training data for group {concept_id}: {e}")
        #     continue # continue to the next group.

        # Save the generated DataFrame, metadata, and current state
        save(
            dump_dir,
            {"concept_id": concept_id + 1},
            data_concept_id,
            concept,
            concept_genres_map,
            ref,
            "train",
            current_df,
            dataset_factory,
        )
        data_concept_id += 1

    # save as a combined ds loadable in hf format
    intermediate_files = glob.glob(os.path.join(dump_dir, "train_data_*.parquet"))
    if intermediate_files:
        final_file_path = os.path.join(dump_dir, "train_data.parquet")
        all_train_data = pd.concat(
            [pd.read_parquet(path) for path in intermediate_files],
            ignore_index=True,
        )
        save_df_to_parquet_safely(all_train_data, final_file_path)

        # Clean up intermediate per-concept parquet files (exclude the final file)
        for path in intermediate_files:
            if path != final_file_path:  # Don't remove the final combined file
                os.remove(path)
                logger.debug(f"Removed intermediate file: {path}")

    logger.info("Finished creating training dataset.")


def save_preference(dump_dir, concept_id, partition, current_df):
    # This function saves DataFrames per partition
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Save DataFrame using Parquet
    rotation_freq = 500
    file_index = concept_id // rotation_freq if concept_id != -1 else 0
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_train_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_train_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    save_df_to_parquet_safely(combined_df, df_path)


def save_state_preference(dump_dir, state, partition):
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def load_state_preference(dump_dir, partition):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}/generate", f"{partition}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def generate_preference_training(args, generate_args):
    dump_dir = args.dump_dir
    dump_dir = Path(dump_dir) / "generate"
    generate_args.data_dir = f"{args.dump_dir}/generate"
    # check the generate directory exists.
    if not os.path.exists(dump_dir):
        raise ValueError(f"Generate directory does not exist: {dump_dir}")
    # check the train_data.parquet exists.
    if not os.path.exists(os.path.join(dump_dir, "train_data.parquet")):
        raise ValueError(
            f"Train data does not exist: {os.path.join(dump_dir, 'train_data.parquet')}"
        )

    num_of_examples = generate_args.num_of_examples

    # Load and optionally shuffle concepts
    set_seed(int(generate_args.seed))

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )

    # Create a console handler and set its formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    # Optionally, create a file handler per rank
    """
    log_file = f'log_rank_{rank}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    """
    data_dir = generate_args.data_dir
    num_of_examples = generate_args.preference_num_of_examples
    metadata = load_metadata_flatten(data_dir)
    # Get list of all concept_ids
    concept_ids = list(range(len(metadata)))
    concepts = [metadata[i]["concept"] for i in concept_ids]

    # Load the state if it exists.
    state = (
        load_state_preference(generate_args.dump_dir, "preference")
        if not generate_args.ignore_generate_state
        else None
    )
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    if start_concept_id >= len(concept_ids):
        logger.warning("Datasets for all concepts have been generated. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Init the dataset factory.
    dataset_factory = get_dataset_factory(
        generate_args.dataset_type,
        client,
        None,
        generate_args.dataset_category,
        num_of_examples,
        int(generate_args.output_length),
        dump_dir,
        use_cache=generate_args.lm_use_cache,
        master_data_dir=generate_args.master_data_dir,
        seed=int(generate_args.seed),
        lm_model=generate_args.lm_model,
        start_concept_id=start_concept_id,
        is_inference=True,
        is_preference=True,
        concepts=concepts,
        # disable_local_model=generate_args.disable_local_model,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    # get negative and do nothing on them just renaming.
    existing_df = pd.read_parquet(os.path.join(dump_dir, "train_data.parquet"))
    existing_df = existing_df.rename(
        columns={
            "output": "winning_output",
        }
    )

    model, tokenizer = None, None
    if args.keep_orig_axbench_format:
        # we need to load the model and the tokenizer back in this case to generate
        # the same data distribution as AxBench original to keep the comparison fair.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16
        )
        model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
        tokenizer.padding_side = "right"

    progress_bar = tqdm(
        range(start_concept_id, len(metadata)), desc="Processing concept"
    )
    preference_dfs = []
    for start_idx in progress_bar:
        concept_id = metadata[start_idx]["concept_id"]
        concept = metadata[start_idx]["concept"]
        logger.info(f"Generating DPO data for concept: {concept}")
        current_df = existing_df[existing_df["concept_id"] == concept_id].copy()
        preference_df = dataset_factory.create_preference_df(
            current_df,
            output_length=int(args.output_length),
            batch_size=int(args.inference_batch_size),
            model=model,
            tokenizer=tokenizer,
            keep_orig_axbench_format=args.keep_orig_axbench_format,
            steer_data_type=args.steer_data_type,
        )

        preference_dfs.append(preference_df)
        save_preference(dump_dir, concept_id, "preference", preference_df)
        logger.warning(
            f"Saved preference dataset for concept {concept_id} to preference_train_data.parquet"
        )
        # After processing, save state
        current_state = {"concept_id": concept_id}
        save_state_preference(dump_dir, current_state, "preference")

    # save as a combined ds loadable in hf format
    intermediate_files = glob.glob(
        os.path.join(dump_dir, "preference_train_data_*.parquet")
    )
    if intermediate_files:
        final_file_path = os.path.join(dump_dir, "preference_train_data.parquet")
        all_preference_data = pd.concat(
            [pd.read_parquet(path) for path in intermediate_files],
            ignore_index=True,
        )
        save_df_to_parquet_safely(all_preference_data, final_file_path)

        # Clean up intermediate per-concept parquet files
        for path in intermediate_files:
            os.remove(path)
            logger.debug(f"Removed intermediate file: {path}")

    logger.info("Finished creating preference dataset.")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    config = cfg.experiment if hasattr(cfg, "experiment") else cfg
    generate_config = config_to_pydantic(config.generate, GenerateConfig)

    logger.warning("Generating datasets with the following configuration:")
    logger.warning(generate_config)

    if generate_config.mode == "training":
        generate_training(config, generate_config)
    elif generate_config.mode == "preference_training":
        generate_preference_training(config, generate_config)
    else:
        raise ValueError(f"Invalid mode: {generate_config.mode}")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    load_dotenv(override=True)
    main()
