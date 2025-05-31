import asyncio
import os
import time
from collections import namedtuple

import pandas as pd
from datasets import concatenate_datasets, load_dataset

from hypersteer.utils.constants import CHAT_MODELS, EMPTY_CONCEPT
from hypersteer.utils.helpers import get_logger
from hypersteer.utils.language_models import LanguageModel
from hypersteer.utils.model_utils import get_model_continues, get_suffix_length
from hypersteer.utils.prompt_utils import (
    continue_with_concept,
    continue_with_polysemantic_concepts,
    continue_without_concept,
    get_concept_genres,
    get_contrast_concepts,
    get_random_content,
    response_with_concept,
    response_with_polysemantic_concepts,
    response_without_concept,
)

from .base import (
    BaseDatasetFactory,
    BaseSteeringDatasetFactory,
    register_factory,
    register_training_dataset,
)

logger = get_logger(__name__)


T_PROMPT_STEERING = """You must answer the question with content \
related to %s even if it is not related to the question or it does not make sense."""


T_GENERATE_STEERING_PROMPT = """Generate a prompt to guide a language \
model in producing responses. 

Objective: 
Direct the model to include content related to %s (the concept) in its responses. 
Ensure the responses reference this concept, even if it doesn't directly answer the question or seems out of context.
Optionally, provide in-context examples to reinforce this behavior.
        
Return only the final prompt without any additional text."""


# special types for dataset
Prompt = namedtuple("Prompt", ["concept", "tag", "content"])


async def run_tasks(tasks):
    # Gather and run all provided tasks concurrently, and collect their results
    results = await asyncio.gather(*tasks)
    return results


def apply_chat_template_llama(
    example, tokenizer, suffix_length, binarize=False, output_length=None
):
    """Apply chat template for Llama models"""
    if binarize:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        nobos = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )[1:-suffix_length]
        return {"input": tokenizer.decode(nobos)}
    else:
        # For instruction tuning
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example["input"]},
        ]
        nobos = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )[1:]
        formatted_input = tokenizer.decode(nobos)

        # Handle output
        suffix_str = tokenizer.decode([tokenizer.eos_token_id])
        if output_length and len(tokenizer.tokenize(example["output"])) < output_length:
            formatted_output = example["output"] + suffix_str
        else:
            formatted_output = example["output"]

        return {"input": formatted_input, "output": formatted_output}


def apply_chat_template_generic(
    example, tokenizer, suffix_length, binarize=False, output_length=None
):
    """Apply chat template for generic models"""
    if binarize:
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        nobos = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )[1:-suffix_length]
        return {"input": tokenizer.decode(nobos)}
    else:
        # For instruction tuning
        messages = [{"role": "user", "content": example["input"]}]
        nobos = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )[1:]
        return {"input": tokenizer.decode(nobos), "output": example["output"]}


def apply_no_chat_template(example, tokenizer, binarize=False):
    """Apply no chat template for non-chat models"""
    if binarize:
        return {"input": example["input"] + example["output"]}
    else:
        return {"input": example["input"], "output": example["output"]}


def process_dataset_for_training(
    dataset,
    tokenizer,
    model_name,
    binarize=False,
    train_on_negative=False,
    output_length=None,
    max_num_of_examples=None,
    negative_example_ratio=1,
    replace_negative_description=True,
):
    """
    Process HuggingFace dataset for training using map functions.

    Args:
        dataset: HuggingFace Dataset
        tokenizer: Tokenizer instance
        model_name: Name of the model
        binarize: Whether to binarize the dataset
        train_on_negative: Whether to include negative examples in training
        output_length: Output length for chat models
        max_num_of_examples: Maximum number of examples per concept
        negative_example_ratio: Ratio of negative to positive examples
        replace_negative_description: Whether to replace negative descriptions

    Returns:
        Processed HuggingFace Dataset ready for training
    """
    suffix_length, suffix_str = get_suffix_length(tokenizer)
    is_chat_model = model_name in CHAT_MODELS if model_name else False

    # Filter positive and negative examples
    positive_dataset = dataset.filter(
        lambda x: x["output_concept"] != EMPTY_CONCEPT and x["category"] == "positive"
    )
    negative_dataset = dataset.filter(
        lambda x: x["output_concept"] == EMPTY_CONCEPT and x["category"] == "negative"
    )

    # Limit examples if specified
    if max_num_of_examples and max_num_of_examples > 0:
        positive_dataset = positive_dataset.select(
            range(min(max_num_of_examples // 2, len(positive_dataset)))
        )
        negative_dataset = negative_dataset.select(
            range(min(max_num_of_examples // 2, len(negative_dataset)))
        )

    # Handle negative example ratio
    if negative_example_ratio is not None and len(negative_dataset) > 0:
        # Group positive examples by concept_id
        concept_ids = list(set(positive_dataset["concept_id"]))

        processed_negative_examples = []
        for concept_id in concept_ids:
            concept_positive = positive_dataset.filter(
                lambda x: x["concept_id"] == concept_id
            )
            if len(concept_positive) == 0:
                continue

            positive_example_per_concept = len(concept_positive)
            negative_example_per_concept = int(
                positive_example_per_concept * negative_example_ratio
            )

            # Sample negative examples for this concept
            if negative_example_per_concept > 0 and len(negative_dataset) > 0:
                # Get the concept description from positive examples
                concept_description = concept_positive[0]["output_concept"]

                # Sample negative examples
                sampled_indices = list(
                    range(min(negative_example_per_concept, len(negative_dataset)))
                )
                concept_negative = negative_dataset.select(sampled_indices)

                # Replace description if needed
                if replace_negative_description:

                    def update_negative_example(example):
                        example["output_concept"] = concept_description
                        example["concept_id"] = concept_id
                        return example

                    concept_negative = concept_negative.map(update_negative_example)

                processed_negative_examples.append(concept_negative)

        # Combine all negative examples
        if processed_negative_examples:
            negative_dataset = concatenate_datasets(processed_negative_examples)

    # Combine datasets based on training mode
    if train_on_negative and len(negative_dataset) > 0:
        combined_dataset = concatenate_datasets([positive_dataset, negative_dataset])
    else:
        combined_dataset = positive_dataset

    # Apply chat templates
    if is_chat_model:
        if model_name == "meta-llama/Llama-3.1-8B-Instruct":

            def template_fn(x):
                return apply_chat_template_llama(
                    x, tokenizer, suffix_length, binarize, output_length
                )
        else:

            def template_fn(x):
                return apply_chat_template_generic(
                    x, tokenizer, suffix_length, binarize, output_length
                )
    else:

        def template_fn(x):
            return apply_no_chat_template(x, tokenizer, binarize)

    # Apply the template function
    processed_dataset = combined_dataset.map(template_fn, num_proc=4)

    # Add labels for binarized datasets
    if binarize:

        def add_labels(example):
            if example.get("category") == "positive":
                example["labels"] = 1
            else:
                example["labels"] = 0
            return example

        processed_dataset = processed_dataset.map(add_labels)

    return processed_dataset


@register_training_dataset("axbench")
def get_training_dataset(
    dataset_name,
    data_files=None,
    split="train",
    cache_dir=None,
    tokenizer=None,
    model_name=None,
    binarize=False,
    train_on_negative=False,
    output_length=None,
    max_num_of_examples=None,
    negative_example_ratio=1,
    replace_negative_description=True,
    select_concept_ids=None,
    max_concepts=None,
    **kwargs,
):
    """
    Load and process AxBench training dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        data_files: Optional dict specifying data files to load
        split: Dataset split to load
        cache_dir: Optional cache directory
        tokenizer: Tokenizer instance
        model_name: Name of the model
        binarize: Whether to binarize the dataset
        train_on_negative: Whether to include negative examples in training
        output_length: Output length for chat models
        max_num_of_examples: Maximum number of examples per concept
        negative_example_ratio: Ratio of negative to positive examples
        replace_negative_description: Whether to replace negative descriptions
        select_concept_ids: List of concept IDs to select
        max_concepts: Maximum number of concepts to include
        **kwargs: Additional arguments

    Returns:
        Processed HuggingFace Dataset ready for training
    """
    logger.info(f"Loading training dataset {dataset_name} with files {data_files}")

    # Load dataset from HuggingFace
    dataset = load_dataset(
        dataset_name,
        data_files=data_files,
        split=split if kwargs.get("use_split", False) else "train",
        cache_dir=cache_dir,
    )

    logger.info(f"Loaded dataset with {len(dataset)} examples")

    # Filter by concept IDs if specified
    if select_concept_ids:
        dataset = dataset.filter(
            lambda x: x["concept_id"] in select_concept_ids, num_proc=4
        )
        logger.info(f"Filtered to {len(dataset)} examples for selected concepts")

    # Limit concepts if specified
    if max_concepts:
        concept_ids = list(set(dataset["concept_id"]))
        concept_ids = [
            cid for cid in concept_ids if cid >= 0
        ]  # Filter out negative concept IDs
        concept_ids.sort()
        limited_concept_ids = concept_ids[:max_concepts]

        dataset = dataset.filter(
            lambda x: x["concept_id"] in limited_concept_ids, num_proc=4
        )
        logger.info(
            f"Limited to {len(limited_concept_ids)} concepts with {len(dataset)} examples"
        )

    # Process dataset for training
    if tokenizer is not None and model_name is not None:
        dataset = process_dataset_for_training(
            dataset=dataset,
            tokenizer=tokenizer,
            model_name=model_name,
            binarize=binarize,
            train_on_negative=train_on_negative,
            output_length=output_length,
            max_num_of_examples=max_num_of_examples,
            negative_example_ratio=negative_example_ratio,
            replace_negative_description=replace_negative_description,
        )
        logger.info(
            f"Processed dataset with {len(dataset)} examples ready for training"
        )

    return dataset


@register_factory("axbench")
class AxbenchDatasetFactory(BaseDatasetFactory):
    """AxBench dataset factory for async generating training pairs for two subspaces"""

    def __init__(
        self,
        model,
        client,
        tokenizer,
        dataset_category,
        num_of_examples,
        output_length,
        dump_dir,
        use_cache=True,
        start_concept_id=0,
        is_chat_model=True,
        include_system_prompt=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir

        # prepare lm model
        lm_model = kwargs.get("lm_model", "gpt-4o-mini")
        self.use_cache = use_cache
        self.lm_model = LanguageModel(
            lm_model,
            client,
            dump_dir,
            use_cache=use_cache,
        )
        self.seed = kwargs.get("seed", 42)
        self.logger = kwargs.get("logger", logger)

        # load seed sentences and instructions from HuggingFace
        self.seed_sentences = load_dataset("hypersteer/seed_sentences", split="train")
        self.seed_instructions = load_dataset(
            "hypersteer/seed_instructions", split="train"
        )
        self.dataset_category = dataset_category
        self.overwrite_inference_data_dir = kwargs.get(
            "overwrite_inference_data_dir", None
        )
        if self.overwrite_inference_data_dir is not None and os.path.exists(
            self.overwrite_inference_data_dir
        ):
            # load pre-generated data
            self.pregenerated_inference_df = pd.read_parquet(
                os.path.join(
                    self.overwrite_inference_data_dir, "latent_eval_data.parquet"
                )
            )
            self.logger.warning(
                f"Loaded pre-generated data from {self.overwrite_inference_data_dir}."
            )

        # create a shared genre-based negative pools all at once
        if start_concept_id == 0 and not kwargs.get("is_inference", False):
            per_category_n = int(num_of_examples // 2)
            start = time.time()
            self.logger.warning(
                "Creating genre-based and shared negative examples for all concepts."
            )
            random_examples = []
            for genre in ["text", "math", "code"]:
                random_content = get_random_content(
                    self.seed_sentences
                    if self.dataset_category == "continuation"
                    else self.seed_instructions,
                    tokenizer=tokenizer,
                    count=per_category_n,
                    genres=[genre],
                    concepts=["random"],
                    length=None,
                    split="train",
                )
                concept_outputs = get_model_continues(
                    self.model,
                    self.tokenizer,
                    random_content["random"],
                    max_new_tokens=int(output_length * 1.5),
                    is_chat_model=is_chat_model,
                    include_system_prompt=include_system_prompt,
                )
                for i, (prompt, output) in enumerate(
                    zip(random_content["random"], concept_outputs)
                ):
                    random_examples += [
                        [
                            prompt,
                            output,
                            EMPTY_CONCEPT,
                            genre,
                            "negative",
                            self.dataset_category,
                        ]
                    ]
            self.negative_df = pd.DataFrame(
                random_examples,
                columns=[
                    "input",
                    "output",
                    "output_concept",
                    "concept_genre",
                    "category",
                    "dataset_category",
                ],
            )
            self.negative_df["concept_id"] = -1
            self.logger.warning(
                f"Finished creating negative examples in {round(time.time() - start, 3)} sec."
            )

    def load_hf_dataset(
        self, dataset_name, data_files=None, split="train", cache_dir=None
    ):
        """
        Load dataset from HuggingFace Hub.

        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            data_files: Optional dict specifying data files to load
            split: Dataset split to load
            cache_dir: Optional cache directory

        Returns:
            Loaded HuggingFace Dataset
        """
        logger.debug(f"Loading dataset {dataset_name} with files {data_files}")

        dataset = load_dataset(
            dataset_name,
            data_files=data_files,
            split=split,
            cache_dir=cache_dir,
        )

        logger.debug(f"Loaded dataset with {len(dataset)} examples")
        return dataset

    def save_cache(self):
        """Save the language model cache before exiting"""
        self.lm_model.save_cache()

    def reset_stats(self):
        """Reset API costs"""
        if self.use_cache:
            self.lm_model.dump()
        self.lm_model.stats.print_report()
        self.lm_model.stats.reset()

    def prepare_genre_concepts(self, concepts, **kwargs):
        start = time.time()
        tasks = []

        # prepare genres if needed
        concept_genres_map = kwargs.get("concept_genres_map", None)
        if concept_genres_map is None:
            logger.warning("Creating genre for the inputs (not provided).")
            genre_task = get_concept_genres(
                self.lm_model, concepts, api_tag=kwargs.get("api_tag", "")
            )
            tasks.append(genre_task)

        # run tasks
        res = asyncio.run(run_tasks(tasks))
        concept_genres_map = res[0]

        # log
        logger.warning(f"Init finished in {round(time.time() - start, 3)} sec.")
        return concept_genres_map

    def prepare_concepts(self, concepts, **kwargs):
        if self.overwrite_inference_data_dir is not None and os.path.exists(
            self.overwrite_inference_data_dir
        ):
            self.logger.warning("Using pre-generated metadata.")
            return {}, {}

        start = time.time()
        tasks = []

        # contrast concepts
        logger.warning("Creating contrast concepts for the inputs.")
        contrast_task = get_contrast_concepts(
            self.lm_model,
            concepts,
            kwargs.get("contrast_concepts_map", None),
            api_tag=kwargs.get("api_tag", ""),
        )
        tasks.append(contrast_task)

        # prepare genres if needed
        concept_genres_map = kwargs.get("concept_genres_map", None)
        if concept_genres_map is None:
            logger.warning("Creating genre for the inputs (not provided).")
            genre_task = get_concept_genres(
                self.lm_model, concepts, api_tag=kwargs.get("api_tag", "")
            )
            tasks.append(genre_task)

        # run tasks
        res = asyncio.run(run_tasks(tasks))
        contrast_concepts_map = res[0]
        if len(res) > 1:
            concept_genres_map = res[1]

        # log
        for concept in concepts:
            logger.warning(
                f"Found {len(contrast_concepts_map[concept])} contrast concept(s) for concept: {concept}."
            )
        logger.warning(f"Init finished in {round(time.time() - start, 3)} sec.")
        return concept_genres_map, contrast_concepts_map

    def create_imbalance_eval_df(self, subset_n, factor=100):
        # we dont care about concept, there is only one unified imbalanced negative set.
        self.logger.warning(
            "Using pre-generated data for imbalanced eval dataset "
            "(positive examples only occupy less than 1% of the dataset)."
        )
        if factor is None:
            factor = 100
        negative_n_upsamples = int(
            subset_n * factor
        )  # 100x more negative examples than positive ones.
        # we sample negative_n_upsamples from other concepts.
        negative_df = self.pregenerated_inference_df[
            self.pregenerated_inference_df["category"] == "negative"
        ].copy()
        negative_df = negative_df.sample(negative_n_upsamples, random_state=self.seed)
        negative_df["output_concept"] = EMPTY_CONCEPT
        # overwrite negative df fields to be compatible.
        concept_df = negative_df
        return concept_df

    def create_eval_df(
        self,
        concepts,
        subset_n,
        concept_genres_map,
        train_contrast_concepts_map,
        eval_contrast_concepts_map,
        mode="balance",
        **kwargs,
    ):
        """category: positive, negative, hard negative"""

        if self.overwrite_inference_data_dir is not None and os.path.exists(
            self.overwrite_inference_data_dir
        ):
            if mode == "balance":
                self.logger.warning("Using pre-generated data.")
                concept_df = self.pregenerated_inference_df[
                    self.pregenerated_inference_df["concept_id"]
                    == kwargs.get("concept_id")
                ].copy()
                if len(concept_df) < subset_n * 2:
                    self.logger.warning(
                        f"Number of examples does not meet the requirement. {len(concept_df)} < {subset_n * 2}"
                    )
            else:
                raise ValueError(f"Unknown mode: {mode}")
            return concept_df

        # start logging
        start = time.time()
        self.logger.warning("Creating dataframe.")

        # init vars
        lm_model, _model, tokenizer = self.lm_model, self.model, self.tokenizer
        output_length = kwargs.get("output_length", 32)

        all_examples = []
        concepts_random_content = get_random_content(
            self.seed_sentences
            if self.dataset_category == "continuation"
            else self.seed_instructions,
            tokenizer=tokenizer,
            count=subset_n * 2,
            genres=[concept_genres_map[concepts[0]][0]],
            concepts=concepts,
            length=None,
            split="test",
        )

        genre_balanced_random_content = {concept: [] for concept in concepts}
        genre_subset_n = {"math": int(subset_n * 0.15), "code": int(subset_n * 0.15)}
        genre_subset_n["text"] = (
            subset_n - genre_subset_n["math"] - genre_subset_n["code"]
        )
        genre_concept_map = {concept: [] for concept in concepts}
        for genre in ["text", "math", "code"]:
            genre_concepts_random_content = get_random_content(
                self.seed_sentences
                if self.dataset_category == "continuation"
                else self.seed_instructions,
                tokenizer=tokenizer,
                count=genre_subset_n[genre],
                genres=[genre],
                concepts=concepts,
                length=None,
                split="test",
            )
            for concept in concepts:
                genre_concept_map[concept] += [genre] * genre_subset_n[genre]
                genre_balanced_random_content[concept] += genre_concepts_random_content[
                    concept
                ]

        if self.dataset_category == "continuation":
            functors = [
                continue_with_concept,
                continue_without_concept,
                continue_with_polysemantic_concepts,
            ]
        elif self.dataset_category == "instruction":
            functors = [
                response_with_concept,
                response_without_concept,
                response_with_polysemantic_concepts,
            ]
        else:
            raise ValueError(f"Unknown dataset category: {self.dataset_category}")

        for idx, concept in enumerate(concepts):
            # positive continuation / instruction
            continue_task = functors[0](
                self.lm_model,
                self.tokenizer,
                concepts=[concept] * len(concepts_random_content[concept][:subset_n]),
                content=concepts_random_content[concept][:subset_n],
                length=output_length,
            )
            concept_outputs = asyncio.run(run_tasks([continue_task]))[0]
            for i, (prompt, output) in enumerate(
                zip(concepts_random_content[concept][:subset_n], concept_outputs)
            ):
                all_examples += [
                    [
                        prompt,
                        output,
                        concept,
                        concept_genres_map[concepts[0]][0],
                        "positive",
                        self.dataset_category,
                    ]
                ]

            # negative continuation / instruction (genre balanced based on global genre distribution)
            continue_task = functors[1](
                self.lm_model,
                self.tokenizer,
                content=genre_balanced_random_content[concept],
                concepts=[concept] * len(genre_balanced_random_content[concept]),
                length=output_length,
            )
            concept_outputs = asyncio.run(run_tasks([continue_task]))[0]
            for i, (negative_genre, prompt, output) in enumerate(
                zip(
                    genre_concept_map[concept],
                    genre_balanced_random_content[concept],
                    concept_outputs,
                )
            ):
                all_examples += [
                    [
                        prompt,
                        output,
                        concept,
                        negative_genre,
                        "negative",
                        self.dataset_category,
                    ]
                ]

            # hard negative continuation / instruction
            splits = [
                ("hard negative", eval_contrast_concepts_map[concept]),
            ]
            eval_tasks = []
            tags = []
            for label, polysemantic_meanings in splits:
                if len(polysemantic_meanings) != 0:
                    polysemantic_random_content = concepts_random_content[concept][
                        subset_n : subset_n + len(polysemantic_meanings)
                    ]
                    eval_tasks.append(
                        functors[2](
                            client=lm_model,
                            tokenizer=tokenizer,
                            polysemantic_concepts=polysemantic_meanings,
                            concept=concept,
                            content=polysemantic_random_content,
                            length=output_length,
                        )
                    )
                    tags.append((label, concept, idx))
            hard_negative_eval_content = asyncio.run(run_tasks(eval_tasks))
            for (tag, concept, idx), eval_content in zip(
                tags, hard_negative_eval_content
            ):
                all_examples += [
                    [
                        content[0],
                        content[2],
                        "//".join(content[1]),
                        concept_genres_map[concepts[0]][0],
                        tag,
                        self.dataset_category,
                    ]
                    for content in eval_content[1]
                ]

        # update the column definitions of the DataFrame
        df = pd.DataFrame(
            all_examples,
            columns=[
                "input",
                "output",
                "output_concept",
                "concept_genre",
                "category",
                "dataset_category",
            ],
        )
        self.logger.warning(
            f"Finished creating current dataframe in {round(time.time() - start, 3)} sec."
        )
        return df

    def create_train_df(self, concept, n, concept_genres_map, **kwargs):
        _lm_model, _model, tokenizer = self.lm_model, self.model, self.tokenizer

        start = time.time()
        self.logger.warning("Creating dataframe.")
        all_examples = []

        output_length = kwargs.get("output_length", 32)

        functors = []
        if self.dataset_category == "continuation":
            functors = [continue_with_concept, continue_without_concept]
        else:
            functors = [response_with_concept, response_without_concept]

        # random sentence or instruction
        genre = concept_genres_map[concept][0]
        concepts_random_content = get_random_content(
            self.seed_sentences
            if self.dataset_category == "continuation"
            else self.seed_instructions,
            tokenizer=tokenizer,
            count=n,
            genres=[genre],
            concepts=[concept],
            length=None,
            split="train",
        )
        per_category_n = int(n // 2)

        # positive continuation / instruction
        continue_task = functors[0](
            self.lm_model,
            self.tokenizer,
            concepts=[concept] * len(concepts_random_content[concept][:per_category_n]),
            content=concepts_random_content[concept][:per_category_n],
            length=output_length,
        )
        concept_outputs = asyncio.run(run_tasks([continue_task]))[0]
        for i, (prompt, output) in enumerate(
            zip(concepts_random_content[concept][:per_category_n], concept_outputs)
        ):
            all_examples += [
                [prompt, output, concept, genre, "positive", self.dataset_category]
            ]

        # update the column definitions of the DataFrame
        df = pd.DataFrame(
            all_examples,
            columns=[
                "input",
                "output",
                "output_concept",
                "concept_genre",
                "category",
                "dataset_category",
            ],
        )
        self.logger.warning(
            f"Finished creating current dataframe in {round(time.time() - start, 3)} sec."
        )
        return df

    def create_dpo_df(self, existing_df, **kwargs):
        _lm_model, _model, _tokenizer = self.lm_model, self.model, self.tokenizer
        start = time.time()
        self.logger.warning("Creating dataframe.")
        batch_size = kwargs.get("batch_size", 8)
        output_length = kwargs.get("output_length", 32)
        is_chat_model = kwargs.get("is_chat_model", True)
        include_system_prompt = kwargs.get("include_system_prompt", False)

        positive_df = existing_df[existing_df["category"] == "positive"]
        positive_prompts = positive_df["input"].tolist()

        losing_outputs = get_model_continues(
            self.model,
            self.tokenizer,
            positive_prompts,
            max_new_tokens=int(output_length * 1.5),
            is_chat_model=is_chat_model,
            include_system_prompt=include_system_prompt,
            batch_size=batch_size,
            verbose=True,
        )
        positive_df["losing_output"] = losing_outputs
        positive_df["losing_output_concept"] = EMPTY_CONCEPT

        self.logger.warning(
            f"Finished creating current dataframe in {round(time.time() - start, 3)} sec."
        )
        return positive_df


async def get_steering_prompts(client, concepts):
    prompts = []
    for concept in concepts:
        prompts += [T_GENERATE_STEERING_PROMPT % (concept)]
    responses = await client.chat_completions("get_steering_prompts", prompts)
    return responses


@register_factory("axbench_steering")
class AxbenchSteeringDatasetFactory(BaseSteeringDatasetFactory):
    def __init__(
        self,
        tokenizer,
        dump_dir,
        has_prompt_steering=False,
        master_data_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        if kwargs.get("lm_client", None):
            self.lm_model = LanguageModel(
                kwargs.get("lm_model", "gpt-4o-mini"),
                kwargs["lm_client"],
                dump_dir,
                use_cache=True,
                master_data_dir=master_data_dir,
            )
        self.has_prompt_steering = has_prompt_steering

    async def augment_train_df_with_steered_prompts(self, train_df, concepts):
        """Augment AlpacaEval seeded training dataset with steered prompts"""
        steering_prompts = await get_steering_prompts(self.lm_model, concepts)
        steering_prompts = [prompt.strip() for prompt in steering_prompts]

        # Add steered prompts to matching concepts in train_df
        train_df["steered_input"] = train_df["input"]  # Initialize with original input
        for concept, steering_prompt in zip(concepts, steering_prompts):
            # Find rows where output_concept matches current concept
            matching_rows = train_df[train_df["output_concept"] == concept].copy()
            # Add steered prompt format to each matching row
            for idx, row in matching_rows.iterrows():
                steered_prompt = f" {steering_prompt}\n\nQuestion: {row['input']}"
                train_df.at[idx, "steered_input"] = steered_prompt

        return train_df

    def create_eval_df(
        self,
        concepts,
        subset_n,
        steering_factors,
        steering_datasets,
        concept_id,
        steering_model_name,
    ):
        for dataset_name in steering_datasets:
            if dataset_name == "OUATPrefix":
                # we generate subset_n * n_steering_factors examples for OUATPrefix.
                # OUATPrefix is basically a prefix dataset.
                # "Once upon a time, " is the prefix, and there is no other labels.
                # we also need to label these in groups:
                # each one of subset_n group has the same group id.
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        for factor in steering_factors:
                            all_examples += [
                                [
                                    dataset_name,
                                    idx,
                                    concept,
                                    i,
                                    factor,
                                    "Once upon a time, there was a ",
                                ]
                            ]
                df = pd.DataFrame(
                    all_examples,
                    columns=[
                        "dataset_name",
                        "concept_id",
                        "input_concept",
                        "input_id",
                        "factor",
                        "input",
                    ],
                )
                return df
            elif dataset_name == "AlpacaEval":
                # load alpaca eval dataset from HuggingFace
                alpaca_eval_df = load_dataset(
                    "tatsu-lab/alpaca_eval", split="eval", trust_remote_code=True
                ).to_pandas()

                # get gpt-4o boosted steering prompts.
                if self.has_prompt_steering:
                    steering_prompts = asyncio.run(
                        get_steering_prompts(self.lm_model, concepts)
                    )
                    steering_prompts = [prompt.strip() for prompt in steering_prompts]
                else:
                    # simply just a dummy one since no method is going to use it.
                    steering_prompts = [
                        T_PROMPT_STEERING % (concept) for concept in concepts
                    ]
                all_examples = []
                for idx, concept in enumerate(concepts):
                    # sample a random example from alpaca eval dataset.
                    sampled_prompts = alpaca_eval_df.sample(
                        subset_n, random_state=int(concept_id)
                    )["instruction"].tolist()
                    for i in range(subset_n):
                        sampled_prompt = sampled_prompts[i]
                        # for prompt-based steering ONLY.
                        steering_prompt = (
                            steering_prompts[idx]
                            if steering_prompts[idx] != ""
                            else T_PROMPT_STEERING % (concept)
                        )
                        steered_prompt = (
                            f" {steering_prompt}\n\nQuestion: {sampled_prompt}"
                        )
                        if steering_model_name == "meta-llama/Llama-3.1-8B-Instruct":
                            formatted_steered_prompt = (
                                self.tokenizer.apply_chat_template(
                                    [
                                        {
                                            "role": "system",
                                            "content": "You are a helpful assistant.",
                                        },
                                        {"role": "user", "content": steered_prompt},
                                    ],
                                    tokenize=True,
                                    add_generation_prompt=True,
                                )[1:]
                            )  # get rid of bos token
                            formatted_steered_prompt = self.tokenizer.decode(
                                formatted_steered_prompt
                            )
                            # apply the tokenizer chat format to the prompt.
                            formatted_prompt = self.tokenizer.apply_chat_template(
                                [
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant.",
                                    },
                                    {"role": "user", "content": sampled_prompt},
                                ],
                                tokenize=True,
                                add_generation_prompt=True,
                            )[1:]  # get rid of bos token
                            formatted_prompt = self.tokenizer.decode(formatted_prompt)
                        else:
                            formatted_steered_prompt = (
                                self.tokenizer.apply_chat_template(
                                    [{"role": "user", "content": steered_prompt}],
                                    tokenize=True,
                                    add_generation_prompt=True,
                                )[1:]
                            )  # get rid of bos token
                            formatted_steered_prompt = self.tokenizer.decode(
                                formatted_steered_prompt
                            )
                            # apply the tokenizer chat format to the prompt.
                            formatted_prompt = self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": sampled_prompt}],
                                tokenize=True,
                                add_generation_prompt=True,
                            )[1:]  # get rid of bos token
                            formatted_prompt = self.tokenizer.decode(formatted_prompt)

                        for factor in steering_factors:
                            all_examples += [
                                [
                                    dataset_name,
                                    idx,
                                    concept,
                                    i,
                                    factor,
                                    sampled_prompt,
                                    formatted_steered_prompt,
                                    formatted_prompt,
                                ]
                            ]
                df = pd.DataFrame(
                    all_examples,
                    columns=[
                        "dataset_name",
                        "concept_id",
                        "input_concept",
                        "input_id",
                        "factor",
                        "original_prompt",
                        "steered_input",
                        "input",
                    ],
                )
                return df
            elif (
                dataset_name == "AlpacaEval_Suppress"
                or dataset_name == "AlpacaEval_Synergy"
            ):
                # load alpaca eval dataset.
                assert self.lm_model is not None, (
                    "Language model is required for AlpacaEval."
                )
                alpaca_eval_path = os.path.join(
                    self.lm_model.dump_dir, "alpaca_eval.json"
                )
                alpaca_eval_df = pd.read_json(alpaca_eval_path)
                common_steering_factors = steering_factors
                if dataset_name == "AlpacaEval_Suppress":
                    common_steering_factors = [
                        f * -1.0 for f in common_steering_factors
                    ]
                # get gpt-4o boosted steering prompts.
                steering_prompts = asyncio.run(
                    get_steering_prompts(self.lm_model, concepts)
                )
                steering_prompts = [prompt.strip() for prompt in steering_prompts]
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        # sample a random example from alpaca eval dataset.
                        sampled_prompt = alpaca_eval_df.sample(1)[
                            "instruction"
                        ].tolist()[0]
                        # for prompt-based steering ONLY.
                        steering_prompt = (
                            steering_prompts[idx]
                            if steering_prompts[idx] != ""
                            else T_PROMPT_STEERING % (concept)
                        )
                        steered_prompt = (
                            f" {steering_prompt}\n\nQuestion: {sampled_prompt}"
                        )
                        formatted_steered_prompt = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": steered_prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for factor in common_steering_factors:
                            all_examples += [
                                [
                                    dataset_name,
                                    idx,
                                    concept,
                                    i,
                                    factor,
                                    sampled_prompt,
                                    formatted_steered_prompt,
                                ]
                            ]
                df = pd.DataFrame(
                    all_examples,
                    columns=[
                        "dataset_name",
                        "concept_id",
                        "input_concept",
                        "input_id",
                        "factor",
                        "original_prompt",
                        "input",
                    ],
                )
                return df
            else:
                # not implemented yet.
                raise NotImplementedError(
                    f"Steering dataset {dataset_name} not implemented."
                )
