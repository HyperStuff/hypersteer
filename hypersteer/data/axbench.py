import asyncio
import os
import time
from collections import namedtuple

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm.auto import tqdm

from hypersteer.utils.constants import EMPTY_CONCEPT
from hypersteer.utils.helpers import get_logger
from hypersteer.utils.language_models import LanguageModel
from hypersteer.utils.model_utils import get_model_continues, get_suffix_length
from hypersteer.utils.prompt_utils import (
    continue_with,
    continue_with_concept,
    continue_without_concept,
    get_concept_genres,
    get_random_content,
    response_with,
    response_with_concept,
    response_without_concept,
)

from .base import (
    BaseDatasetFactory,
    register_factory,
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
    concept_genres_map,
    tokenizer,
    model_name,
    binarize=False,
    train_on_negative=False,
    output_length=None,
    max_num_of_examples=None,
    negative_example_ratio=1,
    replace_negative_description=True,
    is_chat_model=False,
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
        for concept_id in tqdm(
            concept_ids, desc="Processing negative examples", disable=len(concept_ids) < 5
        ):
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


def get_seed_sentences_dataset():
    """
    Returns a DatasetDict with text, math, and code seed sentences (train/test splits).
    """
    # text data
    wikisum_ds = load_dataset("zhengxuanzenwu/wikitext-2-split-128")
    text_train = [ex["text"] for ex in wikisum_ds["train"]]
    ag_ds = load_dataset("fancyzhx/ag_news")
    text_test = [ex["text"] for ex in ag_ds["train"]]
    # math data
    gsm_ds = load_dataset("openai/gsm8k", "main")
    math_train = [ex["answer"] for ex in gsm_ds["train"]]
    comp_ds = load_dataset("qwedsacf/competition_math")
    math_test = [ex["problem"] for ex in comp_ds["train"]]
    # code data
    code_ds = load_dataset("christopher/rosetta-code")
    code_all = [ex["code"][:500] for ex in code_ds["train"] if len(ex["code"]) > 500]
    code_train = code_all[: len(code_all) // 2]
    code_test = code_all[len(code_all) // 2 :]
    data = {
        "text_train": text_train[:1000],
        "text_test": text_test[:1000],
        "math_train": math_train[:1000],
        "math_test": math_test[:1000],
        "code_train": code_train[:1000],
        "code_test": code_test[:1000],
    }
    return DatasetDict(
        {
            "text_train": Dataset.from_dict({"input": data["text_train"]}),
            "text_test": Dataset.from_dict({"input": data["text_test"]}),
            "math_train": Dataset.from_dict({"input": data["math_train"]}),
            "math_test": Dataset.from_dict({"input": data["math_test"]}),
            "code_train": Dataset.from_dict({"input": data["code_train"]}),
            "code_test": Dataset.from_dict({"input": data["code_test"]}),
        }
    )


def get_seed_instructions_dataset():
    """
    Returns a DatasetDict with text, math, and code seed instructions (train/test splits).
    """
    # text instructions
    dolly_ds = load_dataset("databricks/databricks-dolly-15k")
    text_train = [
        ex["instruction"]
        for ex in dolly_ds["train"]
        if ex["category"] == "open_qa"
        and ex["context"] == ""
        and len(ex["instruction"]) < 500
    ]
    # math instructions
    gsm_ds = load_dataset("openai/gsm8k", "main")
    math_train = [ex["question"] for ex in gsm_ds["train"] if len(ex["question"]) < 500]
    # code instructions
    alpaca_ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
    code_train = [
        ex["instruction"]
        for ex in alpaca_ds["train"]
        if ex["input"] == "" and len(ex["instruction"]) < 500
    ]
    data = {
        "text_train": text_train[:1000],
        "math_train": math_train[:1000],
        "code_train": code_train[:1000],
        "text_test": text_train[1000:2000],
        "math_test": math_train[1000:2000],
        "code_test": code_train[1000:2000],
    }
    return DatasetDict(
        {
            "text_train": Dataset.from_dict({"input": data["text_train"]}),
            "text_test": Dataset.from_dict({"input": data["text_test"]}),
            "math_train": Dataset.from_dict({"input": data["math_train"]}),
            "math_test": Dataset.from_dict({"input": data["math_test"]}),
            "code_train": Dataset.from_dict({"input": data["code_train"]}),
            "code_test": Dataset.from_dict({"input": data["code_test"]}),
        }
    )


@register_factory("axbench")
class AxbenchDatasetFactory(BaseDatasetFactory):
    """AxBench dataset factory for generating HuggingFace datasets for training and evaluation"""

    def __init__(
        self,
        model=None,
        client=None,
        tokenizer=None,
        dataset_category="instruction",
        num_of_examples=1000,
        output_length=32,
        dump_dir=None,
        use_cache=True,
        start_concept_id=0,
        is_chat_model=True,
        include_system_prompt=False,
        has_prompt_steering=False,
        master_data_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.use_cache = use_cache
        self.dataset_category = dataset_category
        self.num_of_examples = num_of_examples
        self.output_length = output_length
        self.include_system_prompt = include_system_prompt
        self.is_chat_model = is_chat_model
        self.seed = kwargs.get("seed", 42)
        self.logger = kwargs.get("logger", logger)
        self.lm_model = None
        self.has_prompt_steering = has_prompt_steering or kwargs.get(
            "has_prompt_steering", False
        )
        self.master_data_dir = master_data_dir or kwargs.get("master_data_dir", None)
        if client is not None:
            lm_model = kwargs.get("lm_model", "gpt-4o-mini")
            self.lm_model = LanguageModel(
                lm_model,
                client,
                dump_dir,
                use_cache=use_cache,
                master_data_dir=self.master_data_dir,
            )
        # Load seed sentences and instructions
        self.seed_sentences = get_seed_sentences_dataset()
        self.seed_instructions = get_seed_instructions_dataset()

        # create a shared genre-based negative pools all at once
        if start_concept_id == 0 and not kwargs.get("is_inference", False):
            per_category_n = int(num_of_examples // 2)
            start = time.time()
            self.logger.warning(
                "Creating genre-based and shared negative examples for all concepts."
            )
            random_examples = []
            for genre in tqdm(["text", "math", "code"], desc="Processing genres"):
                random_content = get_random_content(
                    self.seed_sentences
                    if self.dataset_category == "continuation"
                    else self.seed_instructions,
                    tokenizer=self.tokenizer,
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

    async def _get_steering_prompts(self, concepts):
        # Use the LanguageModel to generate steering prompts for each concept
        prompts = [T_GENERATE_STEERING_PROMPT % (concept) for concept in concepts]
        completions = await self.lm_model.chat_completions(
            api_names=[self.lm_model.model] * len(prompts),
            prompts=prompts,
            batch_size=8,
        )
        return [c.strip() for c in completions]

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

    def create_train_ds(
        self,
        dataset_name,
        data_files=None,
        split="train",
        cache_dir=None,
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
        Load and process AxBench training dataset from HuggingFace, returning a HuggingFace dataset.
        """
        logger.debug(f"Loading training dataset {dataset_name} with files {data_files}")
        dataset = load_dataset(
            dataset_name,
            data_files=data_files,
            split=split if kwargs.get("use_split", False) else "train",
            cache_dir=cache_dir,
        )
        logger.debug(f"Loaded dataset with {len(dataset)} examples")
        if select_concept_ids:
            dataset = dataset.filter(
                lambda x: x["concept_id"] in select_concept_ids, num_proc=4
            )
            logger.debug(f"Filtered to {len(dataset)} examples for selected concepts")
        if max_concepts:
            concept_ids = list(set(dataset["concept_id"]))
            concept_ids = [cid for cid in concept_ids if cid >= 0]
            concept_ids.sort()
            limited_concept_ids = concept_ids[:max_concepts]
            dataset = dataset.filter(
                lambda x: x["concept_id"] in limited_concept_ids, num_proc=4
            )
            logger.debug(
                f"Limited to {len(limited_concept_ids)} concepts with {len(dataset)} examples"
            )
        if self.tokenizer is not None and self.model is not None:
            dataset = process_dataset_for_training(
                dataset=dataset,
                tokenizer=self.tokenizer,
                model_name=self.model,
                binarize=binarize,
                train_on_negative=train_on_negative,
                output_length=output_length or self.output_length,
                max_num_of_examples=max_num_of_examples or self.num_of_examples,
                negative_example_ratio=negative_example_ratio,
                replace_negative_description=replace_negative_description,
                is_chat_model=self.is_chat_model,
            )
            logger.debug(
                f"Processed dataset with {len(dataset)} examples ready for training"
            )
        return dataset

    def create_eval_ds(
        self,
        concepts,
        subset_n=1,
        steering_factors=None,
        steering_datasets=None,
        steering_model_name=None,
        **kwargs,
    ):
        """
        Generate a HuggingFace Dataset for steering evaluation, for all provided concepts, factors, and dataset types.
        Args:
            concepts: List of concept names/strings to evaluate
            subset_n: Number of prompts/examples per concept
            steering_factors: List of steering factors to use
            steering_datasets: List of steering dataset types (e.g., 'OUATPrefix', 'AlpacaEval', ...)
            steering_model_name: (Optional) Model name for formatting
        Returns:
            HuggingFace Dataset containing all evaluation examples
        """
        assert concepts is not None, "concepts must be provided"
        assert steering_factors is not None, "steering_factors must be provided"
        assert steering_datasets is not None, "steering_datasets must be provided"
        all_datasets = []
        for dataset_name in tqdm(
            steering_datasets, desc="Processing steering datasets", disable=len(steering_datasets) < 2
        ):
            if dataset_name == "OUATPrefix":
                all_examples = []
                for idx, concept in enumerate(
                    tqdm(concepts, desc="Processing concepts (OUAT)", disable=len(concepts) < 5)
                ):
                    for i in range(subset_n):
                        for factor in steering_factors:
                            all_examples.append(
                                {
                                    "dataset_name": dataset_name,
                                    "concept_id": idx,
                                    "input_concept": concept,
                                    "input_id": i,
                                    "factor": factor,
                                    "input": "Once upon a time, there was a ",
                                }
                            )
                all_datasets.append(
                    Dataset.from_dict(
                        {k: [ex[k] for ex in all_examples] for k in all_examples[0]}
                    )
                )
            elif dataset_name == "AlpacaEval":
                alpaca_eval_df = load_dataset(
                    "tatsu-lab/alpaca_eval", split="eval", trust_remote_code=True
                ).to_pandas()
                if self.has_prompt_steering and self.lm_model is not None:
                    steering_prompts = asyncio.run(self._get_steering_prompts(concepts))
                else:
                    steering_prompts = [
                        T_PROMPT_STEERING % (concept) for concept in concepts
                    ]
                all_examples = []
                for idx, concept in enumerate(
                    tqdm(concepts, desc="Processing concepts (AlpacaEval)", disable=len(concepts) < 5)
                ):
                    sampled_prompts = alpaca_eval_df.sample(
                        subset_n, random_state=int(idx)
                    )["instruction"].tolist()
                    for i in range(subset_n):
                        sampled_prompt = sampled_prompts[i]
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
                            )
                            formatted_steered_prompt = self.tokenizer.decode(
                                formatted_steered_prompt
                            )
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
                            )[1:]
                            formatted_prompt = self.tokenizer.decode(formatted_prompt)
                        else:
                            formatted_steered_prompt = (
                                self.tokenizer.apply_chat_template(
                                    [{"role": "user", "content": steered_prompt}],
                                    tokenize=True,
                                    add_generation_prompt=True,
                                )[1:]
                            )
                            formatted_steered_prompt = self.tokenizer.decode(
                                formatted_steered_prompt
                            )
                            formatted_prompt = self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": sampled_prompt}],
                                tokenize=True,
                                add_generation_prompt=True,
                            )[1:]
                            formatted_prompt = self.tokenizer.decode(formatted_prompt)
                        for factor in steering_factors:
                            all_examples.append(
                                {
                                    "dataset_name": dataset_name,
                                    "concept_id": idx,
                                    "input_concept": concept,
                                    "input_id": i,
                                    "factor": factor,
                                    "original_prompt": sampled_prompt,
                                    "steered_input": formatted_steered_prompt,
                                    "input": formatted_prompt,
                                }
                            )
                all_datasets.append(
                    Dataset.from_dict(
                        {k: [ex[k] for ex in all_examples] for k in all_examples[0]}
                    )
                )
            elif dataset_name in ("AlpacaEval_Suppress", "AlpacaEval_Synergy"):
                alpaca_eval_df = load_dataset(
                    "tatsu-lab/alpaca_eval", split="eval", trust_remote_code=True
                ).to_pandas()
                common_steering_factors = steering_factors
                if dataset_name == "AlpacaEval_Suppress":
                    common_steering_factors = [
                        f * -1.0 for f in common_steering_factors
                    ]
                if self.has_prompt_steering and self.lm_model is not None:
                    steering_prompts = asyncio.run(self._get_steering_prompts(concepts))
                else:
                    steering_prompts = [
                        T_PROMPT_STEERING % (concept) for concept in concepts
                    ]
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        sampled_prompt = alpaca_eval_df.sample(1)[
                            "instruction"
                        ].tolist()[0]
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
                            all_examples.append(
                                {
                                    "dataset_name": dataset_name,
                                    "concept_id": idx,
                                    "input_concept": concept,
                                    "input_id": i,
                                    "factor": factor,
                                    "original_prompt": sampled_prompt,
                                    "input": formatted_steered_prompt,
                                }
                            )
                all_datasets.append(
                    Dataset.from_dict(
                        {k: [ex[k] for ex in all_examples] for k in all_examples[0]}
                    )
                )
            else:
                raise NotImplementedError(
                    f"Steering dataset {dataset_name} not implemented."
                )
        if len(all_datasets) == 1:
            return all_datasets[0]
        return concatenate_datasets(all_datasets)

    def create_train_df(self, concept, n, concept_genres_map, **kwargs) -> pd.DataFrame:
        """Lower-level utility fns for use in dataset creation."""
        start = time.time()
        logger.warning("Creating dataframe.")
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
            tokenizer=self.tokenizer,
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
        logger.warning(
            f"Finished creating current dataframe in {round(time.time() - start, 3)} sec."
        )
        return df

    def create_dpo_df(self, existing_df, **kwargs) -> pd.DataFrame:
        """Lower-level utility fns for use in dataset creation."""
        start = time.time()
        logger.warning("Creating dataframe.")
        batch_size = kwargs.get("batch_size", 8)
        output_length = kwargs.get("output_length", 32)
        is_chat_model = kwargs.get("is_chat_model", True)
        include_system_prompt = kwargs.get("include_system_prompt", False)
        keep_orig_axbench_format = kwargs.get("keep_orig_axbench_format", False)
        steer_data_type = kwargs.get("steer_data_type", "concept")

        positive_df = existing_df[existing_df["category"] == "positive"]
        positive_prompts = positive_df["input"].tolist()

        # get the concept for this existing_df
        concept = existing_df["output_concept"].iloc[0]

        if keep_orig_axbench_format:
            logger.warning(
                f"keep_orig_axbench_format is set to True. Using the local model to generate responses."
            )
            losing_outputs = get_model_continues(
                kwargs["model"],
                kwargs["tokenizer"],
                positive_prompts,
                max_new_tokens=int(output_length * 1.5),
                is_chat_model=is_chat_model,
                include_system_prompt=include_system_prompt,
                batch_size=batch_size,
                verbose=True,
            )
        else:
            if steer_data_type == "concept":
                losing_output_tasks = response_without_concept(
                    self.lm_model, concept, positive_prompts
                )
            # else:
            #     losing_output_tasks = response_without_rule(
            #         self.lm_model, concept, positive_prompts
            #     )
            losing_outputs = asyncio.run(run_tasks([losing_output_tasks]))[0]
        positive_df["losing_output"] = losing_outputs

        # TODO: comment them out as they are not selected in our offline hyperparameter sweeps.
        # alright, let's get two types of steered inputs and outputs.
        # we should separate between concepts and rules of input
        # if steer_data_type == "concept":
        #     steered_prompt_tasks = get_dpo_steering_prompt(
        #         self.lm_model, positive_prompts, concept)

        # elif steer_data_type == "rule":
        #     steered_prompt_tasks = get_dpo_steering_prompt_rule(
        #         self.lm_model, positive_prompts, concept)

        # blend_in_steered_prompts = asyncio.run(run_tasks([steered_prompt_tasks]))[0]
        # blend_in_steered_output_tasks = response_with(
        #     self.lm_model, blend_in_steered_prompts)
        # blend_in_steered_outputs = asyncio.run(run_tasks([blend_in_steered_output_tasks]))[0]
        # positive_df["blend_in_steered_input"] = blend_in_steered_prompts
        # positive_df["blend_in_steered_output"] = blend_in_steered_outputs

        # if steer_data_type == "concept":
        #     prepend_steered_prompt_tasks = get_dpo_steering_prompt(
        #         self.lm_model, positive_prompts, concept, use_simple=True)
        # else:
        #     prepend_steered_prompt_tasks = get_dpo_steering_prompt_rule(
        #         self.lm_model, positive_prompts, concept, use_simple=True)

        # prepend_steered_prompts = asyncio.run(run_tasks([prepend_steered_prompt_tasks]))[0]
        # prepend_steered_prompts = [
        #     f"{steering_prompt}\n\nQuestion: {sampled_prompt}"
        #     for steering_prompt, sampled_prompt in zip(prepend_steered_prompts, positive_prompts)]
        # prepend_steered_output_tasks = response_with(
        #     self.lm_model, prepend_steered_prompts)
        # prepend_steered_outputs = asyncio.run(run_tasks([prepend_steered_output_tasks]))[0]
        # positive_df["prepend_steered_input"] = prepend_steered_prompts
        # positive_df["prepend_steered_output"] = prepend_steered_outputs

        logger.warning(
            f"Finished creating current dataframe in {round(time.time() - start, 3)} sec."
        )
        return positive_df
