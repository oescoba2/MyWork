from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer
    )    
from trl import (
    DPOConfig,
    DPOTrainer,
    RewardConfig, 
    RewardTrainer,
    SFTConfig, 
    SFTTrainer
    )
from trl.experimental.ppo import PPOConfig, PPOTrainer
from typing import Optional
from utils import timer
import torch
import random
import wandb

class DPO():
    """
    Direct Preference Optimization (DPO) training for preference-aligning a
    a pre-trained causal large language model (LLM) on a given preference data.
    Handles the Anthropic hh-rlhf dataset specifically.
    """

    def __init__(self, 
                 model: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer, 
                 max_seq_len: int = 1024) -> None:
        """Initialize the DPO class.

        Parameters:
            - model (AutoModelForCausalLM) : The SFT model to improve with preference-alignment.
            - tokenizer (AutoTokenizer) : Tokenizer corresponding to the model.
            - max_seq_length (int) : Maximum length that a given prompt (input) token sequence 
                                     can have. This controls input size.

        Returns:
            - None
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"
        self.max_seq_len = max_seq_len
        self.metrics = {}
        self._v_num = 0

    def _init_wandb_run(self, 
                        pipeline: str, 
                        component: str, 
                        version: int, 
                        log_with: str = None) -> None:
        """Initialize a W&B run automatically for the trainer.
    
        Parameters:
            - pipeline (str): 'DPO' or 'RLHF' (used as project name)
            - component (str): 'SFT', 'DPO', 'Reward', 'PPO' (used as run_name)
            - version (int): pipeline version (used as group_name)
            - log_with (str): 'wandb' or None
        """

        if log_with == "wandb":
            if not hasattr(self, "_wandb_run") or self._wandb_run is None:

                self._wandb_run = wandb.init(
                    project=pipeline,
                    group=f"pipeline-v{version}",
                    name=component,
                    reinit=True
                )

    def _extract_prompt_and_responses(self, 
                                      example : str) -> dict[str, str]:
        """
        Convert hh-rlhf dataset example into DPO format:
        - prompt: everything before last assistant response
        - chosen: last assistant response in 'chosen'
        - rejected: last assistant response in 'rejected'

        Parameters:
            - example (str) : the element in the dataset.

        Returns:
            - (dict[str, str])
        """

        def split_last_assistant(text : str) -> tuple[str, str]:
            # Split into human/assistant segments
            segments = text.strip().split("Assistant:")
            if len(segments) < 2:
                raise ValueError("Expected at least one Assistant segment.")
            # Prompt is everything before the last assistant
            prompt = "Assistant:".join(segments[:-1]).strip()
            # Last assistant segment is the response
            last_response = segments[-1].strip()
            return prompt, last_response

        prompt_chosen, chosen_response = split_last_assistant(example["chosen"])
        prompt_rejected, rejected_response = split_last_assistant(example["rejected"])
        return {
            "prompt": prompt_chosen,
            "chosen": chosen_response,
            "rejected": rejected_response
        }

    def _preprocess_dataset(self, 
                            dataset_name: str = "Anthropic/hh-rlhf",
                            split: str = "train",
                            subsample_size: Optional[int] = None,
                            val_frac: float = 0.1,
                            seed: int = 42) -> tuple[Dataset, Dataset]:
        """
        Load and preprocess hh-rlhf dataset into DPO-ready format.

            - dataset_name (str): HuggingFace dataset name.
            - split (str): Dataset split to use ('train' by default).
            - subsample_size (Optional[int]): Number of examples to subsample.
            - val_frac (float) : the percentage of the data to be set aside for validation.
            - seed (int): Random seed for reproducibility.

        Returns:
            - tuple[Dataset, Dataset]
        """
        dataset = load_dataset(dataset_name, split=split)

        # Optional subsample
        if subsample_size is not None and subsample_size < len(dataset):
            random.seed(seed)
            indices = random.sample(range(len(dataset)), subsample_size)
            dataset = dataset.select(indices)

        # Map to DPO format
        dataset = dataset.map(self._extract_prompt_and_responses, remove_columns=dataset.column_names)

        # Split train/val
        val_size = int(len(dataset) * val_frac)
        if val_size > 0:
            dataset = dataset.train_test_split(test_size=val_size, seed=seed)
            train_dataset = dataset["train"]
            val_dataset = dataset["test"]
        else:
            train_dataset = dataset
            val_dataset = None

        return train_dataset, val_dataset
    
    @timer
    def train(self, 
              dataset_name: str = "Anthropic/hh-rlhf",
              output_dir: str = "./dpo_model",
              num_epochs: int = 1,
              batch_size: int = 2,
              max_resp_len : int = 64,
              lr: float = 5e-5,
              β : float = 0.1,
              val_frac: float = 0.1,
              subsample_size: Optional[int] = None,
              seed: int = 42,
              log_with: Optional[str] = 'wandb',
              project_name: Optional[str] = None) -> tuple[AutoModelForCausalLM, dict[str, list[float]]]:
        """
        Train the model on preference data using DPO.

        Parameters:
            - dataset_name (str): HuggingFace dataset name containing preference data.
            - output_dir (str): Directory to save model checkpoints.
            - num_epochs (int): Number of training epochs.
            - batch_size (int): Training batch size per device.
            - lr (float) : Learning rate for optimizer.
            - β (float) : a float controlling how much the model in training deviates 
                          from the reference model. Default is 0.1. Higher value
                          means to stay closer to reference model.
            - val_frac (float) : the percentage of the data to be set aside for validation.
            - split (str): Dataset split to use ('train' by default).
            - subsample_size (Optional[int]): Subsample size for dataset.
            - seed (int): Random seed for reproducibility.
            - project_name (Optional[str]): Name for logging project (e.g., W&B).
            - log_with (Optional[str]): Logging platform, e.g., 'wandb'.

        Returns:
            Tuple[AutoModelForCausalLM, Dict[str, list]]:
                - The trained LLM.
                - Training metrics log (loss, eval_loss, etc.).
        """

        self._v_num += 1
        train_dataset, val_dataset = self._preprocess_dataset(dataset_name=dataset_name,
                                                              split="train",
                                                              subsample_size=subsample_size,
                                                              val_frac=val_frac,
                                                              seed=seed)

        # Configure DPO
        if not log_with:
            config = DPOConfig(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                max_completion_length=max_resp_len,
                learning_rate=lr,
                beta=β,
                logging_steps=50,
                save_strategy="epoch",
                fp16=torch.cuda.is_available()
            )
        else:
            pipeline = project_name if project_name else 'DPO'
            component = self.__class__.__name__
            self._init_wandb_run(pipeline, component, self._v_num, log_with)
            config = DPOConfig(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                max_completion_length=max_resp_len,
                learning_rate=lr,
                beta=β,
                logging_steps=50,
                save_strategy="epoch",
                fp16=torch.cuda.is_available(),
                project=project_name,
                report_to=log_with
            )

        # Initialize trainer
        trainer = DPOTrainer(
            model = self.model,
            args = config,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            processing_class = self.tokenizer
        )

        # Train
        trainer.train()
        self.metrics = trainer.state.log_history

        if log_with == 'wandb':
            self._wandb_run.finish()
            self._wandb_run = None

        return self.model, self.metrics

class PPO():
    """Class to implement RLHF on a pre-trained LLM using HuggingFace.
    The main assumption is to take a SFT LLM as the policy to train
    using the Proximal Policy Optimization algorithm.
    """

    def __init__(self,
                 base_model: str = "distilgpt2",
                 sft_model: AutoModelForCausalLM | None = None,
                 reward_model: AutoModelForSequenceClassification | None = None) -> None:
        """Create an instance to perform RLHF.
        
        Parameters:
            - base_model (str) : the pre-trained LLM to use to create all models 
                                 (or just the tokenizer and state-value model).
            - sft_model (AutoModelForCausalLM) : the SFT LLM that will be used as
                                                 the main policy for RL to improve
                                                 using PPO. Default is None so that
                                                 a generic model is made using the
                                                 passed name in base_model.
            - reward_model (AutoModelForSequenceClassification) : the trained reward
                                                                  model that outputs
                                                                  a reward for a given
                                                                  response, based on a
                                                                  prompt. This guides
                                                                  the sft_model to 
                                                                  align with human 
                                                                  preferences. Default
                                                                  is None so that a 
                                                                  generic model is made
                                                                  using the name passed
                                                                  to base_model.
        Returns:
            - None
        """

        self.model_name = base_model
        self._v_num = 0

        # -------------------------
        # Tokenizer (PPO-compatible)
        # -------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # -------------------------
        # Policy model
        # -------------------------
        if sft_model is None:
            self.policy = AutoModelForCausalLM.from_pretrained(base_model)
            self.policy.resize_token_embeddings(len(self.tokenizer))
        else:
            self.policy = sft_model
        # Prevents PolicyAndWrapper attribute error
        self.policy.gradient_checkpointing_disable()

        # -------------------------
        # Reference policy (frozen)
        # -------------------------
        self.ref_policy = AutoModelForCausalLM.from_pretrained(base_model)
        self.ref_policy.resize_token_embeddings(len(self.tokenizer))
        self.ref_policy.load_state_dict(self.policy.state_dict())
        self.ref_policy.eval()

        # -------------------------
        # Reward model (scalar; frozen)
        # -------------------------
        if reward_model is None:
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(base_model, 
                                                                                   num_labels = 1)
            self.reward_model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.reward_model = reward_model
        self.reward_model.eval()

        # -------------------------
        # State Value model (learned)
        # -------------------------
        self.value_model = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                                              num_labels = 1)
        self.value_model.resize_token_embeddings(len(self.tokenizer))

    def _init_wandb_run(self, 
                        pipeline: str, 
                        component: str, 
                        version: int, 
                        log_with: str = None) -> None:
        """Initialize a W&B run automatically for the trainer.
    
        Parameters:
            - pipeline (str): 'DPO' or 'RLHF' (used as project name)
            - component (str): 'SFT', 'DPO', 'Reward', 'PPO' (used as run_name)
            - version (int): pipeline version (used as group_name)
            - log_with (str): 'wandb' or None
        """

        if log_with == "wandb":
            if not hasattr(self, "_wandb_run") or self._wandb_run is None:

                self._wandb_run = wandb.init(
                    project=pipeline,
                    group=f"pipeline-v{version}",
                    name=component,
                    reinit=True
                )
    
    def _load_prompt_only_dataset(self, 
                                  dataset_name: str, 
                                  subsample_size : int = None,
                                  split: str = "train", 
                                  text_key: str = None,
                                  seed : int = 42,
                                  max_seq_len : int = 2048,
                                  val_size: float = 0.1,):
        """
            Load a dataset and convert it to prompt-only tokenized train/validation datasets.

            Parameters:
                dataset_name (str): HF dataset identifier.
                subsample_size (int, optional): Optional random subsample size.
                split (str): Dataset split to load.
                text_key (str, optional): Explicit prompt field override.
                seed (int): RNG seed.
                max_seq_len (int): Max tokenized prompt length (input size).
                val_size (float): Fraction of data used for validation (e.g., 0.1 = 10%).

            Returns:
                train_dataset, val_dataset
        """

        # -------------------------
        # Load raw dataset
        # -------------------------
        dataset = load_dataset(dataset_name, split=split)

        # -------------------------
        # Optional subsampling
        # -------------------------
        if subsample_size is not None and subsample_size < len(dataset):
            random.seed(seed)
            indices = random.sample(range(len(dataset)), subsample_size)
            dataset = dataset.select(indices)

        # -------------------------
        # Identify prompt field
        # -------------------------
        if "prompt" in dataset.column_names:
            dataset_text_field = "prompt"
        elif text_key and text_key in dataset.column_names:
            dataset_text_field = text_key
        else:
            candidates = ["instruction", "text", "input", "question"]
            found = next((c for c in candidates if c in dataset.column_names), None)
            if found is None:
                raise ValueError(
                    f"Could not find a text field in dataset columns: {dataset.column_names}"
                )
            dataset_text_field = found

        # -------------------------
        # Train / validation split
        # -------------------------
        if not (0.0 < val_size < 1.0):
            raise ValueError("val_size must be a float in (0, 1)")

        split_datasets = dataset.train_test_split(
            test_size=val_size,
            seed=seed,
            shuffle=True,
        )

        train_dataset = split_datasets["train"]
        val_dataset = split_datasets["test"]

        # -------------------------
        # Tokenization
        # -------------------------
        def tokenize_fn(batch):
            outputs = self.tokenizer(
                batch[dataset_text_field],
                truncation=True,
                max_length=max_seq_len,
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        val_dataset = val_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=val_dataset.column_names,
        )

        return train_dataset, val_dataset
   
    @timer
    def train(self, 
              dataset_name: str = "tatsu-lab/alpaca",
              subsample_size : int = 2000,
              seed :int = 42,
              max_seq_len : int = 2048,
              max_resp_len : int = 64,
              val_size : float = 0.10,
              lr : float = 1e-4,
              device_batch_size : int = 4,
              grad_accum_steps : int = 1,
              total_episodes : int = 1000,
              num_mini_batches : int = 4,
              num_ppo_epochs : int = 4,
              λ : float = 0.95,
              γ : float = 1.0,
              kl_coef : float = 0.05,
              vf_coef : float = 0.1,
              temperature : float = 0.7,
              log_with : str = 'wandb',
              project_name : str = None
              ):
        """Train the SFT model using RL with the preference data. One episode equals 
        one prompt and one generated response (i.e. one episode is (prompt, response
        reward)-tuple). Note that reward is generated on the response by a frozen
        reward model.

        Parameters:
            - dataset_name
            - subsample_size
            - seed
            - max_seq_len (int) : the max number of tokens to use when generating the
                                  sequence of the prompt (i.e. input). This controls
                                  the input size. Default is 2048.
            - max_resp_len (int) : the max number of tokens to use when the policy (SFT
                                   LLM) generates a response. This controls the size of
                                   the output. Default is 128.
            - lr (float) : the learning rate to use on the optimization. Default is 1e-5.
            - device_batch_size (int) : the size of the batch that each device, GPU/CPU,
                                        will work with. Default is 4. With the defaults
                                        of total_episodes, num_mini_batches, we break
                                        1000 episodes into 4 batches of 250 episodes 
                                        each. The 250 batch is then broken into 4
                                        batches, of about 62 episdes each, to be handled 
                                        by the device.
            - grad_accum_steps (int) : the number of batches to wait before running the
                                        optimizer step. Default is 1, so that the optimizer
                                        step is ran at each device batch (collection of 62
                                        episodes).
            - total_episodes (int) : the total number of episodes run the policy (SFT
                                     model) on the environment (prompt). Default is 
                                     1000, so 1000 (prompt, response, reward)
            - num_mini_batches (int) : the total number of episodes to use when performing
                                       one optimizing step (i.e. loss, backprop, step)
                                       of the policy's parameters. Default is 4, so that
                                       for the default total_episodes, 250 prompt-response
                                       pairs are used to update the policy's parameters.
            - num_ppo_epochs (int) : the number epochs to train the policy's parameters on
                                     the generated total_episodes data. Default is 4.
                                     Thus, for total_episodes and num_mini_batches default
                                     values, we have a total 16 parameter udpates (1
                                     update per mini batch, for a total of 4 updates,
                                     but episodes are generated 4 times total).
            - λ (float) : lambda value used in Generalized Advantage Estimation (GAE).
                           Default is 0.95
            - γ (float) : the discount factor, used in RL and GAE. Default is 1.0.
            - kl_coef (float) : the Kullback-Leibler (KL) divergence coefficient used
                                in the PPO loss function. Default is 0.05.
            - vl_coef (float) : the state value function coefficient. Default is 0.1
            - log_with (Optional[str]): Logging platform, e.g., 'wandb'. Default is
                                        None.
            - project_name (Optional[str]): Project name for W&B or other logger.
                                            Default is None.

        """

        self._v_num += 1
        train_dataset, val_dataset = self._load_prompt_only_dataset(dataset_name,
                                                                    subsample_size = subsample_size,
                                                                    seed = seed,
                                                                    max_seq_len = max_seq_len,
                                                                    val_size = val_size)
        if log_with:

            pipeline = project_name if project_name else 'RLHF'
            component = self.__class__.__name__
            self._init_wandb_run(pipeline, component, self._v_num, log_with)
            config = PPOConfig(output_dir="./ppo-distilgpt2",
                               learning_rate = lr,
                               per_device_train_batch_size = device_batch_size,
                               gradient_accumulation_steps = grad_accum_steps,
                               num_ppo_epochs = num_ppo_epochs,
                               num_mini_batches = num_mini_batches,
                               total_episodes = total_episodes,
                               response_length = max_resp_len,
                               lam = λ,
                               gamma = γ,
                               kl_coef = kl_coef,
                               vf_coef = vf_coef,
                               temperature = temperature,
                               save_strategy = "epoch",
                               project = project_name,
                               report_to = log_with,
                               logging_steps = 50,
                               fp16=torch.cuda.is_available()
                               )
        else:
            config = PPOConfig(output_dir="./ppo-distilgpt2",
                               learning_rate = lr,
                               per_device_train_batch_size = device_batch_size,
                               gradient_accumulation_steps = grad_accum_steps,
                               num_ppo_epochs = num_ppo_epochs,
                               num_mini_batches = num_mini_batches,
                               total_episodes = total_episodes,
                               response_length = max_resp_len,
                               lam = λ,
                               gamma = γ,
                               kl_coef = kl_coef,
                               vf_coef = vf_coef,
                               temperature = temperature,
                               save_strategy = "epoch",
                               logging_steps = 50,
                               fp16=torch.cuda.is_available()
                               )
        trainer = PPOTrainer(args = config,
                             processing_class = self.tokenizer,
                             model = self.policy,
                             ref_model = self.ref_policy,
                             reward_model = self.reward_model,
                             value_model = self.value_model,
                             train_dataset = train_dataset,
                             eval_dataset = val_dataset)
        
        trainer.train()
        self.metrics = trainer.state.log_history

        if log_with == 'wandb':
            self._wandb_run.finish()
            self._wandb_run = None

        return self.policy, self.metrics

class Reward():
    """Take a base pre-trained large language model (LLM) as the reward model 
    for needed in RLHF's PPO stage that is trained on Anthropic HH-RLHF.
    Outputs a scalar reward.
    """

    def __init__(self, 
                 llm_name: str = "distilgpt2",
                 max_seq_len: int = 2048) -> None:
        """Create an instance the reward model to train.

        Parameters: 
            - llm_name (str) : the pre-trained LLM to use (from HF).
            - max_seq_len (int) : max length of tokenized sequence. Default
                                     is 128.
        Returns:
            - None
        """

        # Tokenizer (PPOTrainer-compatible)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Scalar reward head
        self.model = AutoModelForSequenceClassification.from_pretrained(llm_name, num_labels=1)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_seq_len = max_seq_len
        self.metrics = {}
        self._v_num = 0

    def _init_wandb_run(self, 
                        pipeline: str, 
                        component: str, 
                        version: int, 
                        log_with: str = None) -> None:
        """Initialize a W&B run automatically for the trainer.
    
        Parameters:
            - pipeline (str): 'DPO' or 'RLHF' (used as project name)
            - component (str): 'SFT', 'DPO', 'Reward', 'PPO' (used as run_name)
            - version (int): pipeline version (used as group_name)
            - log_with (str): 'wandb' or None
        """

        if log_with == "wandb":
            if not hasattr(self, "_wandb_run") or self._wandb_run is None:

                self._wandb_run = wandb.init(
                    project=pipeline,
                    group=f"pipeline-v{version}",
                    name=component,
                    reinit=True
                )
    
    def _load_and_split_dataset(self,
                                val_size : float = 0.10,
                                subsample_size : Optional[int] = None,
                                seed: int = 42) -> Dataset:
        """
        Loads and optionally subsamples Anthropic HH-RLHF dataset.

        Parameters:

        Returns :
            - (tuple[Dataset, Dataset]) : the training and validation 
                                          datasets, in that order.
        """
        if not (0.0 < val_size < 1.0):
            raise ValueError("val_size must be in (0, 1)")
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")

        if subsample_size and subsample_size < len(dataset):
            random.seed(seed)
            indices = random.sample(range(len(dataset)), subsample_size)
            dataset = dataset.select(indices)

        split = dataset.train_test_split(test_size = val_size,
                                         seed = seed,
                                         shuffle = True)

        return split['train'], split['test']

    def _tokenize_preferences(self, dataset : Dataset) -> Dataset:
        """Tokenize the (chosen, rejected) preference pairs.
        
        """
        def tokenize_func(batch : dict[str, list[float]]) -> dict[str, list[float]]:
            """The actual function that tokenizes. Meant to be applied
            accross the whole Dataset object.
            

            """

            chosen = self.tokenizer(
                batch["chosen"],
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
            )
            rejected = self.tokenizer(
                batch["rejected"],
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
            )

            return {"input_ids_chosen" : chosen["input_ids"],
                    "attention_mask_chosen": chosen["attention_mask"],
                    "input_ids_rejected": rejected["input_ids"],
                    "attention_mask_rejected": rejected["attention_mask"]}

        return dataset.map(tokenize_func,
                           batched=True,
                           remove_columns=dataset.column_names)
    
    @timer
    def train(self,
              num_epochs: int = 1,
              batch_size: int = 8,
              lr: float = 1e-5,
              val_size: float = 0.10,
              subsample_size: Optional[int] = None,
              seed: int = 42,
              log_with : Optional[str] = 'wandb',
              project_name: Optional[str] = None) -> tuple[AutoModelForSequenceClassification, 
                                       dict[str, list[float]]]:
        """
        Train reward model with pairwise preference loss.

        Parameters:
            - 
        """

        self._v_num += 1
        # Load + split
        train_ds, val_ds = self._load_and_split_dataset(val_size = val_size,
                                                        subsample_size = subsample_size,
                                                        seed = seed)

        # # Tokenize
        # train_ds = self._tokenize_preferences(train_ds)
        # val_ds = self._tokenize_preferences(val_ds)

        if not log_with:

            config = RewardConfig(
                per_device_train_batch_size = batch_size,
                num_train_epochs = num_epochs,
                learning_rate = lr,
                fp16 = torch.cuda.is_available(),
                max_length = self.max_seq_len,
                logging_steps = 50,
                eval_strategy="epoch"
            )

        else:
            
            if not project_name and (log_with == 'wandb'):
                raise ValueError("Must specify the project name for wandb.")
            pipeline = project_name 
            component = self.__class__.__name__
            self._init_wandb_run(pipeline, component, self._v_num, log_with)
            config = RewardConfig(
                per_device_train_batch_size = batch_size,
                num_train_epochs = num_epochs,
                learning_rate = lr,
                fp16 = torch.cuda.is_available(),
                max_length = self.max_seq_len,
                logging_steps = 50,
                eval_strategy = "epoch",
                project = project_name,
                report_to = log_with
            )

        trainer = RewardTrainer(
            model = self.model,
            args = config,
            train_dataset = train_ds,
            eval_dataset = val_ds,
            processing_class = self.tokenizer,
        )

        trainer.train()
        self.metrics = trainer.state.log_history

        if log_with == 'wandb':
            self._wandb_run.finish()
            self._wandb_run = None

        return self.model, self.metrics

class SFT():
    """Take a pre-trained large language model (LLM) and instruction tune 
    (i.e. supervised fine-tune, SFT) the LLM using HuggingFace (HF).
    """

    def __init__(self, 
                 llm_name : str = 'distilgpt2',
                 max_seq_len : int = 2048,
                 for_ppo : bool = False) -> None:
        """Create an instance of SFT to use.

        Parameters:
            - llm_name (str) : the name of the pre-trained LLM to use. Default
                               is 'distilgpt2'.
            - max_seq_len (int) : the max sequence length for tokenization.
                                  Default is 128.
            - for_ppo (bool) : whether the instruction tuned model is to be
                               used by PPO in RLHF. Default is False. The
                               SFT LLM will become the main model in that
                               the PPO stage of RLHF will seek to improve
                               using preference data and a trained reward
                               model.
        Returns
            - None
        """

        self.for_ppo = for_ppo
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

        # HF PPOTrainer requires padding
        if self.for_ppo:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(llm_name)
        if self.for_ppo:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_seq_len = max_seq_len
        self.metrics = {}
        self._v_num = 0

    def _init_wandb_run(self, 
                        pipeline: str, 
                        component: str, 
                        version: int, 
                        log_with: str = None) -> None:
        """Initialize a W&B run automatically for the trainer.
    
        Parameters:
            - pipeline (str): 'DPO' or 'RLHF' (used as project name)
            - component (str): 'SFT', 'DPO', 'Reward', 'PPO' (used as run_name)
            - version (int): pipeline version (used as group_name)
            - log_with (str): 'wandb' or None
        """

        if log_with == "wandb":
            if not hasattr(self, "_wandb_run") or self._wandb_run is None:

                self._wandb_run = wandb.init(
                    project=pipeline,
                    group=f"pipeline-v{version}",
                    name=component,
                    reinit=True
                )
    
    def _preprocess_data(self, 
                         dataset_name: str = "tatsu-lab/alpaca",
                         data_subsample_size: Optional[int] = None,
                         val_frac: float = 0.1,
                         seed: int = 42) -> DatasetDict:
        """
        Load, subsample, split, and tokenize a HF dataset for SFT. Handles left
        padding if for_ppo=True.

        Parameters:
            - dataset_name (str) : Name of the HF dataset. Must be instruction
                                   dataset. Default is 'tatsu/alpaca.'
            - data_subsample_size (Optional[int]):  Number of examples to subsample
                                                    for training. If None, uses the 
                                                    full dataset.
            - val_frac (float) : Fraction of the dataset to use as validation.
                                 Default is 0.1, meaning 10% of the gathered data
                                 is a validation set.
            - seed (int) : Random seed for reproducibility.

        Returns:
            - (DatasetDict) : Tokenized dataset with keys 'train' and 'val'.
        """

        dataset = load_dataset(dataset_name)

        if data_subsample_size is not None and data_subsample_size < len(dataset["train"]):
            random.seed(seed)
            indices = random.sample(range(len(dataset["train"])), data_subsample_size)
            dataset["train"] = dataset["train"].select(indices)

        # Split train/val
        split_datasets = dataset["train"].train_test_split(test_size=val_frac, seed=seed)
        train_dataset, val_dataset = split_datasets["train"], split_datasets["test"]

        def tokenize_batch(examples):
            prompts = [
                f"Below is an instruction:\n### Instruction:\n{instr}"
                + (f"\n### Input:\n{inp}" if inp.strip() else "")
                + f"\n### Response:\n{out}"
                for instr, inp, out in zip(
                    examples["instruction"],
                    examples.get("input", [""] * len(examples)),
                    examples["output"]
                )
            ]

            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length" if self.for_ppo else False
            )

            # For PPO, remove attention mask and only keep input_ids
            if self.for_ppo:
                return {"input_ids": tokenized["input_ids"]}
            else:
                return tokenized

        train_dataset = train_dataset.map(tokenize_batch, batched=True)
        val_dataset = val_dataset.map(tokenize_batch, batched=True)

        if self.for_ppo:
            train_dataset.set_format(type="torch", columns=["input_ids"])
            val_dataset.set_format(type="torch", columns=["input_ids"])
        else:
            train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
            val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        return DatasetDict({"train": train_dataset, "val": val_dataset})

    @timer   
    def train(self, 
              dataset_name: str = "tatsu-lab/alpaca", 
              data_subsample_size: Optional[int] = None,
              num_epochs: int = 3, 
              batch_size: int = 4,
              output_dir: str = "./sft_model",
              lr: float = 1e-5,
              val_size: float = 0.10,
              seed: int = 42,
              log_with: Optional[str] = 'wandb',
              project_name: Optional[str] = None) -> tuple[AutoModelForCausalLM, dict[str, list[float]]]:
        """Train the pre-trained LLM on a instruction-dataset using SFT.

        Parameters:
            - dataset_name (str): HF dataset name for training. Default is 
                                  'tatsu-lab/alpaca'.
            - num_epochs (int): Number of training epochs. Default is 3.
            - batch_size (int): Training batch size per device. Default is 4.
            - output_dir (str): Directory to save model checkpoints. Default
                                is "./sft_model'.
            - lr (float): Learning rate for optimizer. Default is 1e-5.
            - data_subsample_size (Optional[int]): Number of examples to subsample.
                                                   Default is None.
            - val_size (float): Fraction of dataset to use for validation. Default
                                is 0.1.
            - seed (int): Random seed for reproducibility. Default is 42.
            - log_with (Optional[str]): Logging platform, e.g., 'wandb'. Default is
                                        None.
            - project_name (Optional[str]): Project name for W&B or other logger.
                                            Default is None.

        Returns:
            - tuple(AutoModelForCausalLM, 
                    dict[str, list[floats]]) : the SFT HF model and the Training 
                                               metrics log (loss, eval_loss, etc.).
        """

        self._v_num += 1
        tokenized_dataset = self._preprocess_data(
            dataset_name,
            data_subsample_size,
            val_size,
            seed
        )

        if not log_with:
            config = SFTConfig(
                output_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=lr,
                logging_steps=50,
                save_strategy="epoch",
                fp16=torch.cuda.is_available()
            )
        else:
            
            if not project_name and (log_with == 'wandb'):
                raise ValueError("Must specify the project name for wandb.")
            pipeline = project_name 
            component = self.__class__.__name__
            self._init_wandb_run(pipeline, component, self._v_num, log_with)
            config = SFTConfig(
                output_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=lr,
                logging_steps=50,
                save_strategy="epoch",
                fp16=torch.cuda.is_available(),
                project=project_name,
                report_to=log_with
            )

        trainer = SFTTrainer(
            self.model,
            config,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['val'],
            processing_class=self.tokenizer
        )

        trainer.train()
        self.metrics = trainer.state.log_history

        if log_with == 'wandb':
            self._wandb_run.finish()
            self._wandb_run = None

        return self.model, self.metrics