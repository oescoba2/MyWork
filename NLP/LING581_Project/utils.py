from functools import wraps
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Callable
import json
import os
import time
import torch

prompts = [
    "Explain the difference between AI and machine learning.",
    "Write a short story about a robot learning empathy.",
    "Summarize the plot of Hamlet in 3 sentences.",
    "What is the capital of France?",
    "What is the largest planet in our solar system?",
    "Tell me about the Book of Mormon.",
    "How does Catholicism differ from Islam?",
    "Give a clear explanation of overfitting and how to prevent it.",
    "Write an email politely requesting an extension on a project deadline.",
    "Provide a beginner-friendly explanation of gradient descent.",
    "Rewrite the following sentence to sound more formal: 'I can’t make it today but maybe tomorrow works.'",
    "Explain how transformers use attention mechanisms.",
    "Generate a creative dialogue between a historian and an AI assistant discussing ancient Rome.",
    "Summarize the main idea of the poem ‘The Road Not Taken’.",
    "What are the main differences between supervised and unsupervised learning?",
    "Explain the concept of market equilibrium in economics.",
    "Write a short motivational message for a student preparing for exams.",
    "Describe the process of photosynthesis in simple terms.",
    "Give step-by-step instructions for making a basic HTTP request in Python.",
    "Explain why clean energy is important.",
    "Compare the roles of the legislative and judicial branches of government.",
    "Provide a concise explanation of Bayesian inference.",
    "Write a three-line poem about winter mornings.",
    "Explain the purpose of regularization in neural networks.",
    "Describe the plot of a fictional sci-fi novel in two paragraphs.",
    "List and explain three common causes of inflation.",
    "Give a simple explanation of what a random variable is.",
    "Write a friendly message welcoming a new member to a study group.",
    "Explain the ethical concerns around genetic modification.",
    "Describe how reinforcement learning differs from supervised learning."
]

def generate_model_responses(model : AutoModelForCausalLM,
                             tokenizer : AutoTokenizer,
                             prompts : list[str] = prompts,
                             max_new_tokens : int = 128,
                             temperature : float = 1.0,
                             batch_size : int = 8,
                             save_responses : bool = False,
                             filename : str = "responses.json") -> list[dict]:
    """Generate responses for a list of prompts using a trained LLM with batching,
    and optionally save them to a JSON file.

    Parameters:
        model (AutoModelForCausalLM): Trained LLM.
        tokenizer (AutoTokenizer): Corresponding tokenizer.
        prompts (list[str]): List of text prompts to generate from.
        max_new_tokens (int): Max number of new tokens to generate per prompt.
        temperature (float): Sampling temperature.
        batch_size (int): Number of prompts to generate at once.
        save_responses (bool): Whether to save the results to JSON.
        filename (str): Name of the JSON file to save.

    Returns:
        List[dict]: Each dict contains:
            {
                "prompt": original prompt string,
                "response": generated text string
            }
    """

    model.eval()
    device = model.device
    outputs = []

    # Batch prompts
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                temperature=temperature
            )

        # Map generated text back to each prompt
        for prompt, gen_ids in zip(batch_prompts, generated_ids):
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            outputs.append({"prompt": prompt, "response": text})

    # Optionally save to JSON
    if save_responses:
        os.makedirs("model_responses", exist_ok=True)
        filepath = os.path.join("model_responses", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

    return outputs

def timer(method : Callable) -> Callable:
    """Decorator to time a train method and store the duration as an 
    instance attribute. The attribute will be named 
    <ClassName>_<func_name>_trainer.

    Paramers:
        - method (Callable) : the callable class function to time.

    Returns:
        - (callable) : the actual function that times execution.
    """

    @wraps(method)
    def wrapper(self, 
                *args : tuple[Any], 
                **kwargs : dict[str, Any]) -> Any:
        """Time a class method.

        Parameters:
            - self : the class instance arg.
            *args (tuple[Any]) : positional arguments
            **kwargs (dict[str, Any]) : keyword arguments.

        Returns:
            - result (Any) : the output of func.
        """

        start_time = time.time()
        result = method(self, *args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        # Create attribute name: ClassName_funcname_trainer
        class_name = type(self).__name__
        attr_name = f"{class_name}_{method.__name__}_exec_time"

        if duration < 60:
            setattr(self, attr_name, duration)

        else:
            tot_mins = duration / 60
            mins = duration // 60
            secs = (tot_mins - mins) * 60
            setattr(self, attr_name, f"{mins} mins {secs:.3f} secs")

        return result

    return wrapper