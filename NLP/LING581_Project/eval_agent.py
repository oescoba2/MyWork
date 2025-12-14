from openai import OpenAI
from tqdm import tqdm
import json
import os
import re

# === CONFIGURATION ===
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = key)

def clean_json_text(text):
    # Remove ```json ... ``` or ``` ... ``` blocks
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

# === HELPER FUNCTION TO EVALUATE A SINGLE PROMPT/RESPONSE ===
def evaluate_response(prompt_text : str, 
                      response_text : str,
                      model : str ):
    """
    Uses the new Responses API to evaluate a single response.
    Returns a dictionary with 'score' (1-5) and 'comment'.
    """

    # Structured instruction for the model
    instruction = (
        "You are an expert AI evaluator. Your job is to critique responses to prompts "
        "and give them a score from 1 to 5, where 1 is poor and 5 is excellent. "
        "Consider structure, clarity, accuracy, and relevance as a definition for quality. "
        "Also provide a brief comment explaining the score." 
        "Do keep in mind that the AI was trained with very limited data, so do add some leniency.\n\n"
        f"Prompt: {prompt_text}\n"
        f"Response: {response_text}\n\n"
        "Step 1: Assign an integer score 1-5 based on response quality.\n"
        "Step 2: Write a brief comment explaining why you gave that score.\n"
        "Step 3: ONLY return a JSON object with keys 'score' and 'comment'."
    )

    resp = client.responses.create(
        model = model,
        input = instruction,
        temperature = 0, 
    )
    output_text = clean_json_text(resp.output_text.strip())

    try:
        result = json.loads(output_text)
        return result
    except json.JSONDecodeError:
        # fallback if model returns invalid JSON
        return {"score": -1, "comment": output_text}

# === MAIN SCRIPT ===
def main(json_file : str,
         model : str):
    # Load input JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    scores = []
    comments = []

    # Loop over each prompt/response pair
    for item in tqdm(data, desc='Agent Prompt Evaluation', unit='prompt'):
        prompt_text = item["prompt"]
        response_text = item["response"]

        evaluation = evaluate_response(prompt_text, response_text, model)
        scores += [evaluation.get("score")]
        comments += [evaluation.get("comment")]

    # Compute final usefulness score
    mean_score = sum([s for s in scores if s is not None]) / len([s for s in scores if s is not None])
    final_usefulness = round(mean_score) 

    # === SAVE RESULTS TO JSON ===
    # Ensure directory exists
    os.makedirs("agent_eval", exist_ok=True)

    # Create output filename based on input file
    base_filename = os.path.basename(json_file).replace(".json", "_eval.json")
    output_path = os.path.join("agent_eval", base_filename)

    # Prepare output data
    output_data = {
        "input_file": json_file,
        "scores": scores,
        "comments": comments,
        "final_usefulness": final_usefulness
    }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    import sys
    json_file = sys.argv[1]

    try:
        model = sys.argv[2]
    except Exception:
        model = 'gpt-4.1-mini'
    
    main(json_file, model)