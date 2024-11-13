import torch
from tokenizers import Tokenizer
from train import GPT, GPTConfig  # Assuming your model definition is in train.py

import torch.nn.functional as F

def nucleus_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    logits[sorted_indices[sorted_indices_to_remove]] = -float('Inf')
    probabilities = F.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probabilities, num_samples=1).item()
    return next_token_id

def load_model_and_tokenizer():
    # Load the model configuration and tokenizer
    config = GPTConfig()
    model = GPT(config)
    model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    tokenizer = Tokenizer.from_file("az_tokenizer.json")  # Load tokenizer
    return model, tokenizer

def apply_repetition_penalty(logits, input_ids, penalty=1.2):
    # Penalize the logits for tokens that have already been generated
    for token_id in set(input_ids):
        logits[0, token_id] /= penalty
    return logits

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.001, p=0.95, repetition_penalty=1.5, device='cpu'):
    model = model.to(device)
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            output_logits, _ = model(input_tensor)
        
        # Apply temperature scaling
        logits = output_logits[:, -1, :] / temperature
        
        # Apply repetition penalty
        logits = apply_repetition_penalty(logits.clone(), input_ids, penalty=repetition_penalty)
        
        # Use nucleus sampling
        next_token_id = nucleus_sampling(logits[0], p=p)
        
        input_ids.append(next_token_id)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        if next_token_id == tokenizer.token_to_id('[END]'):  # Replace with actual end token if applicable
            break
    
    generated_text = tokenizer.decode(input_ids)
    return generated_text.replace(' i ', ' ')  # Example: minor post-processing to clean up spaces


def main():
    model, tokenizer = load_model_and_tokenizer()
    prompt = "Azərbaycanın tarixi"  # Your input prompt
    generated_text = generate_text(model, tokenizer, prompt, p=0.9)  # Adjust p as needed
    print("Generated Text:", generated_text)

if __name__ == '__main__':
    main()
