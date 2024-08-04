from transformers import AutoTokenizer

def analyze_token_length(text, model_name_or_path):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Tokenize the text without truncation
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Get the number of tokens
    token_length = len(tokens)

    return token_length, tokens

if __name__ == "__main__":
    # Example usage
    model_name_or_path = 'gpt2'  # Replace with your model name or path
    #model_name_or_path = 'gpt-4o-mini'
    model_max_length = 2048  # Replace with the model's maximum sequence length

    text = "Your example string here."
    signal_descr = "The TwoLeadECG time series data exhibits a complex pattern characterized by several distinct segments. Initially, the signal shows a slight increase from 0.03 to 0.29, indicating a gradual rise in amplitude, followed by a sharp decline that reaches a minimum of -3.15, suggesting a significant negative deflection, possibly indicative of a critical event or anomaly in the cardiac cycle. This downward trend is marked by a series of negative values, with the lowest point occurring around the 30th data point. Following this trough, the signal experiences a pronounced recovery, with values rising steadily from -3.15 to a peak of 1.26, reflecting a strong positive response. The latter part of the series demonstrates a gradual decrease in amplitude, oscillating between positive values and returning to near-zero levels, indicating a stabilization phase. Overall, the time series captures a dynamic cardiac event with an initial rise, a sharp decline, a significant trough, and a subsequent recovery, culminating in a return to a more stable state."
    #token_length, delta = analyze_token_length(signal_descr, model_name_or_path, model_max_length)
    token_length, tokens = analyze_token_length("[0.10, 0.20, 0.30]", model_name_or_path)
    print(f"Token length of input text: {token_length}")
    print(f"Tokens: {tokens}")

