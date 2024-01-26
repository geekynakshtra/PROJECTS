from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned GPT-2 model and tokenizer
model_name = r"C:\Users\NAKSHTRA\PycharmProjects\pythonProject\fine-tuned-gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Example questions
questions = [
    "Can you give me an example from history where the enemy was crushed totally?",
    "What's the point of making myself less accessible?",
    "Can you tell me the story of Queen Elizabeth I from this 48 laws of power book?"
]

# Generate responses for each question
for question in questions:
    # Tokenize the question
    input_ids = tokenizer.encode(question, return_tensors="pt")

    # Generate a response from the model
    output = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print the generated response
    print(f"Question: {question}")
    print(f"Generated Response: {generated_text}\n")
