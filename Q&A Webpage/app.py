from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the fine-tuned GPT-2 model and tokenizer
model_name = r"C:\Users\NAKSHTRA\PycharmProjects\pythonProject\fine-tuned-gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        question = request.form['question']

        # Tokenize the question
        input_ids = tokenizer.encode(question, return_tensors="pt")

        # Generate a response from the model
        output = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

        # Decode the generated output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return render_template('index.html', question=question, answer=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
