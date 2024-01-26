import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Fine-tune GPT-2 model
def fine_tune_gpt2(train_file, output_dir):
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Create a dataset from your text file
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128,
    )

    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Fine-tuning arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Fine-tune GPT-2 model
    train_file = r"C:\Users\NAKSHTRA\PycharmProjects\pythonProject\48lawsofpower.txt"  # Replace with the path to your training dataset
    output_dir = r"C:\Users\NAKSHTRA\PycharmProjects\pythonProject\fine-tuned-gpt2"  # Specify the output directory for fine-tuned model
    fine_tune_gpt2(train_file, output_dir)
