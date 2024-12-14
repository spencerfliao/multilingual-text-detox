#!/usr/bin/env python3

import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
import os
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
import logging
import wandb
from transformers import BartTokenizerFast, T5TokenizerFast
from backtranslation_baseline import get_model, detoxify_batch

class DetoxDataset(Dataset):
    """
    A PyTorch Dataset class for loading and tokenizing text data for detoxification tasks.

    Attributes:
        input_texts (list): A list of input (toxic) texts loaded from the input JSONL file.
        gold_texts (list): A list of target (detoxified) texts loaded from the gold JSONL file.
        tokenizer: The tokenizer used for encoding the texts.
        max_len (int): Maximum length of the tokenized input and target sequences.
    """
    def __init__(self, input_file_obj, gold_file_obj, tokenizer, max_len=64):
        self.input_texts = [json.loads(line)["text"] for line in input_file_obj]
        self.gold_texts = [json.loads(line)["text"] for line in gold_file_obj]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        source_text = self.input_texts[idx]
        target_text = self.gold_texts[idx]

        source_encodings = self.tokenizer(
            source_text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        target_encodings = self.tokenizer(
            target_text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source_encodings.input_ids.flatten(),
            "attention_mask": source_encodings.attention_mask.flatten(),
            "labels": target_encodings.input_ids.flatten(),
        }


def load_datasets(
    tokenizer, train_input, train_gold, valid_input, valid_gold, batch_size=32
):
    """
    Initializes the DetoxDataset class instances and creates DataLoader 
    instances for both the training and validation datasets to facilitate batch processing 
    during model training and validation.

    Args:
        tokenizer: The tokenizer to be used for tokenizing the input and gold texts.
        train_input (file object): An open file object for the training input JSONL file.
        train_gold (file object): An open file object for the training gold JSONL file.
        valid_input (file object): An open file object for the validation input JSONL file.
        valid_gold (file object): An open file object for the validation gold JSONL file.
        batch_size (int, optional): The size of the batches of data. Defaults to 32.

    Returns:
        tuple: A tuple containing the training DataLoader (`train_loader`) and the
               validation DataLoader (`valid_loader`).
    """
    train_dataset = DetoxDataset(train_input, train_gold, tokenizer)
    valid_dataset = DetoxDataset(valid_input, valid_gold, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def initialize_optimizers(model, train_loader, num_epochs=4):
    """
    Initializes the model, tokenizer, optimizer, and scheduler based on the specified language.

    Args:
        model : The pretrained model to finetune.
        train_loader (DataLoader): The DataLoader instance for the training dataset.
        num_epochs (int, optional): The number of epochs for which the model will be trained. Defaults to 4.

    Returns:
        tuple: A tuple containing the optimizer and scheduler
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs
    )
    return optimizer, scheduler


def train_one_epoch(model, device, train_loader, optimizer, scheduler):
    """
    Trains the model for one epoch through the entire training dataset.

    Args:
        model: The model to be trained.
        device: The device (CPU or CUDA) on which the model is located.
        train_loader (DataLoader): The DataLoader instance that provides batches of training data.
        optimizer: The optimizer used for updating the model parameters based on computed gradients.
        scheduler: The learning rate scheduler used to adjust the learning rate over epochs.

    Returns:
        float: The average loss over all batches in the training dataset for the current epoch.
    """
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, device, valid_loader):
    """
    Validates the model on a validation dataset.

    Args:
        model: The model to be validated.
        device: The device (CPU or CUDA) on which the model is located.
        valid_loader (DataLoader): The DataLoader instance that provides batches of validation data.

    Returns:
        float: The average loss over all batches in the validation dataset.
    """
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            total_val_loss += loss.item()
    return total_val_loss / len(valid_loader)


def fine_tune_model(
    num_epochs,
    model,
    device,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    patience,
):
    """
    Fine-tunes the model on the training dataset and evaluates it on the validation dataset.
    It implements early stopping based on validation loss to prevent overfitting. 
    The model with the lowest validation loss is saved.

    Args:
        num_epochs (int): The number of epochs to train the model.
        model: The model to be fine-tuned.
        device: The device (CPU or CUDA) on which the model is located.
        train_loader (DataLoader): The DataLoader instance for the training data.
        valid_loader (DataLoader): The DataLoader instance for the validation data.
        optimizer: The optimizer used for updating the model parameters.
        scheduler: The learning rate scheduler for adjusting the learning rate.
        patience (int): The number of epochs to wait for improvement in validation loss before stopping.

    Returns:
        None: The function saves the best model state and potentially triggers early stopping.
    """
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(
            model, device, train_loader, optimizer, scheduler
        )
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

        avg_val_loss = validate(model, device, valid_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


def generate_detox_texts(model, tokenizer, inputs, output_file, batch_size):
    model.eval()
    texts = [entry["text"] for entry in inputs]
    doc_ids = [entry["id"] for entry in inputs]
    detoxified_texts = detoxify_batch(texts, model, tokenizer, batch_size)
    
    for doc_id, text in zip(doc_ids, detoxified_texts):
        output_file.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False))
        output_file.write("\n")


def main():
    """
    Main function to handle argument parsing and orchestrate the detoxification training and prediction workflow.
    """
    parser = argparse.ArgumentParser(
        description="PAN 2024 text detoxification task"
        "Finetune the baseline for for English and Russian"
        "Then use finetuned model to generate the detoxified output"
    )
    parser.add_argument(
        "--mode",
        choices=["fine-tune", "use-existing"],
        required=True,
        help="Mode of operation: 'fine-tune' a new model or 'use-existing' model.",
    )
    parser.add_argument(
        "--train_input",
        type=argparse.FileType("r", encoding="utf-8"),
        help="Training Input JSONL file (required for fine-tuning).",
    )
    parser.add_argument(
        "--train_gold",
        type=argparse.FileType("r", encoding="utf-8"),
        help="Training Gold JSONL file (required for fine-tuning).",
    )
    parser.add_argument(
        "--valid_input",
        type=argparse.FileType("r", encoding="utf-8"),
        help="Validation Input JSONL file (required for fine-tuning).",
    )
    parser.add_argument(
        "--valid_gold",
        type=argparse.FileType("r", encoding="utf-8"),
        help="Validation Gold JSONL file (required for fine-tuning).",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        type=argparse.FileType("w", encoding="utf-8"),
        help="Path to save the detoxified output JSONL file.",
    )
    parser.add_argument(
        "--model_path",
        help="Path to save the fine-tuned model or to load an existing model (depending on the mode).",
    )
    parser.add_argument(
        "--language",
        required=True,
        type=str,
        choices=["ru", "en"],
        help="Language of the input data. Should be one of" "['ru', 'en']",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for translation and detoxification.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs for finetuning.",
    )
    parser.add_argument(
        "--log_to_wb",
        action="store_true",
        help="Flag to log the run status to Weights & Biases."
    )
    args = parser.parse_args()

    if args.log_to_wb:
        file_path = './wbapi.json'
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                api = data["key"]
                wandb.login(key=api)
            if args.language == "en":
                run_name = "Finetuned-Bart-en"
            elif args.language == "ru":
                run_name = "Finetuned-T5-ru"
            wandb.init(project="detox", name=run_name, entity="speech_sanitizers")
        except FileNotFoundError:
            print(f"Warning: '{file_path}' not found. Proceeding without Weights & Biases logging.")

    logging.basicConfig(level=logging.INFO)

    if args.mode == "fine-tune":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.language == "en":
            model, tokenizer = get_model("en_detoxifier")
        elif args.language == "ru":
            model, tokenizer = get_model("ru_detoxifier")
        patience = 2
        train_loader, valid_loader = load_datasets(
            tokenizer,
            args.train_input,
            args.train_gold,
            args.valid_input,
            args.valid_gold,
            args.batch_size,
        )
        optimizer, scheduler = initialize_optimizers(model, train_loader, args.epochs)
        fine_tune_model(
            args.epochs,
            model,
            device,
            train_loader,
            valid_loader,
            optimizer,
            scheduler,
            patience,
        )
        if args.model_path:
            torch.save(model, args.model_path)
        args.train_input.seek(0)
        inputs = [json.loads(line) for line in args.train_input]
        generate_detox_texts(model, tokenizer, inputs, args.output_file, args.batch_size)

    elif args.mode == "use-existing":
        if not args.model_path or not os.path.exists(args.model_path):
            raise ValueError("Valid model path is required to use an existing model.")
        model = torch.load(args.model_path)
        if args.language == "en":
            tokenizer = BartTokenizerFast.from_pretrained("s-nlp/bart-base-detox")
        elif args.language == "ru":
            tokenizer = T5TokenizerFast.from_pretrained("s-nlp/ruT5-base-detox")
        inputs = [json.loads(line) for line in args.train_input]
        generate_detox_texts(model, tokenizer, inputs, args.output_file, args.batch_size)

    if args.log_to_wb:
        wandb.finish()

if __name__ == "__main__":
    main()