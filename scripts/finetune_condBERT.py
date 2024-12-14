#!/usr/bin/env python3

import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
import os

from torch.optim import AdamW
import numpy as np
import logging
import wandb
from transformers import BertTokenizer, BertForMaskedLM

import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

####################################################################################################
###################################Classifier for toxic words#######################################
####################################################################################################
russian_stop_words = [
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но',
    'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще',
    'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть',
    'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего',
    'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам',
    'чтоб', 'без', 'будто', 'человек', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот',
    'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы',
    'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой',
    'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве',
    'сказать', 'всю', 'три'
]


def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            data.append(json_data)
    return pd.DataFrame(data)


def prepare_data_for_cls(toxic_path, non_toxic_path):
    toxic_data = load_data(toxic_path)
    nontoxic_data = load_data(non_toxic_path)
    toxic_data['label'] = 1  # toxic
    nontoxic_data['label'] = 0  # non-toxic
    combined_data = pd.concat([toxic_data, nontoxic_data], ignore_index=True)
    texts = combined_data['text']
    labels = combined_data['label']
    return texts, labels


def train_cls(texts, labels, lang):
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    if lang == 'english':
        vectorizer = TfidfVectorizer(stop_words=lang) # lang='english' or 'russian'
    elif lang == 'russian':
        vectorizer = TfidfVectorizer(stop_words=russian_stop_words)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    classifier = LogisticRegression()
    classifier.fit(X_train_tfidf, y_train)
    return vectorizer, classifier


####################################################################################################
##################################### Finetuning BERT model#########################################
####################################################################################################
class DetoxDataset(Dataset):
    def __init__(self, toxic_file_obj, detox_file_obj, tokenizer, toxic_words, max_len=128):
        self.toxic_data = [json.loads(line) for line in toxic_file_obj]
        self.detox_data = [json.loads(line) for line in detox_file_obj]
        self.tokenizer = tokenizer
        self.toxic_words = set(toxic_words)
        self.max_len = max_len

    def __len__(self):
        return len(self.toxic_data)

    def __getitem__(self, idx):
        toxic_text = self.toxic_data[idx]["text"]
        detox_text = self.detox_data[idx]["text"]

        tokens = self.tokenizer.tokenize(toxic_text)
        masked_tokens = [self.tokenizer.mask_token if token in self.toxic_words else token for token in tokens]
        encoded_inputs = self.tokenizer.encode_plus(
            masked_tokens,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encoded = self.tokenizer.encode_plus(
            detox_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_inputs['input_ids'].squeeze(0),
            'attention_mask': encoded_inputs['attention_mask'].squeeze(0),
            'labels': target_encoded['input_ids'].squeeze(0)
        }


def train(model, loader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}")


####################################################################################################
####################################### Generate output ############################################
####################################################################################################
def prepare_input_and_mask(text, toxic_words, tokenizer):
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_masks = [1] * len(input_ids)

    for i, token in enumerate(tokens):
        if token in toxic_words:
            input_ids[i] = tokenizer.mask_token_id
    
    return torch.tensor([input_ids]), torch.tensor([attention_masks])


def perform_style_transfer(input_ids, attention_masks, model, tokenizer):
    with torch.no_grad():
        # Process the input through the model
        outputs = model(input_ids=input_ids, attention_mask=attention_masks)
        predictions = outputs[0]

    predicted_tokens = []
    for i, idx in enumerate(input_ids[0]):
        if idx == tokenizer.mask_token_id:
            # Get the top 5 candidate indices
            top_preds = torch.topk(predictions[0, i], 5).indices.tolist()
            for pred_idx in top_preds:
                token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
                if token not in ['[PAD]', '[CLS]', '[SEP]', '[MASK]']:  # Filter out non-suitable predictions
                    predicted_tokens.append(token)
                    break
            else:
                predicted_tokens.append('[UNK]')  # Use unknown token if no suitable prediction is found
        else:
            predicted_tokens.append(tokenizer.convert_ids_to_tokens([idx])[0])

    return tokenizer.convert_tokens_to_string(predicted_tokens)


def detoxify(text, toxic_words, tokenizer, model, device):
    input_ids, attention_masks = prepare_input_and_mask(text, toxic_words, tokenizer)
    
    # Move tensors to the same device as the model
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    
    detoxified_text = perform_style_transfer(input_ids, attention_masks, model, tokenizer)
    return detoxified_text


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
        "--toxic_path",
        help="Training Input JSONL file (required for fine-tuning).",
    )
    parser.add_argument(
        "--non_toxic_path",
        help="Training Gold JSONL file (required for fine-tuning).",
    )
    parser.add_argument(
        "--valid_toxic_path",
        help="Validation Input JSONL file (required for fine-tuning).",
    )
    parser.add_argument(
        "--output_file",
        required=True,
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
        "--batch_size",
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
                run_name = "Finetuned-condBERT-en"
            elif args.language == "ru":
                run_name = "Finetuned-condBERT-ru"
            wandb.init(project="detox", name=run_name, entity="speech_sanitizers")
        except FileNotFoundError:
            print(f"Warning: '{file_path}' not found. Proceeding without Weights & Biases logging.")

    logging.basicConfig(level=logging.INFO)

    if args.language == "en":
        lang = 'english'
        threshold = 1.69
        model_name = 'bert-base-uncased'
    elif args.language == "ru":
        lang = 'russian'
        threshold = 2.0
        model_name = "DeepPavlov/rubert-base-cased"

    texts, labels = prepare_data_for_cls(args.toxic_path, args.non_toxic_path)
    vectorizer, classifier = train_cls(texts, labels, lang)
    feature_to_coef = { word: coef for word, coef in zip(vectorizer.get_feature_names_out(), classifier.coef_[0])}
    
    toxic_words = {word: coef for word, coef in feature_to_coef.items() if coef > threshold}

    tokenizer = BertTokenizer.from_pretrained(model_name)

    with open(args.toxic_path, 'r', encoding='utf-8') as input_file, \
        open(args.non_toxic_path, 'r', encoding='utf-8') as gold_file:
        dataset = DetoxDataset(input_file, gold_file, tokenizer, toxic_words)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "fine-tune":
        
        model = BertForMaskedLM.from_pretrained(model_name)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        train(model, loader, optimizer, device, args.epochs)

        if args.model_path:
            torch.save(model, args.model_path)


    elif args.mode == "use-existing":
        model = torch.load(args.model_path)
        model = model.to(device)


    valid_toxic_data = load_data(args.valid_toxic_path)
    valid_toxic_data['detoxified_text'] = valid_toxic_data['text'].apply(lambda x: detoxify(x, toxic_words, tokenizer, model, device))

    with open(args.output_file, 'w', encoding='utf-8') as file:
            for _, row in valid_toxic_data.iterrows():
                if args.language == "ru":
                    data = {'id': row['id'], 'text': row['detoxified_text'].encode('utf-8').decode('unicode-escape')}
                else:
                    data = {'id': row['id'], 'text': row['detoxified_text']}
                file.write(json.dumps(data) + '\n')

    if args.log_to_wb:
        wandb.finish()

if __name__ == "__main__":
    main()