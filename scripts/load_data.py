from datasets import load_dataset
import json

def load_data():
    # Load the ParaDetox datasets for English, split into training and validation
    en_data = load_dataset('s-nlp/paradetox', split='train')
    en_train, en_valid = en_data.train_test_split(test_size=0.1, seed=42).values()

    # Load the ParaDetox datasets for Russian, split into training and validation
    ru_train = load_dataset('s-nlp/ru_paradetox', split='train')
    ru_valid = load_dataset('s-nlp/ru_paradetox', split='validation')

    # Load the DEV datasets - multilingual ParaDetox data for evaluation
    dev_data = load_dataset('textdetox/multilingual_paradetox')
    en_dev = dev_data['en']
    ru_dev = dev_data['ru']

    return {
        "en_train": en_train,
        "en_valid": en_valid,
        "ru_train": ru_train,
        "ru_valid": ru_valid,
        "en_dev": en_dev,
        "ru_dev": ru_dev
    }

def transform_train_valid_data(dataset, output_path, lang, type):
    """
    Transforms training or validation data for both toxic (input for model) and neutral (gold) comments into the required format and writes to two separate files.
    
    Args:
    - dataset: The dataset to transform.
    - output_path: The path to the output directory (not the file).
    - lang: The language of the dataset ("en" or "ru").
    - type: The type of dataset ("train" or "valid").
    """
    toxic_key = f"{lang}_toxic_comment"
    neutral_key = f"{lang}_neutral_comment"
    
    with open(f'{output_path}/{lang}_{type}_input.jsonl', 'w', encoding='utf-8') as toxic_file, \
         open(f'{output_path}/{lang}_{type}_gold.jsonl', 'w', encoding='utf-8') as neutral_file:
        
        for idx, entry in enumerate(dataset):
            toxic_data = {
                "id": f"{lang}-{type}{idx}",
                "text": entry[toxic_key]
            }
            neutral_data = {
                "id": f"{lang}-{type}{idx}",
                "text": entry[neutral_key]
            }
            toxic_file.write(json.dumps(toxic_data, ensure_ascii=False) + '\n')
            neutral_file.write(json.dumps(neutral_data, ensure_ascii=False) + '\n')


def transform_dev_data(dataset, output_path, lang):
    """
    Transforms dev data into the required format and writes to one file.
    
    Args:
    - dataset: The dataset to transform.
    - output_path: The path to the output directory (not the file).
    - lang: The language of the dataset ("en" or "ru").
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        for idx, entry in enumerate(dataset):
            transformed_data = {
                "id": f"{lang}-dev{idx}",
                "text": entry["toxic_sentence"]
            }
            file.write(json.dumps(transformed_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    data = load_data()

    # Specify the output directory for the transformed data
    output_dir = "data"
    
    # Transform and save English training and validation data
    transform_train_valid_data(data['en_train'], f'{output_dir}', 'en', 'train')
    transform_train_valid_data(data['en_valid'], f'{output_dir}', 'en', 'valid')
    
    # Transform and save Russian training and validation data
    transform_train_valid_data(data['ru_train'], f'{output_dir}', 'ru', 'train')
    transform_train_valid_data(data['ru_valid'], f'{output_dir}', 'ru', 'valid')
    
    # Transform and save English and Russian dev data
    transform_dev_data(data['en_dev'], f'{output_dir}/en_dev.jsonl', 'en')
    transform_dev_data(data['ru_dev'], f'{output_dir}/ru_dev.jsonl', 'ru')
