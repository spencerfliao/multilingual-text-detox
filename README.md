# Multilingual Text Detoxification

This project aims to address the challenge of online toxicity by transforming toxic comments into non-toxic ones while maintaining the original meaning. It involves the use of advanced natural language processing techniques and machine learning models to ensure content safety and inclusiveness in digital communication platforms.
This repository contains the implementation of a multilingual text detoxification system developed as part of the TextDetox 2024 challenge. The system focuses on converting toxic text into non-toxic text while preserving the original content's meaning. The models handle multiple languages with a focus on English and Russian.

## Task Description

Text detoxification is critical for creating safer and more inclusive online environments. This process involves the following steps:
1. **Style Transfer:** Transform toxic phrases into non-toxic ones while maintaining non-toxicity levels.
2. **Content Preservation:** Ensure the transformed sentences retain the same meaning as the original toxic sentences.
3. **Grammatical Correctness:** Maintain grammatical integrity and readability in the transformed text.

## Datasets

We utilize the official ParaDetox datasets for English and Russian, comprising pairs of toxic and detoxified comments. Each dataset involves a set training and validation splits, detailed as follows:

- **English Dataset:** 17,769 training pairs, 1,975 validation pairs.
- **Russian Dataset:** 11,090 training pairs, 1,116 validation pairs.

## Implementation

### Setup

To set up the project environment and install dependencies, run:

```bash
conda env create -f speech_sanitizers_env.yml
conda activate textdetox
```

### File Structure

- **data/**: Contains training, validation, and development datasets.
  - `en_train_input.json`, `en_valid_input.json`, `ru_train_input.json`, `ru_valid_input.json`: Input data for training and validation.
  - `en_train_gold.json`, `en_valid_gold.json`, `ru_train_gold.json`, `ru_valid_gold.json`: Gold standard detoxified outputs for training and validation.
- **output/**: Stores the output from various model runs.
  - `en_generated_detox_texts_f1.json`, `ru_generated_detox_texts_f1.json`: Outputs from the first fine-tuning iteration.
  - `en_valid_output_gpt2_1.json`, `ru_valid_output_gpt2_1.json`: Validation outputs from the GPT-2 models.
- **scripts/**: Scripts for training and evaluating models.
  - `backtranslation_baseline.py`: Implements the baseline model using backtranslation.
  - `finetune_baseline.py`: Script for fine-tuning the baseline models.
  - `finetune_condBERT.py`, `gpt2.ipynb`: Scripts and notebooks for fine-tuning and evaluating CondBERT and GPT-2 models.
  - `evaluate.py`: Script to evaluate outputs against the gold standard using specified metrics.

### Baseline Model

- Utilizes backtranslation for detoxification.
- To run the baseline model: `sh run_baseline.sh`

### Fine-tuned Models

- CondBERT and GPT-2 Models: Advanced models fine-tuned for better performance and accuracy in detoxifying texts.
- For fine-tuning and generating outputs: `python scripts/finetune_baseline.py`

## Results

- Our models demonstrate a significant improvement in handling toxicity while retaining the content and grammatical correctness.
- To evaluate model outputs: `python scripts/evaluate.py`

### Evaluation Metrics

Our evaluation framework includes:
- **Style Transfer Accuracy (STA):** Measures non-toxicity levels of transformed text.
- **Content Preservation (SIM):** Assesses semantic similarity between original and transformed text.
- **Character n-gram F-score (ChrF):** Evaluates grammatical and fluency aspects of the output text.

## Acknowledgments

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Authors

Minsi Lai
Chenxin Wang
Jingyi Liao
Fangge Liao

University of British Columbia
CLEF 2024 Organizers
All participants and contributors to the project.
