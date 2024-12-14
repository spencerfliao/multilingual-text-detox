# Multilingual Text Detoxification (TextDetox) 2024

This project aims to detoxify text by transforming toxic content into non-toxic versions while preserving the original meaning as much as possible. Our approach focuses on handling multiple languages, with initial implementations for English and Russian.

## Project Timeline

- **February 1, 2024:** Data availability and run submissions open.
- **May 6, 2024:** Deadline for run submissions; results announced.
- **May 31, 2024:** Deadline for paper submissions.

## Task Description

Text detoxification is critical for creating safer and more inclusive online environments. This process involves the following steps:
1. **Style Transfer:** Transform toxic phrases into non-toxic ones while maintaining non-toxicity levels.
2. **Content Preservation:** Ensure the transformed sentences retain the same meaning as the original toxic sentences.
3. **Grammatical Correctness:** Maintain grammatical integrity and readability in the transformed text.

## Datasets

We utilize the official ParaDetox datasets for English and Russian, comprising pairs of toxic and detoxified comments. Each dataset involves a set training and validation splits, detailed as follows:

- **English Dataset:** 17,769 training pairs, 1,975 validation pairs.
- **Russian Dataset:** 11,090 training pairs, 1,116 validation pairs.

## Evaluation Metrics

Our evaluation framework includes:
- **Style Transfer Accuracy (STA):** Measures non-toxicity levels of transformed text.
- **Content Preservation (SIM):** Assesses semantic similarity between original and transformed text.
- **Character n-gram F-score (ChrF):** Evaluates grammatical and fluency aspects of the output text.

## Model Implementation

We have implemented models using BART and T5 architectures for English and Russian respectively. The models are trained to optimize for the above evaluation metrics.

### Baseline Model

- Utilizes backtranslation for detoxification.
- [Model and training details](https://github.com/path/to/model_details)

### Fine-tuned Models

- Improved versions of baseline models with adjusted learning rates and epochs.
- [Fine-tuning specifics](https://github.com/path/to/fine_tuning_details)

## Results

Our models demonstrate a significant improvement in handling toxicity while retaining the content and grammatical correctness. Detailed performance metrics are available in the [results section](https://github.com/path/to/results).

## Setup and Installation

```bash
git clone https://github.com/your-repository/TextDetox.git
cd TextDetox
pip install -r requirements.txt
