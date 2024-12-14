# Milestone 4

## Progress Report
The link to the progress report: [here](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/Progress_report.pdf)

## Models
Model description, Reflection and Result & Error analysis are detailed in the updated progress report.

### Script
- This week we implement [gpt2.ipynb](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/scripts/gpt2.ipynb). 
- We built fintuned GPT-2 models for both Russian and English detoxification task. 
- We are keeping this week's work in `.ipynb`, becuase we encountered some issues related to the system's locale settings while trying to run external Python scripts in Colab, which we haven't resolved yet. Therefore, we decided to embed the functions from this week's model Python script directly into the Colab notebook, instead of running them as external scripts.


### Model Output - Predictions
List of models output files:  
- `en_valid_output_gpt2_1.jsonl`: generated from model `en-gpt2-1.pth` with `en_valid_input.jsonl` as the input toxic texts.
- `ru_valid_output_gpt2_1.jsonl`: generated from model `ru-gpt2-1.pth` with `ru_valid_input.jsonl` as the input toxic texts.

**These output files are located under `COLX_531_speech_sanitizers/tira-output/`**

### Saved Models
See `COLX_531_speech_sanitizers/models/`

### W&B logs
- Finetune 1 - `en-gpt2-1.pth`: [finetuned-gpt2-en](https://wandb.ai/speech_sanitizers/detox/runs/ou4qn976)
- Evaluation of Finetune 1: [finetuned-gpt2-en_eval](https://wandb.ai/speech_sanitizers/detox/runs/n00wmd7g)
- Finetune 2 - `ru-gpt2-1.pth`: [finetuned-gpt2-ru](https://wandb.ai/speech_sanitizers/detox/runs/xpm68lzk)
- Evaluation of Finetune 2: [finetuned-gpt2-ru_eval](https://wandb.ai/speech_sanitizers/detox/runs/xfg5nclm)
