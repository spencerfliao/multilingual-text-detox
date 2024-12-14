# Milestone 3

## Progress Report
The link to the progress report: [here](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/Progress_report.pdf)

## Models
Model description, Reflection and Result & Error analysis are detailed in the updated progress report.

### Script
This week we implement [finetune_baseline.py](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/scripts/finetune_baseline.py) to make efforts to improve over the baseline models for English and Russian.  

Example usage of `finetune_baseline.py`, in project root:  
Finetune a model based on language and generate output; if provided `--model_path`, model will be saved:  
```
python scripts/finetune_baseline.py \
    --mode fine-tune \
    --train_input data/ru_train_input.jsonl \
    --train_gold data/ru_train_gold.jsonl \
    --valid_input data/ru_valid_input.jsonl \
    --valid_gold data/ru_valid_gold.jsonl \
    --output_file tira-output/ru_generated_detox_texts.jsonl \
    --model_path models/ru_finetuned-t5.pth \
    --language ru \
    --log_to_wb
```
Load a model from `--model_path` and generate output:  
```
python scripts/finetune_baseline.py \
    --mode use-existing \
    --model_path models/finetuned-t5.pth \
    --train_input data/ru_train_input.jsonl \
    --output_file tira-output/ru_generated_detox_texts.jsonl \
    --language ru \
    --log_to_wb
```

### Model Output - Predictions
**These output files are located under `COLX_531_speech_sanitizers/tira-output/`**  
List of models output files (new from this week):  
- `ru_generated_detox_texts_f1.jsonl`: generated from model `finetuned-t5-f1.pth`, a fintunied T5 model for Russian, with `ru_train_input.jsonl` as the input toxic texts.
- `ru_generated_detox_texts_f2.jsonl`: generated from model `finetuned-t5-f2.pth`, a fintunied T5 model for Russian, with `ru_train_input.jsonl` as the input toxic texts.
- `en_train_output.jsonl`: generated from the baseline Bart model for English, with `en_train_input.jsonl`: as the input toxic texts.
- `en_generated_detox_texts_f1.jsonl`: generated from model `en_finetuned-bart-f1.pth`, a fintunied Bart model for English, with `en_train_input.jsonl` as the input toxic texts.

(from milestone 2):  
- `ru_train_output.jsonl`: generated from the baseline T5 model for Russian, with `ru_train_input.jsonl`: as the input toxic texts.

We are comparing the results from train datasets for T5 and finetuned T5, Bart and finetuned Bart in details, in the **Result analysis** of the progress report.  

**Predictions on the dev datasets, located under `COLX_531_speech_sanitizers/milestone3/`**:  
- `en_dev_output_finetuned-bart-f1.jsonl`: generated from model `en_finetuned-bart-f1.pth`, a fintunied Bart model for English, with `en_dev.jsonl` as the input toxic texts.
- `ru_dev_output_finetuned-bart-f2.jsonl`: generated from model `finetuned-t5-f2.pth`, a fintunied T5 model for Russian, with `ru_dev.jsonl` as the input toxic texts.

We are analyzing the predictions in the **Error analysis** of the progress report. 

### Saved Models
See `COLX_531_speech_sanitizers/models/`

### W&B logs
- `finetuned-t5-f1.pth`: [finetuned-T5-ru-1](https://wandb.ai/speech_sanitizers/detox/runs/m6bvu9ny?nw=nwuserchenxinwang)
- Evaluation of `finetuned-t5-f1.pth`: [finetuned-T5-ru-1_eval](https://wandb.ai/speech_sanitizers/detox/runs/mekngii1?nw=nwuserchenxinwang)
- `finetuned-t5-f2.pth`: [finetuned-T5-ru-2](https://wandb.ai/speech_sanitizers/detox/runs/m6bvu9ny?nw=nwuserchenxinwang)
- Evaluation of `finetuned-t5-f2.pth`: [finetuned-T5-ru-2_eval](https://wandb.ai/speech_sanitizers/detox/runs/mekngii1?nw=nwuserchenxinwang)


- English Baseline Bart: [baseline-Bart-en](https://wandb.ai/speech_sanitizers/detox/runs/nd1bmnlv)
- Evaluation of English Baseline Bart: [baseline-Bart-en_eval](https://wandb.ai/speech_sanitizers/detox/runs/s9n42x41?nw=nwuserchenxinwang)
- `en_finetuned-bart-f1.pth`: [finetuned-Bart-en-f1](https://wandb.ai/speech_sanitizers/detox/runs/q74avvi3?nw=nwuserchenxinwang)
- Evaluation of `en_finetuned-bart-f1.pth`: [finetuned-Bart-en-f1_eval](https://wandb.ai/speech_sanitizers/detox/runs/5l5fpgzi?nw=nwuserchenxinwang)
