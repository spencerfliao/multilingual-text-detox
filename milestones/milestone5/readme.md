### Script
- This week we implement [finetune_condBERT.py](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/scripts/finetune_condBERT.py). 


### Model Output - Predictions
List of models output files:  
- `en_valid_condbert_output.jsonl`: generated from model `en-finetuned-condbert-1.pth` with `en_valid_input.jsonl` as the input toxic texts.
- `ru_valid_condbert_output.jsonl`: generated from model `ru-finetuned-condbert-1.pth` with `ru_valid_input.jsonl` as the input toxic texts.

**These output files are located under `COLX_531_speech_sanitizers/tira-output/`**

### Saved Models
See `COLX_531_speech_sanitizers/models/`

### W&B logs
- Finetune 1 - `en-finetuned-condbert-1.pth`: [finetuned-condBERT-en](https://wandb.ai/speech_sanitizers/detox/runs/bdlw666i?nw=nwuserchenxinwang)
- Evaluation of Finetune 1: [finetuned-condBERT-en_eval](https://wandb.ai/speech_sanitizers/detox/runs/jthkuvi9?nw=nwuserchenxinwang)
- Finetune 2 - `ru-finetuned-condbert-1.pth`: [finetuned-condBERT-ru]()
- Evaluation of Finetune 2: [finetuned-condBERT-ru_eval]()
