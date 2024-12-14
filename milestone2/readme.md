## Milestone 2

### Result Submission
We didn't figure out how to use tira to submit our result to the official leaderboard this week. Also explained in `Contributions` `Milestone 2` section in the progress report.

### Progress Report
The rendered progress report including all sections of milestone 1 and milestone 2 is included in `milestone 2`.

### Baseline
We adapted the `backtranslation_baseline.py` from PAN at CLEF 2024 [here](https://github.com/pan-webis-de/pan-code/tree/master/clef24/text-detoxification/baselines/backtranslation-baseline).  
- script of baseline is located at `scripts/backtranslation_baseline.py`, [here](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/scripts/backtranslation_baseline.py)
- script of evaluation is located at `evaluation_script/evaluate.py`, [here](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/evaluation_script/evaluate.py)
- The input data for baseline we used is at `data/ru_train_input.jsonl` [here](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/data/ru_train_input.jsonl), the predictions from the baseline is at `tira-output/re_train_output.jsonl` [here](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/tira-output/ru_train_output.jsonl)

**The description of the model can be found in the progress report, under `Week 2 Baseline - Backtranslation Model` section, `Model Description` subsection.**

#### W&B logs
- The link to the training run of the baseline model is [here](https://wandb.ai/speech_sanitizers/detox/runs/8wxs9iw2?nw=nwuserchenxinwang). Note that since the baseline model is not actually training any models, there isn't any training status.
- The link to the evaluation run of the result of the baseline model is
[here](https://wandb.ai/speech_sanitizers/detox/runs/fs8j8ctm?nw=nwuserchenxinwang), which includes all official evaluation metrics.

We added argument `--log_to_wb` to the official scripts so that they add logs to W&B. Example usage:
```
python scripts/backtranslation_baseline.py --input data/ru_train_input.jsonl --output tira-output/ru_train_output.jsonl --language ru --log_to_wb
```
```
python evaluation_script/evaluate.py --input=data/ru_train_input.jsonl --golden=data/ru_train_gold.jsonl --prediction=tira-output/ru_train_output.jsonl --log_to_wb
```

### Error analysis
- The sample input to the baseline model is [here](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/milestone2/en_dev_tiny.jsonl).
- The predicted output is [here](https://github.ubc.ca/wangcx12/COLX_531_speech_sanitizers/blob/main/milestone2/en_dev_tiny_output.jsonl).
- **The error analysis can be found in the progress report, under `Week 2 Baseline - Backtranslation Model` section, `Error Analysis` subsection.**




