tira-run \
    --input-dataset pan23-text-detoxification/dev-de-20240305-training \
	--image webis/clef24-text-detoxification-baseline-backtranslation:0.0.1 \
	--command '/backtranslation_baseline.py --input ${inputDataset}/input.jsonl --output ${outputDir}/references.jsonl --src_lang_id de'