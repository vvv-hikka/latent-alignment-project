# PA-CCS on the mixed dataset with OLMo, using l2+median normalization
# (the best-performing pipeline in the gemma notebook: L2 over features, then median).
latent-align run \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --strategy last-token \
  --output-dir runs/olmo_1b_mixed \
  --normalizing l2,median
