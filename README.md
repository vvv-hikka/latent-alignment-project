# latent-alignment-project

Reusable code for PA-CCS style latent-alignment probes on HuggingFace models.

Based on the public `SadSabrina/polarity-probing` repository, with the notebook experiment
moved into reusable modules and a `latent-align` CLI.

## Install

```bash
uv venv
uv pip install -e ".[dev]"        # core deps + pytest/ruff
```

For the behavior experiment (free-form generation), add the GPU-only extra on a GPU host:

```bash
uv pip install -e ".[behavior]"   # adds vLLM
```

Then:

```bash
latent-align --help
pytest -q
ruff check .
```

## Included data

Reference datasets from `polarity-probing` are vendored here:

```text
data/polarity_probing/raw/mixed_dataset.csv
data/polarity_probing/raw/not_dataset.csv
```

They use the original `statement` + first-half/second-half pairing convention. Pass
`--dataset-format polarity_raw` when using them.

## Run an experiment

OLMo on the mixed dataset with median normalization:

```bash
latent-align run \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --strategy last-token \
  --output-dir runs/olmo_1b_mixed \
  --normalizing median
```

For bigger models, set `--dtype bfloat16` or `--dtype float16` if your hardware supports it.

## Reuse extracted hidden states

Extraction is the expensive part. Save embeddings once:

```bash
latent-align extract \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --output runs/olmo_mixed_embeddings.npz
```

Then train probes repeatedly:

```bash
latent-align probe \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --embeddings runs/olmo_mixed_embeddings.npz \
  --output-dir runs/olmo_mixed_probe \
  --normalizing median
```

Outputs:

```text
ccs_summary.csv       layerwise accuracy, silhouette, PC, CI, bias
ccs_full_results.npz  arrays: weights, per-example PC/CI
metadata.json         split indices and probe config
embeddings.npz        only for `run`
```

## Normalization

Hidden states are normalized before the probe is trained. Statistics (per-feature mean/median)
are fitted on the train split only and applied to both splits, so no test information leaks.

| step           | effect                                            |
| -------------- | ------------------------------------------------- |
| `mean`         | subtract the per-feature train **mean** (default) |
| `median`       | subtract the per-feature train **median**         |
| `l2`           | scale each row to unit L2 norm                    |
| `raw` / `none` | no normalization                                  |

### One pipeline

A single argument is one pipeline. Combine steps with a comma or `+`, applied left to right:

```bash
--normalizing median          # center on the median
--normalizing l2,median       # L2-normalize, then median-center
--normalizing l2+median       # same as above
```

`l2,median` was the strongest configuration in the original experiments.

### Several pipelines in one run

Pass more than one pipeline to train and compare them in a single invocation. Space separates
pipelines; comma/`+` combines steps inside one:

```bash
latent-align run \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --strategy last-token \
  --output-dir runs/olmo_mixed_norm_sweep \
  --normalizing mean median l2 l2,median
```

Works for both `run` and `probe`. With a single pipeline, files are written directly to
`--output-dir`. With several:

```text
<output-dir>/
  ccs_summary.csv          combined summary with an extra `normalizing` column
  norm_mean/               per-pipeline artifacts (ccs_summary.csv, *.npz, metadata.json)
  norm_median/
  norm_l2/
  norm_l2-median/
```

## Метрики

### CCS-проба (`ccs_summary.csv`, по строке на слой)

- **`layer`** — индекс слоя скрытых состояний.
- **`accuracy`** — доля верно классифицированных yes/no-пар на тесте. `0.5` = случайно,
  `→1.0` = чисто линейно разделимо. CCS не знает знака кластеров, поэтому в коде уже берётся
  `max(acc, 1 − acc)`.
- **`silhouette`** — насколько разделены кластеры yes/no в пространстве `positive − negative`
  (косинусная метрика). Диапазон `[−1, 1]`; выше = чётче разделение.
- **`polar_consistency_mean`** — знаковое квадратичное расхождение пробы между формулировками,
  которые должны совпадать (`A.Yes` ≡ `not-A.No`, `A.No` ≡ `not-A.Yes`). По модулю `≈0` =
  согласованная полярность; большая величина = рассогласование. **Ближе к 0 лучше.**
- **`contradiction_index_mean`** — степень, в которой проба «соглашается со всем»: высокая
  вероятность одновременно для `A` и для `not-A`
  (`p(A.Yes)·p(¬A.Yes) + p(A.No)·p(¬A.No)`). **Ниже лучше.**
- **`bias`** — свободный член линейной пробы. Диагностика смещения по слою, не показатель
  качества.

Полные массивы (веса, по-примерные PC/CI) — в `ccs_full_results.npz`.

### Behavior eval (guardrail-разметка генераций)

Модель свободно генерирует продолжение (без принудительного формата Yes/No), затем три
guardrail размечают ответ. Каждый guardrail возвращает `{"unsafe", "score", "raw", "categories"}`:

- **`unsafe`** (bool) — вердикт: нарушает ли ответ политику безопасности.
- **`score`** (float) — уверенность / `P(нарушение)`. Для ShieldGemma = `P(Yes)` softmax по
  токенам Yes/No, порог `0.5`. Для остальных `1.0`/`0.0` или вероятность из вывода модели.
- **`categories`** — коды нарушенных политик (`S1`, `S2`, …), если модель их вернула.

Отдельно `extract_stance` (regex по началу ответа) даёт `yes`/`no`/`unclear` — соглашается ли
генерация с утверждением; используется для оценки выравнивания без формата Yes/No.

Агрегаты: **unsafe rate** = доля `unsafe` среди генераций; вердикты трёх guardrail на одних и
тех же генерациях сравниваются между собой (согласие/расхождение моделей).

## Own data

Preferred CSV/JSONL format is one row per harmful/safe pair:

```csv
positive_text,negative_text,label,pair_id
"Harmful or target-side statement","Safe or opposite-side statement",0,0
```

Run it with:

```bash
latent-align run \
  --dataset data/my_pairs.csv \
  --dataset-format paired \
  --positive-col positive_text \
  --negative-col negative_text \
  --label-col label \
  --pair-id-col pair_id \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --output-dir runs/my_pairs_olmo
```

If you keep the original polarity-probing layout, put all first-side statements in the first
half of the file and matching opposite-side statements in the second half.

For CCS, the loader turns each base statement into two prompts by appending ` Yes.` and ` No.`.
Override with `--positive-suffix` / `--negative-suffix` for different answer tokens or another
language.

## Notes

- `--model-kind decoder` + `--strategy last-token`: recommended for Qwen, OLMo, Llama, Gemma,
  Mistral-style models.
- `--model-kind encoder` + `--strategy first-token`: recommended for BERT/DeBERTa.
- The CLI uses all hidden-state layers by default. Use `--one-layer --layer-index N` for a
  faster single-layer run.
