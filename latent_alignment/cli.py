from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from latent_alignment.ccs import ProbeConfig, summarize_results, train_ccs_layers
from latent_alignment.data import load_dataset
from latent_alignment.extract import extract_texts, load_embeddings, load_hf_model, save_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(prog="latent-align")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract hidden states for a dataset")
    _add_data_args(extract_parser)
    extract_parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    extract_parser.add_argument("--output", required=True, help="Output .npz path")
    extract_parser.add_argument(
        "--model-kind",
        default="auto",
        choices=["auto", "encoder", "decoder", "encoder-decoder"],
    )
    extract_parser.add_argument(
        "--strategy",
        default=None,
        choices=["first-token", "last-token", "mean", "custom"],
    )
    extract_parser.add_argument("--layer-index", type=int, default=None)
    extract_parser.add_argument("--one-layer", action="store_true")
    extract_parser.add_argument("--device", default=None)
    extract_parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    extract_parser.add_argument("--trust-remote-code", action="store_true")
    extract_parser.add_argument("--max-length", type=int, default=512)

    probe_parser = subparsers.add_parser("probe", help="Train PA-CCS on saved embeddings")
    _add_data_args(probe_parser)
    _add_probe_args(probe_parser)
    probe_parser.add_argument("--embeddings", required=True, help="Input .npz from extract")
    probe_parser.add_argument("--output-dir", required=True)

    run_parser = subparsers.add_parser("run", help="Extract hidden states and train PA-CCS")
    _add_data_args(run_parser)
    _add_probe_args(run_parser)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument(
        "--model-kind",
        default="auto",
        choices=["auto", "encoder", "decoder", "encoder-decoder"],
    )
    run_parser.add_argument(
        "--strategy",
        default=None,
        choices=["first-token", "last-token", "mean", "custom"],
    )
    run_parser.add_argument("--layer-index", type=int, default=None)
    run_parser.add_argument("--one-layer", action="store_true")
    run_parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    run_parser.add_argument("--trust-remote-code", action="store_true")
    run_parser.add_argument("--max-length", type=int, default=512)

    args = parser.parse_args()
    if args.command == "extract":
        command_extract(args)
    elif args.command == "probe":
        command_probe(args)
    elif args.command == "run":
        command_run(args)


def command_extract(args) -> None:
    dataset = _load_cli_dataset(args)
    model, tokenizer, device = load_hf_model(
        args.model,
        model_kind=args.model_kind,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    positive = extract_texts(
        dataset.positive_texts,
        model,
        tokenizer,
        layer_index=args.layer_index,
        get_all_layers=not args.one_layer,
        strategy=args.strategy,
        model_kind=args.model_kind,
        device=device,
        max_length=args.max_length,
    )
    negative = extract_texts(
        dataset.negative_texts,
        model,
        tokenizer,
        layer_index=args.layer_index,
        get_all_layers=not args.one_layer,
        strategy=args.strategy,
        model_kind=args.model_kind,
        device=device,
        max_length=args.max_length,
    )
    if positive.ndim == 2:
        positive = positive[:, None, :]
        negative = negative[:, None, :]
    save_embeddings(args.output, positive, negative)


def command_probe(args) -> None:
    dataset = _load_cli_dataset(args)
    positive, negative = load_embeddings(args.embeddings)
    _probe_and_write(args, dataset, positive, negative)


def command_run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npz"
    extract_args = argparse.Namespace(**vars(args), output=embeddings_path)
    command_extract(extract_args)
    dataset = _load_cli_dataset(args)
    positive, negative = load_embeddings(embeddings_path)
    _probe_and_write(args, dataset, positive, negative)


def _probe_and_write(args, dataset, positive: np.ndarray, negative: np.ndarray) -> None:
    train_idx, test_idx = dataset.train_test_indices(
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=not args.no_stratify,
    )
    normalizings = _normalizings(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multiple = len(normalizings) > 1
    combined_rows: list[dict[str, object]] = []

    for normalizing in normalizings:
        config = ProbeConfig(
            nepochs=args.nepochs,
            ntries=args.ntries,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            lambda_classification=args.lambda_classification,
            normalizing=normalizing,
            seed=args.seed,
        )
        results = train_ccs_layers(
            positive,
            negative,
            dataset.labels,
            train_idx,
            test_idx,
            config=config,
            opposite_indices=dataset.opposite_indices,
            device=args.device,
        )

        target_dir = output_dir / _normalizing_dirname(normalizing) if multiple else output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        rows = summarize_results(results)
        pd.DataFrame(rows).to_csv(target_dir / "ccs_summary.csv", index=False)
        for row in rows:
            combined_rows.append({"normalizing": normalizing, **row})

        full_results = {
            f"layer_{layer}_{key}": value
            for layer, row in results.items()
            for key, value in row.items()
        }
        np.savez_compressed(target_dir / "ccs_full_results.npz", **full_results)
        metadata = {
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "probe_config": config.__dict__,
        }
        (target_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    if multiple:
        pd.DataFrame(combined_rows).to_csv(output_dir / "ccs_summary.csv", index=False)


def _normalizings(args) -> list[str]:
    """Requested normalization pipelines, de-duplicated and order-preserving."""
    values = args.normalizing
    if isinstance(values, str):
        values = [values]
    return list(dict.fromkeys(values))


def _normalizing_dirname(normalizing: str) -> str:
    """Filesystem-safe subdirectory name for one normalization pipeline."""
    name = normalizing.replace("+", ",").replace(",", "-").strip()
    return f"norm_{name}" if name else "norm_raw"


def _load_cli_dataset(args):
    return load_dataset(
        args.dataset,
        dataset_format=args.dataset_format,
        positive_col=args.positive_col,
        negative_col=args.negative_col,
        text_col=args.text_col,
        label_col=args.label_col,
        pair_id_col=args.pair_id_col,
        positive_suffix=args.positive_suffix,
        negative_suffix=args.negative_suffix,
    )


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, help="CSV/JSON/JSONL dataset path")
    parser.add_argument(
        "--dataset-format",
        default="auto",
        choices=["auto", "paired", "polarity_raw"],
    )
    parser.add_argument("--positive-col", default="positive_text")
    parser.add_argument("--negative-col", default="negative_text")
    parser.add_argument("--text-col", default="statement")
    parser.add_argument("--label-col", default="is_harmfull_opposition")
    parser.add_argument("--pair-id-col", default=None)
    parser.add_argument("--positive-suffix", default=" Yes.")
    parser.add_argument("--negative-suffix", default=" No.")


def _add_probe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=71)
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument("--nepochs", type=int, default=1500)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-classification", type=float, default=0.0)
    parser.add_argument(
        "--normalizing",
        nargs="+",
        default=["mean"],
        help=(
            "One or more normalization pipelines to try, e.g. "
            "--normalizing median l2,median l2. Each token is a single pipeline "
            "(combine steps inside one with comma/plus, e.g. l2,median). "
            "When several are given, every pipeline is trained and compared."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)


if __name__ == "__main__":
    main()
