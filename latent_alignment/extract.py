from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

PoolingStrategy = Literal["first-token", "last-token", "mean", "custom"]
ModelKind = Literal["auto", "encoder", "decoder", "encoder-decoder"]


DECODER_TYPES = {
    "bloom",
    "falcon",
    "gemma",
    "gpt",
    "gpt2",
    "gpt_neo",
    "gpt_neox",
    "gptj",
    "llama",
    "mistral",
    "mixtral",
    "olmo",
    "opt",
    "qwen2",
    "qwen3",
}
ENCODER_TYPES = {"albert", "bert", "deberta", "deberta-v2", "distilbert", "roberta"}


def load_hf_model(
    model_name: str,
    *,
    model_kind: ModelKind = "auto",
    device: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
):
    torch_dtype = _resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    _ensure_padding_token(tokenizer)

    loader = AutoModelForCausalLM if model_kind == "decoder" else AutoModel
    model = loader.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    return model, tokenizer, target_device


def detect_model_kind(model) -> str:
    if getattr(model.config, "is_encoder_decoder", False):
        return "encoder-decoder"
    model_type = getattr(model.config, "model_type", "").lower()
    if model_type in ENCODER_TYPES:
        return "encoder"
    if model_type in DECODER_TYPES:
        return "decoder"
    return "decoder" if getattr(model.config, "is_decoder", False) else "unknown"


def recommended_strategy(model_kind: str) -> PoolingStrategy:
    if model_kind == "encoder":
        return "first-token"
    if model_kind == "decoder":
        return "last-token"
    return "mean"


def extract_texts(
    texts: list[str],
    model,
    tokenizer,
    *,
    layer_index: int | None = None,
    get_all_layers: bool = True,
    strategy: PoolingStrategy | None = None,
    model_kind: ModelKind = "auto",
    use_decoder: bool = False,
    device: str | torch.device | None = None,
    max_length: int = 512,
    token_number: int | None = None,
) -> np.ndarray:
    detected_kind = detect_model_kind(model) if model_kind == "auto" else model_kind
    strategy = strategy or recommended_strategy(detected_kind)
    device = torch.device(device or getattr(model, "device", "cpu"))

    vectors = [
        extract_representation(
            text,
            model,
            tokenizer,
            layer_index=layer_index,
            get_all_layers=get_all_layers,
            strategy=strategy,
            model_kind=detected_kind,
            use_decoder=use_decoder,
            device=device,
            max_length=max_length,
            token_number=token_number,
        )
        for text in tqdm(texts, desc="Extracting hidden states")
    ]
    return np.stack(vectors).astype(np.float32)


def extract_representation(
    text: str,
    model,
    tokenizer,
    *,
    layer_index: int | None = None,
    get_all_layers: bool = True,
    strategy: PoolingStrategy = "last-token",
    model_kind: str | None = None,
    use_decoder: bool = False,
    device: torch.device | None = None,
    max_length: int = 512,
    token_number: int | None = None,
) -> np.ndarray:
    device = device or torch.device(getattr(model, "device", "cpu"))
    model_kind = model_kind or detect_model_kind(model)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    attention_mask = inputs.get("attention_mask")

    with torch.no_grad():
        if model_kind == "encoder-decoder":
            decoder_start_token_id = model.config.decoder_start_token_id or 0
            decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=device)
            outputs = model(
                **inputs,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
            )
        else:
            outputs = model(**inputs, output_hidden_states=True)

    hidden_states = _select_hidden_states(outputs, use_decoder=use_decoder)
    if get_all_layers:
        reps = [
            _pool_hidden_state(layer.float(), attention_mask, strategy, token_number)
            for layer in hidden_states
        ]
        return np.stack(reps).astype(np.float32)

    if layer_index is None:
        layer_index = len(hidden_states) // 2
    layer_index = min(layer_index, len(hidden_states) - 1)
    return _pool_hidden_state(
        hidden_states[layer_index].float(), attention_mask, strategy, token_number
    ).astype(np.float32)


def save_embeddings(path: str | Path, positive: np.ndarray, negative: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, positive=positive, negative=negative)


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["positive"], data["negative"]


def _select_hidden_states(outputs, *, use_decoder: bool):
    if use_decoder and getattr(outputs, "decoder_hidden_states", None) is not None:
        return outputs.decoder_hidden_states
    if getattr(outputs, "encoder_hidden_states", None) is not None:
        return outputs.encoder_hidden_states
    return outputs.hidden_states


def _pool_hidden_state(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    strategy: PoolingStrategy,
    token_number: int | None,
) -> np.ndarray:
    if strategy == "first-token":
        pooled = hidden[:, 0, :]
    elif strategy == "last-token":
        if attention_mask is None:
            pooled = hidden[:, -1, :]
        else:
            last_idx = attention_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
    elif strategy == "mean":
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    elif strategy == "custom":
        if token_number is None:
            raise ValueError("token_number is required for custom pooling")
        pooled = hidden[:, min(token_number, hidden.shape[1] - 1), :]
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")

    result = pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.where(np.isfinite(result), result, 0.0)


def _ensure_padding_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")
