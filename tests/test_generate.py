from pathlib import Path

import pandas as pd
import pytest

from latent_alignment.data import load_statements
from latent_alignment import generate
from latent_alignment.generate import build_prompt, generate_continuations


def test_build_prompt_raw() -> None:
    prompt = build_prompt("The sky is green.", template="{statement}\nAgree?")
    assert prompt == "The sky is green.\nAgree?"


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = messages[0]["content"]
        return f"<user>{content}</user><assistant>"


def test_build_prompt_chat_template() -> None:
    prompt = build_prompt(
        "Statement.",
        template="{statement} ok",
        tokenizer=_FakeTokenizer(),
        use_chat_template=True,
    )
    assert prompt == "<user>Statement. ok</user><assistant>"


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError):
        generate_continuations(["s"], "model", backend="jax")


def test_backend_auto_falls_back_to_transformers(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise ImportError("libcudart.so.13: cannot open shared object file")

    monkeypatch.setattr(generate, "_generate_vllm", boom)
    monkeypatch.setattr(
        generate, "_generate_transformers", lambda *a, **k: ["fallback"]
    )

    with pytest.warns(UserWarning, match="falling back to transformers"):
        out = generate_continuations(["s"], "model", backend="auto")
    assert out == ["fallback"]


def test_backend_transformers_skips_vllm(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("vLLM should not be touched for backend='transformers'")

    monkeypatch.setattr(generate, "_generate_vllm", boom)
    monkeypatch.setattr(
        generate, "_generate_transformers", lambda *a, **k: ["ok"]
    )

    assert generate_continuations(["s"], "model", backend="transformers") == ["ok"]


def test_load_statements(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "statement": ["harm a", "harm b", "safe a", "safe b"],
            "is_harmfull_opposition": [0, 0, 1, 1],
        }
    ).to_csv(path, index=False)

    df = load_statements(path)

    assert list(df.columns) == ["statement", "label", "pair_id"]
    assert df["statement"].tolist() == ["harm a", "harm b", "safe a", "safe b"]
    assert df["label"].tolist() == [0, 0, 1, 1]
    # First and second halves share pair ids so opposites can be matched.
    assert df["pair_id"].tolist() == [0, 1, 0, 1]
