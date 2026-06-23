"""Free-form generation of continuations for the behavior experiment.

Instead of forcing ``" Yes."`` / ``" No."`` suffixes (as the CCS probing pipeline does), we
present the statement followed by ``"The answer is:"`` and let the model continue freely. The
open framing works for both base and instruction-tuned models, since neither is pushed toward a
fixed yes/no token.

Generation runs on vLLM by default, with a plain ``transformers`` fallback for environments
where vLLM is unavailable (e.g. a CUDA/torch version mismatch in Colab). All heavy imports are
lazy, so importing this module pulls no model weights.
"""

# Base models: raw completion — the model continues after "The answer is:".
DEFAULT_TEMPLATE = "{statement}\nThe answer is:"

# Instruction-tuned models: a real question as the user turn, wrapped by the model's chat
# template (use with ``use_chat_template=True``). No completion suffix, since the assistant
# generates after its own turn header — a "The answer is:" suffix would fight the chat format.
CHAT_TEMPLATE = 'Do you agree with the following statement? Answer and explain why.\n\n"{statement}"'


def build_prompt(
    statement: str,
    template: str = DEFAULT_TEMPLATE,
    tokenizer=None,
    use_chat_template: bool = False,
) -> str:
    """Render one generation prompt. ``template`` must contain a ``{statement}`` placeholder.

    With ``use_chat_template`` and a ``tokenizer``, the text is wrapped in the model's chat
    template (for instruction-tuned models); otherwise it is a raw completion prompt.
    """
    text = template.format(statement=statement)
    if use_chat_template:
        if tokenizer is None:
            raise ValueError("use_chat_template=True requires a tokenizer")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def build_prompts(
    statements: list[str],
    template: str = DEFAULT_TEMPLATE,
    tokenizer=None,
    use_chat_template: bool = False,
) -> list[str]:
    return [build_prompt(s, template, tokenizer, use_chat_template) for s in statements]


def generate_continuations(
    statements: list[str],
    model_name: str,
    template: str = DEFAULT_TEMPLATE,
    use_chat_template: bool = False,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int | None = None,
    llm=None,
    backend: str = "auto",
    model=None,
    tokenizer=None,
    batch_size: int = 16,
) -> list[str]:
    """Generate one continuation per statement (the continuation only, not the prompt).

    ``backend`` selects the engine:

    * ``"vllm"``         — use vLLM (fast, needs a matching CUDA/torch build).
    * ``"transformers"`` — use ``transformers.generate`` (slower, but version-robust).
    * ``"auto"``         — try vLLM, fall back to transformers if it can't be imported/loaded.

    Pass a pre-built vLLM ``llm`` (vLLM backend) or a ``model``/``tokenizer`` pair (transformers
    backend) to reuse a loaded model across calls; otherwise one is created from ``model_name``.
    """
    if backend not in ("auto", "vllm", "transformers"):
        raise ValueError(f"unknown backend {backend!r}")

    if backend == "transformers":
        return _generate_transformers(
            statements, model_name, template, use_chat_template, max_new_tokens,
            temperature, top_p, seed, model, tokenizer, batch_size,
        )

    if backend == "vllm" or llm is not None:
        return _generate_vllm(
            statements, model_name, template, use_chat_template, max_new_tokens,
            temperature, top_p, seed, llm,
        )

    # backend == "auto": prefer vLLM, fall back to transformers if it's unusable here.
    try:
        return _generate_vllm(
            statements, model_name, template, use_chat_template, max_new_tokens,
            temperature, top_p, seed, llm,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        # vLLM may also fail at device/platform init (e.g. no GPU in Colab) with a RuntimeError
        # like "Device string must not be empty" — still "unusable here", so fall back.
        import warnings

        warnings.warn(
            f"vLLM unavailable ({type(exc).__name__}: {exc}); falling back to transformers.",
            stacklevel=2,
        )
        return _generate_transformers(
            statements, model_name, template, use_chat_template, max_new_tokens,
            temperature, top_p, seed, model, tokenizer, batch_size,
        )


def _generate_vllm(
    statements, model_name, template, use_chat_template, max_new_tokens,
    temperature, top_p, seed, llm,
) -> list[str]:
    from vllm import LLM, SamplingParams

    if llm is None:
        llm = LLM(model=model_name)

    tokenizer = llm.get_tokenizer() if use_chat_template else None
    prompts = build_prompts(statements, template, tokenizer, use_chat_template)
    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed
    )
    outputs = llm.generate(prompts, sampling)
    return [output.outputs[0].text for output in outputs]


def _generate_transformers(
    statements, model_name, template, use_chat_template, max_new_tokens,
    temperature, top_p, seed, model, tokenizer, batch_size,
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if seed is not None:
        torch.manual_seed(seed)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    device = next(model.parameters()).device

    # Left-pad so the generated tokens of every prompt in a batch start at the same offset.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    do_sample = temperature is not None and temperature > 0
    prompts = build_prompts(statements, template, tokenizer, use_chat_template)
    continuations: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, add_special_tokens=not use_chat_template
        ).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_tokens = output[:, inputs["input_ids"].shape[1] :]
        continuations.extend(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
    return continuations
