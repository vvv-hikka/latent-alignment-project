from latent_alignment.guardrails import (
    extract_stance,
    parse_nandi,
    parse_safe_unsafe,
    parse_xguard,
)


def test_extract_stance_yes() -> None:
    assert extract_stance("Yes, I agree with that completely.") == "yes"


def test_extract_stance_no() -> None:
    assert extract_stance("No, that is false and harmful.") == "no"


def test_extract_stance_disagree_phrase_is_no() -> None:
    # "I don't agree" embeds the word "agree" but is disagreement.
    assert extract_stance("I don't agree with this at all.") == "no"


def test_extract_stance_unclear() -> None:
    assert extract_stance("That is an interesting question to consider.") == "unclear"


def test_extract_stance_first_signal_wins() -> None:
    assert extract_stance("No. Some might say yes, but I disagree.") == "no"


def test_extract_stance_empty() -> None:
    assert extract_stance("") == "unclear"


def test_parse_safe_unsafe_takes_last_verdict() -> None:
    text = "Let me reason: this could be safe, but on reflection it is unsafe."
    result = parse_safe_unsafe(text)
    assert result["unsafe"] is True


def test_parse_safe_unsafe_safe() -> None:
    result = parse_safe_unsafe("After analysis, the response is safe.")
    assert result["unsafe"] is False


def test_parse_safe_unsafe_no_verdict() -> None:
    result = parse_safe_unsafe("no clear decision here")
    assert result["unsafe"] is False
    assert result["score"] == 0.0


def test_parse_xguard_takes_first_verdict() -> None:
    # reason_first=False => the leading verdict wins, even if the explanation says otherwise.
    text = "unsafe. On reflection one might call parts of this safe, but it is unsafe."
    assert parse_xguard(text)["unsafe"] is True


def test_parse_xguard_safe_label() -> None:
    # The model's safe label is "Safe-Safe"; its leading "Safe" is the verdict.
    result = parse_xguard("Safe-Safe\nNo risk categories apply.")
    assert result["unsafe"] is False
    assert result["score"] == 0.0


def test_parse_xguard_no_verdict() -> None:
    assert parse_xguard("could not determine")["unsafe"] is False


def test_parse_nandi_unsafe_json() -> None:
    result = parse_nandi('{"unsafe": true, "categories": ["violence"], "score": 0.97}')
    assert result["unsafe"] is True
    assert result["categories"] == ["violence"]
    assert result["score"] == 0.97


def test_parse_nandi_safe_via_safe_key() -> None:
    result = parse_nandi('Verdict: {"safe": true}')
    assert result["unsafe"] is False
    assert result["categories"] == []


def test_parse_nandi_label_string() -> None:
    assert parse_nandi('{"label": "unsafe"}')["unsafe"] is True


def test_parse_nandi_non_json_falls_back() -> None:
    # Falls back to the free-text safe/unsafe scan when output isn't JSON.
    assert parse_nandi("After review the response is unsafe.")["unsafe"] is True
