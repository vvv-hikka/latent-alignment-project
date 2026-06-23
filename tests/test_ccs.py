import numpy as np

from latent_alignment.ccs import ProbeConfig, _normalize_split, train_ccs_layers


def test_train_ccs_layers_smoke() -> None:
    rng = np.random.default_rng(0)
    positive = rng.normal(size=(8, 2, 4)).astype(np.float32)
    negative = (positive - 0.5).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    train_idx = np.array([0, 1, 2, 4, 5, 6])
    test_idx = np.array([3, 7])

    results = train_ccs_layers(
        positive,
        negative,
        labels,
        train_idx,
        test_idx,
        config=ProbeConfig(nepochs=2, ntries=1, batch_size=-1),
        opposite_indices=np.array([4, 5, 6, 7, 0, 1, 2, 3]),
        device="cpu",
    )

    assert set(results) == {0, 1}
    assert "accuracy" in results[0]


def test_normalize_split_accepts_pipeline() -> None:
    pos_train = np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    pos_test = np.array([[9.0, 12.0]], dtype=np.float32)
    neg_train = np.array([[4.0, 3.0], [8.0, 6.0]], dtype=np.float32)
    neg_test = np.array([[12.0, 9.0]], dtype=np.float32)

    out = _normalize_split(pos_train, pos_test, neg_train, neg_test, "l2,median")

    assert len(out) == 4
    assert all(np.isfinite(arr).all() for arr in out)
