"""Unit tests for the shared Naive Bayes core.

The interesting cases are all failure modes rather than accuracy: a constant
feature column, a class that never appeared in the fit data, and a posterior
that has to move when an input moves. Accuracy on two separated clusters is
here only to confirm the arithmetic is a classifier at all.
"""

import sys
import types

import numpy as np
import pytest

# semseg/types.py belongs to another module of this repo. Prefer the real one
# and stub only if it is genuinely absent, so this test cannot keep passing
# against a stale NUM_CLASSES once the real file lands.
try:
    import semseg.types  # noqa: F401
except ImportError:
    _semseg = sys.modules.setdefault("semseg", types.ModuleType("semseg"))
    if not hasattr(_semseg, "__path__"):
        _semseg.__path__ = []
    _stub = types.ModuleType("semseg.types")
    _stub.NUM_CLASSES = 9
    _stub.UNLABELED = 255
    sys.modules["semseg.types"] = _stub

from baselines.common.nb import GaussianNB, VAR_FLOOR  # noqa: E402
from semseg.types import NUM_CLASSES  # noqa: E402

CLUSTER_A = 3
CLUSTER_B = 6
CLUSTER_SPREAD = 0.3
CLUSTER_N = 400


def two_clusters(seed=0, n=CLUSTER_N, spread=CLUSTER_SPREAD):
    """Two well separated isotropic Gaussians in 2D, labelled with two
    non-adjacent goose9 ids so an off by one in the class indexing shows up."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(0.0, 0.0), scale=spread, size=(n, 2))
    b = rng.normal(loc=(4.0, 4.0), scale=spread, size=(n, 2))

    X = np.vstack([a, b])
    y = np.concatenate([np.full(n, CLUSTER_A), np.full(n, CLUSTER_B)])
    return X, y


def test_recovers_two_clusters():
    X, y = two_clusters(seed=0)
    X_test, y_test = two_clusters(seed=1)

    labels, conf = GaussianNB().fit(X, y).predict(X_test)

    assert (labels == y_test).mean() > 0.99
    assert conf.min() >= 0.5   # the winner of a two class vote always has half
    assert conf.max() <= 1.0


def test_proba_is_normalised():
    X, y = two_clusters()
    proba = GaussianNB().fit(X, y).predict_proba(X)

    assert proba.shape == (X.shape[0], NUM_CLASSES)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0.0)


def test_constant_column_is_safe():
    """A constant feature column has zero variance. errstate(all="raise")
    turns any divide by zero or invalid operation into a test failure, so this
    cannot pass vacuously by nobody looking at the warnings."""
    X, y = two_clusters()
    X = np.column_stack([X, np.full(X.shape[0], 7.0)])

    with np.errstate(all="raise"):
        model = GaussianNB().fit(X, y)
        proba = model.predict_proba(X)
        labels, conf = model.predict(X)

    assert np.all(np.isfinite(proba))
    assert not np.any(np.isnan(proba))
    assert np.isfinite(conf).all()
    assert model.var[CLUSTER_A, 2] == VAR_FLOOR
    assert set(np.unique(labels)) <= {CLUSTER_A, CLUSTER_B}


def test_absent_class_never_predicted():
    X, y = two_clusters()
    model = GaussianNB().fit(X, y)

    with np.errstate(all="raise"):
        proba = model.predict_proba(X)
        labels, _ = model.predict(X)

    absent = [c for c in range(NUM_CLASSES) if c not in (CLUSTER_A, CLUSTER_B)]
    assert np.all(np.isneginf(model.log_prior[absent]))
    assert np.all(proba[:, absent] == 0.0)
    assert not np.any(np.isnan(proba))
    assert set(np.unique(labels)) == {CLUSTER_A, CLUSTER_B}


def test_absent_class_far_from_data():
    """The zero prior has to hold even for a query nowhere near either fitted
    cluster, where the likelihoods are all tiny and the normalisation is at
    its most fragile."""
    X, y = two_clusters()
    model = GaussianNB().fit(X, y)

    # underflow is excluded on purpose and only here: exp() of a log posterior
    # this far below the row maximum legitimately flushes to zero, which is
    # the right answer. A divide, an invalid operation or an overflow would be
    # a real bug, so those still raise.
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        proba = model.predict_proba(np.array([[1e6, -1e6]]))

    assert np.isclose(proba.sum(), 1.0)
    assert not np.any(np.isnan(proba))
    assert proba.argmax() in (CLUSTER_A, CLUSTER_B)


def test_posterior_moves_on_nudge():
    """Perturbation case: shifting one sample towards the other cluster must
    move its posterior. A classifier that ignored its input would pass every
    test above."""
    X, y = two_clusters()
    model = GaussianNB().fit(X, y)

    # queried on the midpoint between the two clusters, where the posterior is
    # near 0.5 and a small move is measurable. Deep inside a cluster the same
    # nudge moves a posterior of 1e-41 to 1e-39, which is a real change that
    # no threshold in absolute terms can see.
    query = np.array([[2.0, 2.0]])
    before = model.predict_proba(query)[0, CLUSTER_B]

    after = model.predict_proba(query + 0.05)[0, CLUSTER_B]

    assert after > before
    assert after - before > 1e-3


def test_label_flips_across_boundary():
    """The same perturbation, taken far enough to move the decision itself."""
    X, y = two_clusters()
    model = GaussianNB().fit(X, y)

    labels_low, _ = model.predict(np.array([[1.5, 1.5]]))
    labels_high, _ = model.predict(np.array([[2.5, 2.5]]))

    assert labels_low[0] == CLUSTER_A
    assert labels_high[0] == CLUSTER_B


def test_prior_shifts_with_class_counts():
    """Perturbation case on the prior rather than the likelihood: duplicating
    one class's samples must move the posterior towards it."""
    X, y = two_clusters()
    balanced = GaussianNB().fit(X, y)

    extra = y == CLUSTER_B
    skewed = GaussianNB().fit(np.vstack([X, X[extra]]), np.concatenate([y, y[extra]]))

    query = np.array([[2.0, 2.0]])
    assert skewed.predict_proba(query)[0, CLUSTER_B] > balanced.predict_proba(query)[0, CLUSTER_B]
    assert skewed.log_prior[CLUSTER_B] > balanced.log_prior[CLUSTER_B]


def test_empty_fit_raises():
    with pytest.raises(ValueError):
        GaussianNB().fit(np.zeros((0, 3)), np.zeros(0, dtype=np.int64))


def test_mismatched_labels_raise():
    X, y = two_clusters()
    with pytest.raises(ValueError):
        GaussianNB().fit(X, y[:-1])
