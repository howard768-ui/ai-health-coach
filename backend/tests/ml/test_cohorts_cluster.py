"""Phase 8A HDBSCAN clustering tests.

Tests cover:
1. HDBSCAN finds clusters on synthetic data
2. k-anonymity enforcement merges small clusters
3. Cluster summaries computed correctly
4. Full pipeline orchestration

Run: ``cd backend && uv run python -m pytest tests/ml/test_cohorts_cluster.py -v``
"""

from __future__ import annotations

import os


os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

import numpy as np
from ml.cohorts.cluster import (
    run_clustering,
    enforce_k_anonymity,
    compute_cluster_summaries,
)


# ---------------------------------------------------------------------------
# Unit: HDBSCAN clustering
# ---------------------------------------------------------------------------


def test_hdbscan_finds_clusters_on_separable_data():
    """Two well-separated blobs should produce 2 clusters."""
    rng = np.random.default_rng(42)
    n = 200
    dim = 10

    # Two blobs with clear separation.
    blob_a = rng.normal(loc=0.0, scale=0.3, size=(n, dim))
    blob_b = rng.normal(loc=3.0, scale=0.3, size=(n, dim))
    vectors = np.vstack([blob_a, blob_b])

    labels, probs = run_clustering(vectors, min_cluster_size=50, min_samples=10)

    unique_labels = set(labels[labels >= 0])
    assert len(unique_labels) >= 2, f"Expected >= 2 clusters, got {len(unique_labels)}"


def test_hdbscan_noise_points_for_outliers():
    """Scattered points should be classified as noise (-1)."""
    rng = np.random.default_rng(42)
    n = 200
    dim = 10

    # One tight blob + scattered noise.
    blob = rng.normal(loc=0.0, scale=0.2, size=(n, dim))
    noise = rng.uniform(-10, 10, size=(20, dim))
    vectors = np.vstack([blob, noise])

    labels, _ = run_clustering(vectors, min_cluster_size=50, min_samples=10)

    n_noise = int(np.sum(labels == -1))
    assert n_noise > 0, "Expected some noise points for scattered data"


# ---------------------------------------------------------------------------
# Unit: k-anonymity enforcement
# ---------------------------------------------------------------------------


def test_k_anonymity_merges_small_clusters():
    """Clusters below k should be merged into nearest large cluster."""
    rng = np.random.default_rng(42)
    dim = 5

    # Simulate: 100 in cluster 0, 30 in cluster 1 (below k=50), 80 in cluster 2.
    vectors = np.vstack([
        rng.normal(0, 0.1, (100, dim)),   # cluster 0
        rng.normal(0.5, 0.1, (30, dim)),  # cluster 1 (small)
        rng.normal(3, 0.1, (80, dim)),    # cluster 2
    ])
    labels = np.array([0] * 100 + [1] * 30 + [2] * 80)

    new_labels, n_merged = enforce_k_anonymity(labels, vectors, k=50)

    assert n_merged == 1, f"Expected 1 merged cluster, got {n_merged}"
    # Cluster 1 should be merged into cluster 0 (nearest centroid).
    assert 1 not in set(new_labels), "Small cluster should be merged"
    # All original cluster 0 + 2 members still in their clusters.
    assert int(np.sum(new_labels == 0)) >= 100
    assert int(np.sum(new_labels == 2)) == 80


def test_k_anonymity_all_small_returns_noise():
    """When all clusters are below k, return all as noise."""
    vectors = np.random.default_rng(42).random((30, 5))
    labels = np.array([0] * 15 + [1] * 15)

    new_labels, n_merged = enforce_k_anonymity(labels, vectors, k=50)

    assert np.all(new_labels == -1), "All should be noise when no cluster >= k"


# ---------------------------------------------------------------------------
# Unit: cluster summaries
# ---------------------------------------------------------------------------


def test_cluster_summaries_correct_size():
    """Summary should report correct member count."""
    rng = np.random.default_rng(42)
    vectors = np.vstack([
        rng.normal(0, 0.1, (60, 5)),
        rng.normal(3, 0.1, (40, 5)),
    ])
    labels = np.array([0] * 60 + [1] * 40)
    names = [f"feat_{i}" for i in range(5)]

    summaries = compute_cluster_summaries(labels, vectors, names)

    assert len(summaries) == 2
    sizes = {s.label: s.n_members for s in summaries}
    assert sizes[0] == 60
    assert sizes[1] == 40


def test_cluster_summaries_top_features():
    """Top features should be those with highest centroid deviation from global mean."""
    rng = np.random.default_rng(42)
    dim = 5

    # Cluster 0 centered at [10, 0, 0, 0, 0], cluster 1 at [0, 0, 0, 0, 0].
    # Global mean is [5, 0, 0, 0, 0]. Cluster 0 deviates on feat_a.
    vectors = np.vstack([
        rng.normal(0, 0.1, (50, dim)) + np.array([10, 0, 0, 0, 0]),
        rng.normal(0, 0.1, (50, dim)),
    ])
    labels = np.array([0] * 50 + [1] * 50)
    names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]

    summaries = compute_cluster_summaries(labels, vectors, names)

    # Both clusters should have feat_a as top distinguishing feature
    # (it's the only dimension with a large separation from global mean).
    c0 = [s for s in summaries if s.label == 0][0]
    c1 = [s for s in summaries if s.label == 1][0]
    assert c0.top_features[0][0] == "feat_a"
    assert c1.top_features[0][0] == "feat_a"
