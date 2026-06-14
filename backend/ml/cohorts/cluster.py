"""HDBSCAN clustering for cross-user health archetypes.

Takes the anonymized, DP-noised pattern vectors from
``ml_anonymized_vectors`` and clusters them with HDBSCAN. Post-hoc
k-anonymity enforcement merges small clusters into their nearest
large neighbor. Cluster summaries (centroid, size, top distinguishing
features) are persisted to ``ml_cohorts``.

All heavy imports (numpy, sklearn, hdbscan) are lazy inside function
bodies per the cold-boot contract.

Entry point is ``run_clustering_pipeline``, called from ``ml.api``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select, update

if TYPE_CHECKING:
    import numpy as np
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClusterSummary:
    """One cluster's metadata."""

    label: int
    n_members: int
    centroid: list[float]
    top_features: list[tuple[str, float]]  # (name, importance)


@dataclass
class ClusteringReport:
    """Summary of a full clustering run."""

    run_id: str
    n_users: int = 0
    n_clusters: int = 0
    n_noise_points: int = 0
    largest_cluster: int = 0
    smallest_cluster: int = 0
    clusters_merged: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def run_clustering(
    vectors: "np.ndarray",
    min_cluster_size: int = 50,
    min_samples: int = 25,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Run HDBSCAN on the pattern vectors.

    Returns (labels, probabilities). Labels == -1 for noise points.
    """
    import hdbscan
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vectors)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        metric="euclidean",
        core_dist_n_jobs=1,  # single-threaded for determinism
    )
    labels = clusterer.fit_predict(X_scaled)
    probabilities = clusterer.probabilities_

    return labels, probabilities


def enforce_k_anonymity(
    labels: "np.ndarray",
    vectors: "np.ndarray",
    k: int = 50,
) -> tuple["np.ndarray", int]:
    """Merge clusters smaller than k into the nearest large cluster.

    Returns (new_labels, n_merged). Clusters that cannot be merged
    (no large cluster exists) are set to -1 (noise).
    """
    import numpy as np

    unique_labels = set(labels[labels >= 0])
    if not unique_labels:
        return labels, 0

    # Classify clusters by size.
    large_clusters = []
    small_clusters = []
    for lbl in unique_labels:
        count = int(np.sum(labels == lbl))
        if count >= k:
            large_clusters.append(lbl)
        else:
            small_clusters.append(lbl)

    if not large_clusters:
        # All clusters too small: set all to noise.
        return np.full_like(labels, -1), len(small_clusters)

    # Compute centroids of large clusters.
    large_centroids = np.array([
        vectors[labels == lbl].mean(axis=0) for lbl in large_clusters
    ])

    # Merge each small cluster into its nearest large cluster.
    new_labels = labels.copy()
    merged = 0
    for sc in small_clusters:
        sc_centroid = vectors[labels == sc].mean(axis=0)
        distances = np.linalg.norm(large_centroids - sc_centroid, axis=1)
        nearest_idx = int(np.argmin(distances))
        new_labels[labels == sc] = large_clusters[nearest_idx]
        merged += 1

    return new_labels, merged


def compute_cluster_summaries(
    labels: "np.ndarray",
    vectors: "np.ndarray",
    feature_names: list[str],
) -> list[ClusterSummary]:
    """Compute per-cluster centroid, size, and top distinguishing features."""
    import numpy as np

    unique_labels = sorted(set(labels[labels >= 0]))
    global_mean = vectors[labels >= 0].mean(axis=0) if np.any(labels >= 0) else np.zeros(vectors.shape[1])

    summaries = []
    for lbl in unique_labels:
        mask = labels == lbl
        cluster_vectors = vectors[mask]
        centroid = cluster_vectors.mean(axis=0)
        n_members = int(np.sum(mask))

        # Top features: those where cluster centroid deviates most from global mean.
        deviations = np.abs(centroid - global_mean)
        top_indices = np.argsort(deviations)[::-1][:5]
        top_features = [
            (feature_names[i] if i < len(feature_names) else f"dim_{i}", float(deviations[i]))
            for i in top_indices
        ]

        summaries.append(ClusterSummary(
            label=int(lbl),
            n_members=n_members,
            centroid=[float(v) for v in centroid],
            top_features=top_features,
        ))

    return summaries


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def persist_clustering_results(
    db: "AsyncSession",
    summaries: list[ClusterSummary],
    run_id: str,
) -> int:
    """Write cluster metadata to ml_cohorts. Deactivate previous run.

    Returns the number of rows written.
    """
    from app.core.time import utcnow_naive
    from app.models.ml_cohorts import MLCohort

    # Deactivate previous clusters.
    await db.execute(
        update(MLCohort).where(MLCohort.is_active.is_(True)).values(is_active=False)
    )

    now = utcnow_naive()
    for s in summaries:
        db.add(
            MLCohort(
                cluster_label=s.label,
                n_members=s.n_members,
                centroid_json=json.dumps(s.centroid),
                is_active=True,
                run_id=run_id,
                created_at=now,
            )
        )

    await db.flush()
    return len(summaries)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def run_clustering_pipeline(
    db: "AsyncSession",
    min_cluster_size: int | None = None,
    min_samples: int = 25,
    k_anonymity: int | None = None,
) -> ClusteringReport:
    """Full clustering pipeline: load vectors, cluster, enforce k-anonymity, persist.

    Does NOT commit. Caller owns the transaction.
    """
    import time

    import numpy as np

    from app.models.ml_cohorts import MLAnonymizedVector
    from ml.config import get_ml_settings

    settings = get_ml_settings()
    if min_cluster_size is None:
        min_cluster_size = settings.cohort_k_anonymity_threshold
    if k_anonymity is None:
        k_anonymity = settings.cohort_k_anonymity_threshold

    run_id = uuid.uuid4().hex
    report = ClusteringReport(run_id=run_id)

    t0 = time.perf_counter()

    # Load all anonymized vectors.
    result = await db.execute(select(MLAnonymizedVector))
    rows = result.scalars().all()

    if not rows:
        logger.info("run_clustering_pipeline: no anonymized vectors, skipping")
        return report

    vectors = np.array([json.loads(r.vector_json) for r in rows], dtype=np.float64)
    [r.pseudonym_id for r in rows]
    feature_names = json.loads(rows[0].feature_names_json)
    report.n_users = len(vectors)

    if len(vectors) < min_cluster_size:
        logger.info(
            "run_clustering_pipeline: only %d vectors (need %d for min cluster), skipping",
            len(vectors),
            min_cluster_size,
        )
        return report

    # Cluster.
    labels, probabilities = run_clustering(vectors, min_cluster_size, min_samples)
    report.n_noise_points = int(np.sum(labels == -1))

    # Enforce k-anonymity.
    labels, merged = enforce_k_anonymity(labels, vectors, k_anonymity)
    report.clusters_merged = merged

    # Compute summaries.
    summaries = compute_cluster_summaries(labels, vectors, feature_names)
    report.n_clusters = len(summaries)

    if summaries:
        report.largest_cluster = max(s.n_members for s in summaries)
        report.smallest_cluster = min(s.n_members for s in summaries)

    # Persist.
    await persist_clustering_results(db, summaries, run_id)

    report.elapsed_seconds = time.perf_counter() - t0
    logger.info(
        "run_clustering_pipeline(%s): users=%d clusters=%d noise=%d "
        "merged=%d largest=%d smallest=%d elapsed=%.2fs",
        run_id,
        report.n_users,
        report.n_clusters,
        report.n_noise_points,
        report.clusters_merged,
        report.largest_cluster,
        report.smallest_cluster,
        report.elapsed_seconds,
    )
    return report
