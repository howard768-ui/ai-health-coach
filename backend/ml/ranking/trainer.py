"""XGBoost LambdaMART ranker trainer for Phase 7.

Trains a learning-to-rank model on labeled insight feedback (thumbs_up,
thumbs_down, dismissed, already_knew) from ``ml_rankings`` joined with
candidate features from ``ml_insight_candidates``.

Cold-start bootstrap: when real labeled pairs are below the configurable
threshold (default 20), generates synthetic training data from the Phase 4.5
synth factory and assigns heuristic-derived labels.

All heavy imports (xgboost, numpy, sklearn) are lazy inside function bodies
per the cold-boot contract.

Entry point is ``train_ranker_pipeline``, called from ``ml.api``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

MODEL_TYPE = "ranker"

# Feature names in model input order. Must match exactly when building
# the feature matrix for prediction.
FEATURE_NAMES: list[str] = [
    "effect_size",
    "confidence",
    "novelty",
    "recency_days",
    "actionability_score",
    "literature_support",
    "directional_support",
    "causal_support",
]

# Label mapping from feedback strings to integer relevance grades.
# XGBoost rank:pairwise requires non-negative integers.
LABEL_MAP: dict[str, int] = {
    "thumbs_up": 3,
    "already_knew": 2,
    "dismissed": 1,
    "thumbs_down": 0,
}

# Default XGBoost hyperparameters for LambdaMART.
DEFAULT_PARAMS: dict = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@5",
    "max_depth": 6,
    "eta": 0.1,
    "n_estimators": 100,
    "lambdarank_num_pair_per_sample": 8,
    "lambdarank_pair_method": "mean",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingData:
    """Prepared features, labels, and group info for XGBoost LTR."""

    X: "object"  # numpy ndarray, shape (n_samples, n_features)
    y: "object"  # numpy ndarray, shape (n_samples,)
    groups: "object"  # numpy ndarray of group sizes
    user_ids: list[str]  # user_id per sample (for GroupKFold)
    n_samples: int = 0
    n_users: int = 0
    source: str = "real"  # "real", "synth", "mixed"


@dataclass
class TrainedModel:
    """Output of a training run."""

    model: object  # xgb.Booster
    model_version: str
    feature_names: list[str]
    hyperparams: dict
    train_samples: int
    val_ndcg: float
    feature_importances: dict[str, float]


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------


async def prepare_real_training_data(db: "AsyncSession") -> TrainingData:
    """Load labeled pairs from ml_rankings + ml_insight_candidates.

    Returns a TrainingData with features, labels, and group info.
    """
    import numpy as np
    from sqlalchemy import select

    from app.models.ml_insights import MLInsightCandidate, MLRanking

    # Join rankings (with feedback) to candidates (with features).
    stmt = (
        select(
            MLRanking.user_id,
            MLRanking.feedback,
            MLInsightCandidate.effect_size,
            MLInsightCandidate.confidence,
            MLInsightCandidate.novelty,
            MLInsightCandidate.recency_days,
            MLInsightCandidate.actionability_score,
            MLInsightCandidate.literature_support,
            MLInsightCandidate.directional_support,
            MLInsightCandidate.causal_support,
        )
        .join(
            MLInsightCandidate,
            MLRanking.candidate_id == MLInsightCandidate.id,
        )
        .where(MLRanking.feedback.isnot(None))
        .order_by(MLRanking.user_id, MLRanking.surface_date)
    )
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        return TrainingData(
            X=np.empty((0, len(FEATURE_NAMES))),
            y=np.empty(0),
            groups=np.empty(0, dtype=np.int32),
            user_ids=[],
            n_samples=0,
            n_users=0,
            source="real",
        )

    # Build feature matrix.
    X_list = []
    y_list = []
    uid_list = []
    for row in rows:
        features = [
            float(row.effect_size),
            float(row.confidence),
            float(row.novelty),
            float(row.recency_days) / 30.0,  # normalize to ~[0, 1]
            float(row.actionability_score),
            1.0 if row.literature_support else 0.0,
            1.0 if row.directional_support else 0.0,
            1.0 if row.causal_support else 0.0,
        ]
        X_list.append(features)
        y_list.append(LABEL_MAP.get(row.feedback, 0.0))
        uid_list.append(row.user_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Compute group sizes (consecutive runs of same user_id).
    unique_users = []
    group_sizes = []
    current_user = None
    current_count = 0
    for uid in uid_list:
        if uid != current_user:
            if current_user is not None:
                group_sizes.append(current_count)
                unique_users.append(current_user)
            current_user = uid
            current_count = 1
        else:
            current_count += 1
    if current_user is not None:
        group_sizes.append(current_count)
        unique_users.append(current_user)

    return TrainingData(
        X=X,
        y=y,
        groups=np.array(group_sizes, dtype=np.int32),
        user_ids=uid_list,
        n_samples=len(X_list),
        n_users=len(unique_users),
        source="real",
    )


async def generate_synth_training_data(
    db: "AsyncSession",
    n_users: int = 50,
    seed: int = 42,
) -> TrainingData:
    """Bootstrap training data from the synth factory.

    Generates synthetic users, runs the full discovery + candidate pipeline,
    then assigns heuristic labels based on ground-truth properties:
    - literature-matched + high effect size -> 1.0 (thumbs_up)
    - developing + moderate effect -> 0.5
    - emerging / low effect -> 0.1 (negative)
    """
    import numpy as np

    from ml.features.store import materialize_for_user
    from ml.ranking.candidates import generate_candidates
    from ml import api as ml_api
    from datetime import date, timedelta

    rng = np.random.default_rng(seed)

    manifest = await ml_api.generate_synth_cohort(
        db, n_users=n_users, days=90, seed=seed
    )
    await db.flush()

    today = date.today()
    start = today - timedelta(days=90)

    all_X = []
    all_y = []
    all_uids = []
    group_sizes = []

    for user_id in manifest.user_ids:
        # Materialize features.
        await materialize_for_user(db, user_id, start, today)
        await db.flush()

        # Run L2 associations.
        await ml_api.run_associations(db, user_id, window_days=60)
        await db.flush()

        # Generate candidates through the same reference date the features
        # were materialized to. `through_date` is a required parameter;
        # omitting it made every cold-start training run crash with a
        # TypeError before any model was produced. Audit P3/D2 (round-1 #31).
        candidates = await generate_candidates(db, user_id, through_date=today)
        if not candidates:
            continue

        user_features = []
        user_labels = []
        for c in candidates:
            features = [
                c.effect_size,
                c.confidence,
                c.novelty,
                float(c.recency_days) / 30.0,
                c.actionability_score,
                1.0 if c.literature_support else 0.0,
                1.0 if c.directional_support else 0.0,
                1.0 if c.causal_support else 0.0,
            ]
            user_features.append(features)

            # Heuristic integer label based on ground truth.
            if c.literature_support and c.effect_size > 0.5:
                label = 3
            elif c.confidence > 0.7:
                label = 2
            elif c.effect_size > 0.3:
                label = 1
            else:
                label = 0
            user_labels.append(label)

        if len(user_features) >= 2:
            all_X.extend(user_features)
            all_y.extend(user_labels)
            all_uids.extend([user_id] * len(user_features))
            group_sizes.append(len(user_features))

    if not all_X:
        return TrainingData(
            X=np.empty((0, len(FEATURE_NAMES))),
            y=np.empty(0),
            groups=np.empty(0, dtype=np.int32),
            user_ids=[],
            n_samples=0,
            n_users=0,
            source="synth",
        )

    return TrainingData(
        X=np.array(all_X, dtype=np.float32),
        y=np.array(all_y, dtype=np.float32),
        groups=np.array(group_sizes, dtype=np.int32),
        user_ids=all_uids,
        n_samples=len(all_X),
        n_users=len(group_sizes),
        source="synth",
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_ranker(
    X: "object",
    y: "object",
    groups: "object",
    user_ids: list[str],
    params: dict | None = None,
) -> TrainedModel:
    """Train XGBoost LambdaMART with GroupKFold validation.

    Returns a TrainedModel with the model, NDCG@5, and feature importances.
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold

    if params is None:
        params = dict(DEFAULT_PARAMS)

    n_estimators = params.pop("n_estimators", 100)

    # GroupKFold by user_id to prevent leakage.
    unique_users = list(dict.fromkeys(user_ids))
    user_to_group = {u: i for i, u in enumerate(unique_users)}
    sample_groups = np.array([user_to_group[u] for u in user_ids])

    # Use 80/20 split if enough users, otherwise train on all.
    if len(unique_users) >= 5:
        gkf = GroupKFold(n_splits=min(5, len(unique_users)))
        splits = list(gkf.split(X, y, groups=sample_groups))
        train_idx, val_idx = splits[0]  # use first fold

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Recompute group sizes for train/val subsets.
        train_groups = _compute_group_sizes(np.array(user_ids)[train_idx])
        val_groups = _compute_group_sizes(np.array(user_ids)[val_idx])

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(train_groups)
        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(val_groups)

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        val_ndcg = float(booster.best_score)
    else:
        # Too few users for proper validation; train on all.
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(groups)
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            verbose_eval=False,
        )
        val_ndcg = 0.0

    # Feature importances.
    importance = booster.get_score(importance_type="gain")
    feature_importances = {}
    for i, name in enumerate(FEATURE_NAMES):
        key = f"f{i}"
        feature_importances[name] = importance.get(key, 0.0)

    # Version string.
    import hashlib
    import time

    version_hash = hashlib.sha256(
        f"{time.time()}-{len(user_ids)}".encode()
    ).hexdigest()[:8]
    model_version = f"ranker-{version_hash}"

    return TrainedModel(
        model=booster,
        model_version=model_version,
        feature_names=FEATURE_NAMES,
        hyperparams=params,
        train_samples=len(y),
        val_ndcg=val_ndcg,
        feature_importances=feature_importances,
    )


def predict_scores(model: object, X: "object") -> "object":
    """Score candidates using a trained XGBoost model.

    Returns numpy array of scores (higher = more relevant).
    """
    import xgboost as xgb

    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    return model.predict(dmatrix)


def _compute_group_sizes(user_ids_arr: "object") -> list[int]:
    """Compute consecutive group sizes from an ordered user_id array."""
    groups = []
    current = None
    count = 0
    for uid in user_ids_arr:
        if uid != current:
            if current is not None:
                groups.append(count)
            current = uid
            count = 1
        else:
            count += 1
    if current is not None:
        groups.append(count)
    return groups


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def train_ranker_pipeline(
    db: "AsyncSession",
    coldstart_threshold: int = 20,
) -> TrainedModel | None:
    """Full training pipeline: load data, optionally bootstrap with synth,
    train, and return the model.

    Returns None if there's not enough data even with synth bootstrap.
    """
    import numpy as np

    real_data = await prepare_real_training_data(db)
    logger.info(
        "Real training data: %d samples from %d users",
        real_data.n_samples,
        real_data.n_users,
    )

    if real_data.n_samples >= coldstart_threshold:
        # Enough real data; train on it directly.
        trained = train_ranker(
            real_data.X, real_data.y, real_data.groups, real_data.user_ids
        )
        logger.info(
            "Trained on real data: NDCG@5=%.4f, samples=%d",
            trained.val_ndcg,
            trained.train_samples,
        )
        return trained

    # Cold start: bootstrap with synth data.
    logger.info(
        "Real data below threshold (%d < %d), bootstrapping with synth",
        real_data.n_samples,
        coldstart_threshold,
    )
    synth_data = await generate_synth_training_data(db, n_users=30, seed=42)
    if synth_data.n_samples == 0:
        logger.warning("Synth training data generation produced 0 samples")
        return None

    # Merge real + synth (real weighted 5x if present). rank:pairwise has no
    # per-sample weights, so weighting is approximated by replicating the
    # real block 5x. Labels, per-row user_ids, and per-group sizes are
    # replicated in LOCKSTEP with X: the previous version replicated only X
    # (5r+s rows against r+s labels), so any mixed-source training crashed
    # on shape mismatch, and the discarded y-extension meant the intended
    # weighting never existed. Replicated rows keep their original user_id,
    # so GroupKFold still puts all copies of a user in the same fold (no
    # leakage). Audit P3/D2 (round-1 #32).
    if real_data.n_samples > 0:
        replication = 5
        X = np.vstack([real_data.X] * replication + [synth_data.X])
        y = np.concatenate([real_data.y] * replication + [synth_data.y])
        user_ids = real_data.user_ids * replication + synth_data.user_ids
        groups = np.concatenate([real_data.groups] * replication + [synth_data.groups])
        source = "mixed"
    else:
        X = synth_data.X
        y = synth_data.y
        user_ids = synth_data.user_ids
        groups = synth_data.groups
        source = "synth"

    # Invariants train_ranker depends on; a violation here means the merge
    # above regressed.
    assert len(X) == len(y) == len(user_ids) == int(np.sum(groups)), (
        f"training data misaligned: X={len(X)} y={len(y)} "
        f"user_ids={len(user_ids)} sum(groups)={int(np.sum(groups))}"
    )

    trained = train_ranker(X, y, groups, user_ids)
    logger.info(
        "Trained on %s data: NDCG@5=%.4f, samples=%d",
        source,
        trained.val_ndcg,
        trained.train_samples,
    )
    return trained
