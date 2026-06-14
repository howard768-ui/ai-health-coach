"""Feature store: orchestration + read / write I/O for ml_feature_values.

**Write path** — ``materialize_for_user`` runs every builder for a given date
range, merges the results into a single wide frame, feeds the frame to the
derived-feature builder, emits data-quality masks, and bulk-writes everything
to ``ml_feature_values``. It also syncs the declarative catalog from
``ml.features.catalog`` to the ``ml_feature_catalog`` table so downstream
services can list features without importing ML internals.

**Read path** — ``get_feature_frame`` returns a wide pandas DataFrame indexed
by date, one column per feature key, suitable for downstream discovery /
forecasting / ranking code. This is the only API the rest of ``backend.ml``
should use to read features. No ad-hoc SQL per module.

Heavy imports (pandas, numpy) are lazy. The cold-boot test guards this.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import and_, delete, select

from ml.features import builders, catalog
from ml.features.builders import MaterializedValue

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class MaterializeResult:
    """Summary of a single materialization run for telemetry + tests."""

    user_id: str
    start: date
    end: date
    rows_written: int
    features_touched: int
    builder_timings_ms: dict[str, float]


# ─────────────────────────────────────────────────────────────────────────
# Catalog sync
# ─────────────────────────────────────────────────────────────────────────


async def sync_catalog_to_db(db: "AsyncSession") -> int:
    """Upsert every FeatureSpec from catalog.py into ml_feature_catalog.

    Idempotent. Returns the number of rows touched (insert or update).
    Called by ``materialize_for_user`` before any write so the DB catalog
    never drifts from the code.
    """
    from app.core.time import utcnow_naive
    from app.models.ml_features import MLFeatureCatalogEntry
    import json

    now = utcnow_naive()
    touched = 0

    # Load existing rows in one query so we can diff.
    existing_result = await db.execute(select(MLFeatureCatalogEntry))
    existing_by_key = {e.feature_key: e for e in existing_result.scalars().all()}

    for spec in catalog.iter_catalog():
        requires_json = json.dumps(list(spec.requires)) if spec.requires else None
        row = existing_by_key.get(spec.key)
        if row is None:
            db.add(
                MLFeatureCatalogEntry(
                    feature_key=spec.key,
                    category=spec.category,
                    domain=spec.domain,
                    description=spec.description,
                    unit=spec.unit,
                    builder_module=spec.builder_module,
                    current_version=spec.version,
                    requires_features=requires_json,
                    created_at=now,
                    updated_at=now,
                )
            )
            touched += 1
        else:
            # Update in place only if anything material changed. Avoids
            # bumping updated_at on every no-op run.
            changed = (
                row.category != spec.category
                or row.domain != spec.domain
                or row.description != spec.description
                or row.unit != spec.unit
                or row.builder_module != spec.builder_module
                or row.current_version != spec.version
                or row.requires_features != requires_json
            )
            if changed:
                row.category = spec.category
                row.domain = spec.domain
                row.description = spec.description
                row.unit = spec.unit
                row.builder_module = spec.builder_module
                row.current_version = spec.version
                row.requires_features = requires_json
                row.updated_at = now
                touched += 1

    await db.flush()
    return touched


# ─────────────────────────────────────────────────────────────────────────
# Write path
# ─────────────────────────────────────────────────────────────────────────


async def materialize_for_user(
    db: "AsyncSession",
    user_id: str,
    start: date,
    end: date,
) -> MaterializeResult:
    """Run every builder for the given user and date window, write to DB.

    Write strategy: for each (user, feature_key, feature_version) in the
    catalog, delete existing rows whose ``feature_date`` falls in ``[start,
    end]`` and insert the fresh batch. This is idempotent, simple, and
    backend-agnostic (no ON CONFLICT DO UPDATE reliance). Wrapped in a
    single transaction so concurrent readers never see partial state.
    """
    import time



    timings: dict[str, float] = {}

    # 1. Sync catalog first so the DB catalog matches code.
    t0 = time.perf_counter()
    await sync_catalog_to_db(db)
    timings["catalog_sync"] = (time.perf_counter() - t0) * 1000

    # 2. Run non-derived builders.
    all_values: list[MaterializedValue] = []

    for builder_name, builder_fn in (
        ("biometric_raw", builders.build_biometric_raw),
        ("activity", builders.build_activity),
        ("nutrition", builders.build_nutrition),
        ("contextual", builders.build_contextual),
        ("quality", builders.build_quality),
    ):
        t0 = time.perf_counter()
        values = await builder_fn(db, user_id, start, end)
        timings[builder_name] = (time.perf_counter() - t0) * 1000
        all_values.extend(values)

    # 3. Build a wide pandas frame from the raw values, then run derived.
    #    Derived builders need the raw values in-memory; they don't re-query.
    t0 = time.perf_counter()
    raw_frame = _values_to_frame(all_values, start, end)
    derived_keys = {
        spec.key
        for spec in catalog.iter_catalog()
        if spec.category in {"biometric_derived", "nutrition"} and spec.requires
    }
    derived_values = builders.build_derived(raw_frame, derived_keys)
    timings["derived"] = (time.perf_counter() - t0) * 1000
    all_values.extend(derived_values)

    # 4. Bulk upsert: delete + insert within one transaction.
    t0 = time.perf_counter()
    rows_written = await _bulk_upsert(db, user_id, all_values, start, end)
    timings["upsert"] = (time.perf_counter() - t0) * 1000

    features_touched = len({v.feature_key for v in all_values})

    return MaterializeResult(
        user_id=user_id,
        start=start,
        end=end,
        rows_written=rows_written,
        features_touched=features_touched,
        builder_timings_ms=timings,
    )


def _values_to_frame(
    values: list[MaterializedValue],
    start: date,
    end: date,
) -> "pd.DataFrame":
    """Build a wide DataFrame from raw (not derived) materialized values.

    The derived builder assumes every raw column the derived features depend
    on is present (NaN-filled if unobserved). We build on the full date
    range so trailing windows can see every day.
    """
    import pandas as pd

    # Only raw / nutrition / activity / contextual / quality here — derived is
    # layered on top after this call.
    rows = [
        {
            "feature_date": v.feature_date,
            "feature_key": v.feature_key,
            "value": v.value if v.is_observed else None,
        }
        for v in values
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    wide = df.pivot_table(
        index="feature_date", columns="feature_key", values="value", aggfunc="first"
    )
    # Reindex to the full date range so derived rolling windows line up
    # correctly, even on days without any observed raw rows.
    full_index = [
        (start + timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]
    wide = wide.reindex(full_index)
    return wide


async def _bulk_upsert(
    db: "AsyncSession",
    user_id: str,
    values: list[MaterializedValue],
    start: date,
    end: date,
) -> int:
    """Delete existing rows in the window, insert the fresh batch.

    Wrapped by the caller's outer transaction (we don't commit here). Returns
    the count of rows inserted.
    """
    from app.core.time import utcnow_naive
    from app.models.ml_features import MLFeatureValue

    if not values:
        return 0

    start_s, end_s = start.isoformat(), end.isoformat()
    feature_keys = {v.feature_key for v in values}

    # Delete the window first. Scope to the keys we are about to write so
    # we don't touch features we didn't materialize this run.
    await db.execute(
        delete(MLFeatureValue).where(
            and_(
                MLFeatureValue.user_id == user_id,
                MLFeatureValue.feature_key.in_(feature_keys),
                MLFeatureValue.feature_date >= start_s,
                MLFeatureValue.feature_date <= end_s,
            )
        )
    )

    now = utcnow_naive()
    to_insert = [
        {
            "user_id": user_id,
            "feature_key": v.feature_key,
            "feature_date": v.feature_date,
            "value": v.value,
            "is_observed": v.is_observed,
            "imputed_by": v.imputed_by,
            "source_hash": v.compute_source_hash() or None,
            "feature_version": v.feature_version,
            "computed_at": now,
        }
        for v in values
    ]
    # SQLAlchemy Core bulk insert — way faster than ORM add() x N.
    from sqlalchemy import insert

    await db.execute(insert(MLFeatureValue), to_insert)
    return len(to_insert)


# ─────────────────────────────────────────────────────────────────────────
# Read path
# ─────────────────────────────────────────────────────────────────────────


async def get_feature_frame(
    db: "AsyncSession",
    user_id: str,
    feature_keys: list[str] | None,
    start: date,
    end: date,
    include_imputed: bool = True,
) -> "pd.DataFrame":
    """Return a wide DataFrame indexed by ``feature_date``, one column per feature.

    ``feature_keys=None`` returns every feature in the catalog (not every
    feature with data — missing features appear as fully-NaN columns).
    ``include_imputed=False`` masks imputed cells to NaN so callers that
    need strict observation can opt out without a second query.
    """
    import pandas as pd

    from app.models.ml_features import MLFeatureValue

    start_s, end_s = start.isoformat(), end.isoformat()
    if feature_keys is None:
        target_keys = [s.key for s in catalog.iter_catalog()]
    else:
        target_keys = list(feature_keys)

    result = await db.execute(
        select(
            MLFeatureValue.feature_date,
            MLFeatureValue.feature_key,
            MLFeatureValue.value,
            MLFeatureValue.is_observed,
        ).where(
            MLFeatureValue.user_id == user_id,
            MLFeatureValue.feature_key.in_(target_keys),
            MLFeatureValue.feature_date >= start_s,
            MLFeatureValue.feature_date <= end_s,
        )
    )
    rows = result.all()

    records = []
    for feature_date, feature_key, value, is_observed in rows:
        if not include_imputed and not is_observed:
            # Mask as missing at read time.
            records.append(
                {"feature_date": feature_date, "feature_key": feature_key, "value": None}
            )
        else:
            records.append(
                {"feature_date": feature_date, "feature_key": feature_key, "value": value}
            )

    if not records:
        # Empty frame indexed by the requested date range so downstream code
        # can assume a stable shape.
        idx = [
            (start + timedelta(days=i)).isoformat()
            for i in range((end - start).days + 1)
        ]
        return pd.DataFrame(index=idx, columns=target_keys, dtype=float)

    df = pd.DataFrame(records)
    wide = df.pivot_table(
        index="feature_date", columns="feature_key", values="value", aggfunc="first"
    )

    # Ensure every requested key is a column even if empty, and every date
    # in [start, end] is present as a row.
    full_index = [
        (start + timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]
    wide = wide.reindex(index=full_index, columns=target_keys)
    return wide
