"""Audit P2b: every model module must be registered on Base.metadata so
alembic autogenerate sees every table. A model file that is not imported by
app/models/__init__.py is absent from target_metadata, and autogenerate would
emit op.drop_table for it -> data loss on the next generated migration.

Run: cd backend && uv run pytest tests/test_models_metadata_complete.py -v
"""

import os
import pathlib
import sys

os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-long-enough-aaaaaaaaaaaa")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)

import app.models  # noqa: F401 -- canonical registration of every model
from app.database import Base

_MODELS_DIR = pathlib.Path(app.models.__file__).parent


def test_every_model_module_is_imported_by_package():
    """Importing app.models must import every model module, or its tables are
    missing from target_metadata."""
    model_files = {
        p.stem for p in _MODELS_DIR.glob("*.py") if p.stem != "__init__"
    }
    for mod in sorted(model_files):
        assert f"app.models.{mod}" in sys.modules, (
            f"app/models/{mod}.py is not imported by app/models/__init__.py; "
            f"its tables are absent from alembic target_metadata (autogenerate "
            f"would drop them). Add it to __init__.py."
        )


def test_ml_and_phi_tables_present_in_metadata():
    """The ML / mascot tables that were previously absent must be in metadata."""
    expected = {
        "ml_feature_values",
        "ml_baselines",
        "ml_change_points",
        "ml_forecasts",
        "ml_anomalies",
        "ml_insight_candidates",
        "ml_rankings",
        "ml_directional_tests",
        "ml_causal_estimates",
        "ml_experiments",
        "ml_n_of_1_results",
        "ml_cohort_consent",
        "ml_anonymized_vectors",
        "ml_cohorts",
        "ml_drift_results",
        "ml_models",
        "ml_training_runs",
        "user_mascot_state",
    }
    missing = expected - set(Base.metadata.tables)
    assert not missing, (
        f"tables absent from target_metadata (autogenerate would drop them): "
        f"{sorted(missing)}"
    )
