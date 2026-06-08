"""Alembic migration environment for Meld backend.

Wires our app models + async SQLAlchemy setup into the Alembic sync runner.
"""
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Import all models so Base.metadata knows about every table.
# This is required for --autogenerate to detect schema changes.
from app.database import Base  # noqa: F401
from app.models import health  # noqa: F401
from app.models import user  # noqa: F401
from app.models import chat  # noqa: F401
from app.models import notification  # noqa: F401
from app.models import meal  # noqa: F401
from app.models import peloton  # noqa: F401
from app.models import garmin  # noqa: F401
from app.models import correlation  # noqa: F401
from app.models import refresh_token  # noqa: F401
from app.models import ml_features  # noqa: F401
from app.models import ml_baselines  # noqa: F401
from app.models import ml_insights  # noqa: F401
from app.models import ml_synth  # noqa: F401
# Canonical complete registration: importing the package pulls EVERY model
# module (see app/models/__init__.py), so target_metadata covers every table
# and autogenerate never emits a spurious drop_table. Audit P2b.
from app import models  # noqa: F401

# Load app settings to get the real DATABASE_URL
from app.config import settings

config = context.config

# Override sqlalchemy.url with the app's database URL.
# Alembic runs synchronously, so strip the async driver prefix if present.
db_url = settings.database_url
if db_url.startswith("sqlite+aiosqlite://"):
    db_url = db_url.replace("sqlite+aiosqlite://", "sqlite://", 1)
elif db_url.startswith("postgresql+asyncpg://"):
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL scripts)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (applies directly to DB)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Compare column types so added/changed columns are detected
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
