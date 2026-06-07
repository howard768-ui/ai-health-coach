import logging

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger("meld.database")


@event.listens_for(Engine, "connect")
def _enforce_sqlite_foreign_keys(dbapi_connection, connection_record):
    """Turn on foreign-key enforcement for every SQLite connection.

    SQLite ignores ``ON DELETE CASCADE`` unless ``PRAGMA foreign_keys=ON`` is
    set per connection. Without this, account-deletion cascades silently did
    nothing in dev and in the test suite, so the deletion tests passed while
    deleting no child rows (audit A3). Postgres enforces FKs natively, so this
    is scoped to SQLite by inspecting the DBAPI module. Registered on the
    generic Engine so it also applies to the per-test in-memory engines.
    """
    module = type(dbapi_connection).__module__.lower()
    if "sqlite" in module:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# pool_pre_ping: cheap SELECT 1 before each checkout, evicts dead connections
# left over from a Postgres restart or network blip. Without it, the first
# request after a Postgres reprovision (see incident 2026-04-29) would have
# returned a stale-conn error before reconnecting on retry.
# pool_recycle: force-close connections older than 30 minutes regardless of
# health. Belt-and-suspenders against connection-age bugs.
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_env == "development",
    pool_pre_ping=True,
    pool_recycle=1800,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    # CRITICAL: invoked via FastAPI Depends; do not rely on call-graph.
    # Every database-touching route declares `db: AsyncSession = Depends(get_db)`,
    # which static analyzers like GitNexus do not model as a CALLS edge.
    # Treat changes to this function as Tier 3 by blast radius even though
    # the file path is not on the Tier 3 list.
    async with async_session() as session:
        yield session
