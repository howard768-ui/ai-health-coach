# Import EVERY model module so SQLAlchemy registers all tables on Base.metadata.
# This is the canonical registration point: alembic/env.py imports this package
# so autogenerate sees every table. A missing module here means autogenerate
# would emit op.drop_table for that table (data loss). Audit P2b. There is a
# test (test_models_metadata_complete.py) that fails if a model file is added
# without being registered here.
from app.models.health import OuraToken, SleepRecord, HealthMetricRecord, ActivityRecord, SourcePriority  # noqa
from app.models.user import User  # noqa
from app.models.chat import Conversation, ChatMessageRecord  # noqa
from app.models.notification import DeviceToken, NotificationRecord, NotificationPreference, NotificationTemplate  # noqa
from app.models.meal import MealRecord, FoodItemRecord  # noqa
from app.models.peloton import PelotonToken, WorkoutRecord  # noqa
from app.models.garmin import GarminToken, GarminDailyRecord  # noqa
from app.models.correlation import UserCorrelation  # noqa
from app.models.waitlist import WaitlistSignup  # noqa
from app.models.mascot import UserMascotState  # noqa
from app.models.refresh_token import RefreshToken  # noqa
# ML tables (Phase 5+). Previously absent from both __init__ and alembic/env,
# so autogenerate would have dropped them (audit P2b / R2).
from app.models import ml_features  # noqa
from app.models import ml_baselines  # noqa
from app.models import ml_insights  # noqa
from app.models import ml_discovery  # noqa
from app.models import ml_experiments  # noqa
from app.models import ml_cohorts  # noqa
from app.models import ml_drift  # noqa
from app.models import ml_models  # noqa
from app.models import ml_synth  # noqa
from app.models import ml_training_runs  # noqa
