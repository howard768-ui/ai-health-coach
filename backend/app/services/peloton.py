"""Peloton API client using pylotoncycle library.

Uses OAuth2 authentication (not the old session cookie approach).
pylotoncycle handles the auth flow using Peloton's known client_id.
"""

import logging
import asyncio

logger = logging.getLogger("meld.peloton")

# pylotoncycle wraps requests — most errors surface as requests.exceptions
# or plain ValueError/KeyError on bad response shapes. We include OSError
# to catch underlying socket failures.
try:
    import requests  # type: ignore
    _PELOTON_FETCH_ERRORS: tuple[type[BaseException], ...] = (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
        OSError,
        ValueError,
        KeyError,
    )
except ImportError:
    _PELOTON_FETCH_ERRORS = (ConnectionError, TimeoutError, OSError, ValueError, KeyError)


class PelotonClient:
    """Interacts with Peloton via pylotoncycle library."""

    def __init__(self):
        self._client = None
        self.peloton_user_id = None

    async def login(self, username: str, password: str) -> dict:
        """Authenticate with Peloton via OAuth2."""
        from pylotoncycle import PylotonCycle

        def _login():
            client = PylotonCycle(username=username, password=password)
            return client

        try:
            self._client = await asyncio.to_thread(_login)
            # Get user info
            user_id = getattr(self._client, 'user_id', None)
            self.peloton_user_id = user_id
            return {
                "session_id": "oauth",  # pylotoncycle manages tokens internally
                "user_id": user_id or "connected",
            }
        except _PELOTON_FETCH_ERRORS as e:
            logger.error("Peloton login failed: %s", e)
            raise

    async def get_workouts(self, limit: int = 20) -> list:
        """Get recent workouts."""
        if not self._client:
            raise ValueError("Not authenticated. Call login() first.")

        def _get():
            return self._client.GetRecentWorkouts(limit)

        try:
            return await asyncio.to_thread(_get)
        except _PELOTON_FETCH_ERRORS as e:
            logger.error("Peloton get_workouts failed: %s", e)
            return []

    async def get_workout_metrics(self, workout_id: str) -> dict:
        """Get detailed metrics for a workout."""
        if not self._client:
            return {}

        def _get():
            return self._client.GetWorkoutMetricsById(workout_id)

        try:
            return await asyncio.to_thread(_get)
        except _PELOTON_FETCH_ERRORS as e:
            logger.error("Peloton get_metrics failed: %s", e)
            return {}

    def parse_workout(self, workout: dict) -> dict:
        """Normalize a Peloton workout to our WorkoutRecord format."""
        fitness_discipline = workout.get("fitness_discipline", "cycling")
        workout_type_map = {
            "cycling": "cycling",
            "running": "running",
            "strength": "strength",
            "yoga": "yoga",
            "meditation": "meditation",
            "stretching": "stretching",
            "walking": "walking",
            "bootcamp": "bootcamp",
            "rowing": "rowing",
        }
        workout_type = workout_type_map.get(fitness_discipline, fitness_discipline)

        ride = workout.get("ride", {}) or {}

        # total_work arrives in joules; /1000 approximates kcal (Peloton
        # convention). Round to nearest instead of floor-dividing: the old
        # `// 1000 or None` turned any low-output workout (<1000 J, e.g.
        # short stretches) into None, silently dropping the calories metric
        # for a workout that DID report work. None now strictly means
        # "Peloton sent no usable total_work". Audit P4/G.
        total_work = workout.get("total_work")
        if isinstance(total_work, (int, float)) and total_work > 0:
            calories = round(total_work / 1000)
        else:
            calories = None

        return {
            "peloton_workout_id": workout.get("id"),
            "workout_type": workout_type,
            "duration_seconds": ride.get("duration", workout.get("ride_duration", 0)) or 0,
            "calories": calories,
            "avg_heart_rate": workout.get("avg_heart_rate"),
            "max_heart_rate": workout.get("max_heart_rate"),
            "avg_output": workout.get("avg_output"),
            "instructor": ride.get("instructor", {}).get("name") if isinstance(ride.get("instructor"), dict) else None,
            "title": ride.get("title"),
            "created_at": workout.get("created_at", 0),
        }
