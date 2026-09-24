# ============================================================================
# Dapr Service
# ============================================================================
# Wrapper for Dapr HTTP API - enforces infrastructure abstraction.
# All communication with Dapr sidecar goes through this service.
#
# Dapr provides:
# - Pub/Sub via Kafka (abstraction from direct Kafka client)
# - Jobs API for scheduled callbacks
# - State Store abstraction
#
# Configuration:
# - DAPR_HTTP_PORT: Dapr sidecar HTTP port (default: 3500)
# - DAPR_HOST: Dapr sidecar host (default: localhost)
# ============================================================================

import httpx
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import os

# Configure logging
logger = logging.getLogger(__name__)


class DaprService:
    """
    Wrapper for Dapr HTTP API - enforces infrastructure abstraction.

    This service provides methods to interact with Dapr sidecar for:
    - Publishing events to Pub/Sub topics
    - Scheduling jobs via Dapr Jobs API
    - Deleting scheduled jobs

    All infrastructure communication goes through Dapr - no direct
    Kafka, Redis, or other infrastructure client dependencies.
    """

    # Dapr sidecar configuration
    DAPR_HTTP_PORT = int(os.getenv("DAPR_HTTP_PORT", "3500"))
    DAPR_HOST = os.getenv("DAPR_HOST", "localhost")
    DAPR_BASE_URL = f"http://{DAPR_HOST}:{DAPR_HTTP_PORT}/v1.0"
    DAPR_JOBS_URL = f"http://{DAPR_HOST}:{DAPR_HTTP_PORT}/v1.0-alpha1/jobs"

    # Default timeout for Dapr API calls (seconds)
    DEFAULT_TIMEOUT = 5.0

    # Dapr component names (configured via Dapr configuration)
    PUBSUB_NAME = os.getenv("DAPR_PUBSUB_NAME", "pubsub.kafka")
    STATE_STORE_NAME = os.getenv("DAPR_STATE_STORE_NAME", "state.postgresql")

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if Dapr sidecar is available.

        Returns:
            bool: True if Dapr sidecar is responding, False otherwise
        """
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{cls.DAPR_BASE_URL}/healthz")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Dapr sidecar not available: {e}")
            return False

    async def publish_event(
        self,
        topic: str,
        data: Dict[str, Any],
        pubsub_name: Optional[str] = None
    ) -> bool:
        """
        Publish event via Dapr Pub/Sub.

        Args:
            topic: The topic to publish to (e.g., "task-events", "reminders")
            data: Event payload as dictionary
            pubsub_name: Dapr pubsub component name (uses default if None)

        Returns:
            bool: True if published successfully, False otherwise

        Example:
            >>> await dapr_service.publish_event(
            ...     topic="task-events",
            ...     data={
            ...         "event_type": "task.completed",
            ...         "event_id": str(uuid.uuid4()),
            ...         "task_id": 123,
            ...         "user_id": 1
            ...     }
            ... )
        """
        pubsub_name = pubsub_name or self.PUBSUB_NAME
        url = f"{self.DAPR_BASE_URL}/publish/{pubsub_name}/{topic}"

        # Ensure event has required fields
        if "event_id" not in data:
            data["event_id"] = str(uuid.uuid4())
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                response = await client.post(url, json=data)
                response.raise_for_status()

                logger.info(
                    f"Published event to {pubsub_name}/{topic}: "
                    f"{data.get('event_type', 'unknown')} (id: {data.get('event_id')})"
                )
                return True

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error publishing event: {e.response.status_code} - {e.response.text}")
            return False
        except httpx.ConnectError:
            logger.warning(f"Dapr sidecar not available at {self.DAPR_BASE_URL}")
            return False
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False

    async def schedule_job(
        self,
        job_name: str,
        schedule_time: datetime,
        payload: Dict[str, Any],
        callback_uri: Optional[str] = None
    ) -> bool:
        """
        Schedule a job via Dapr Jobs API.

        Args:
            job_name: Unique name for the job (e.g., "reminder-123")
            schedule_time: When to trigger the job (UTC datetime)
            payload: Data to send to callback endpoint
            callback_uri: URI for Dapr to call (uses default if None)

        Returns:
            bool: True if scheduled successfully, False otherwise

        Example:
            >>> await dapr_service.schedule_job(
            ...     job_name="reminder-123",
            ...     schedule_time=datetime(2026, 2, 17, 8, 45),
            ...     payload={"task_id": 123, "user_id": 1}
            ... )
        """
        url = self.DAPR_JOBS_URL

        # Default callback URI (should point to our backend)
        if callback_uri is None:
            backend_port = os.getenv("PORT", "8000")
            callback_uri = f"http://localhost:{backend_port}/api/v1/jobs/trigger"

        job_data = {
            "name": job_name,
            "schedule": schedule_time.isoformat(),
            "data": payload,
            "http": {
                "uri": callback_uri
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                response = await client.post(url, json=job_data)
                response.raise_for_status()

                logger.info(f"Scheduled Dapr job: {job_name} at {schedule_time.isoformat()}")
                return True

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error scheduling job: {e.response.status_code} - {e.response.text}")
            return False
        except httpx.ConnectError:
            logger.warning(f"Dapr sidecar not available at {self.DAPR_BASE_URL}")
            return False
        except Exception as e:
            logger.error(f"Failed to schedule job: {e}")
            return False

    async def delete_job(self, job_name: str) -> bool:
        """
        Delete a scheduled job via Dapr Jobs API.

        Args:
            job_name: Name of the job to delete (e.g., "reminder-123")

        Returns:
            bool: True if deleted or not found, False on error

        Note:
            Returns True if job doesn't exist (404) - idempotent operation
        """
        url = f"{self.DAPR_JOBS_URL}/{job_name}"

        try:
            async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                response = await client.delete(url)

                # 404 is acceptable - job may not exist
                if response.status_code == 404:
                    logger.debug(f"Job {job_name} not found (may have already been deleted)")
                    return True

                response.raise_for_status()
                logger.info(f"Deleted Dapr job: {job_name}")
                return True

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error deleting job: {e.response.status_code} - {e.response.text}")
            return False
        except httpx.ConnectError:
            logger.warning(f"Dapr sidecar not available at {self.DAPR_BASE_URL}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete job: {e}")
            return False

    async def schedule_reminder(
        self,
        task_id: int,
        user_id: int,
        task_title: str,
        reminder_time: datetime,
        due_date: Optional[datetime] = None
    ) -> bool:
        """
        Schedule a reminder job via Dapr Jobs API.

        Convenience method for scheduling task reminders.

        Args:
            task_id: The ID of the task
            user_id: The ID of the user
            task_title: Title of the task (for notification)
            reminder_time: When to trigger the reminder
            due_date: Optional due date of the task

        Returns:
            bool: True if scheduled successfully, False otherwise
        """
        job_name = f"reminder-{task_id}"

        payload = {
            "task_id": task_id,
            "user_id": user_id,
            "task_title": task_title,
            "reminder_time": reminder_time.isoformat(),
            "due_date": due_date.isoformat() if due_date else None
        }

        return await self.schedule_job(
            job_name=job_name,
            schedule_time=reminder_time,
            payload=payload
        )

    async def cancel_reminder(self, task_id: int) -> bool:
        """
        Cancel a reminder job via Dapr Jobs API.

        Convenience method for canceling task reminders.

        Args:
            task_id: The ID of the task

        Returns:
            bool: True if canceled or not found, False on error
        """
        job_name = f"reminder-{task_id}"
        return await self.delete_job(job_name)

    async def get_state(
        self,
        key: str,
        state_store_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get state from Dapr State Store.

        Args:
            key: State key to retrieve
            state_store_name: Dapr state store component name (uses default if None)

        Returns:
            State value or None if not found
        """
        state_store_name = state_store_name or self.STATE_STORE_NAME
        url = f"{self.DAPR_BASE_URL}/state/{state_store_name}/{key}"

        try:
            async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                response = await client.get(url)

                if response.status_code == 404:
                    return None

                response.raise_for_status()
                data = response.json()
                # Dapr returns {"key": ..., "data": ..., "etag": ...}
                return data.get("data")

        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return None

    async def save_state(
        self,
        key: str,
        value: Any,
        state_store_name: Optional[str] = None
    ) -> bool:
        """
        Save state to Dapr State Store.

        Args:
            key: State key to save
            value: Value to save (will be JSON serialized)
            state_store_name: Dapr state store component name (uses default if None)

        Returns:
            bool: True if saved successfully, False otherwise
        """
        state_store_name = state_store_name or self.STATE_STORE_NAME
        url = f"{self.DAPR_BASE_URL}/state/{state_store_name}"

        state_data = [{
            "key": key,
            "value": value
        }]

        try:
            async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                response = await client.post(url, json=state_data)
                response.raise_for_status()

                logger.debug(f"Saved state: {key}")
                return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    async def delete_state(
        self,
        key: str,
        state_store_name: Optional[str] = None
    ) -> bool:
        """
        Delete state from Dapr State Store.

        Args:
            key: State key to delete
            state_store_name: Dapr state store component name (uses default if None)

        Returns:
            bool: True if deleted or not found, False on error
        """
        state_store_name = state_store_name or self.STATE_STORE_NAME
        url = f"{self.DAPR_BASE_URL}/state/{state_store_name}/{key}"

        try:
            async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                response = await client.delete(url)

                if response.status_code == 404:
                    return True

                response.raise_for_status()
                logger.debug(f"Deleted state: {key}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete state: {e}")
            return False


# Singleton instance for use across the application
dapr_service = DaprService()
