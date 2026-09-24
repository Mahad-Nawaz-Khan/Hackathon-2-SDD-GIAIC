# ============================================================================
# Jobs Router
# ============================================================================
# Endpoints for Dapr Jobs API integration:
# - /api/jobs/trigger: Callback endpoint for scheduled reminder jobs
#
# These endpoints are called by Dapr when a scheduled job triggers.
# ============================================================================

from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("/trigger")
async def handle_job_trigger(request: Request) -> Dict[str, Any]:
    """
    Dapr Jobs callback endpoint for reminder triggers.

    Called by Dapr when a scheduled reminder job reaches its trigger time.
    This endpoint publishes a reminder event to the reminders topic for
    the notification service to process.

    Request Body (from Dapr Job data):
        {
            "task_id": integer,
            "user_id": integer,
            "task_title": string,
            "reminder_time": "ISO-8601 datetime",
            "due_date": "ISO-8601 datetime" | null
        }

    Event Flow:
        1. User creates task with reminder_time
        2. Backend schedules Dapr job via Dapr Jobs API
        3. At scheduled time, Dapr calls this endpoint
        4. This endpoint publishes reminder.triggered event
        5. Notification service (via Dapr subscriber) processes event

    Returns:
        {"status": "processed" | "error", "message": string | None}
    """
    try:
        payload = await request.json()
        logger.info("Job triggered", extra={"payload": payload})

        task_id = payload.get("task_id")
        user_id = payload.get("user_id")
        task_title = payload.get("task_title", "Unknown")
        reminder_time = payload.get("reminder_time")
        due_date = payload.get("due_date")

        # Import Dapr service to publish event
        from ..services.dapr_service import dapr_service

        # Publish reminder event to Dapr Pub/Sub
        event_published = await dapr_service.publish_event(
            topic="reminders",
            data={
                "event_type": "reminder.triggered",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "user_id": user_id,
                "task_title": task_title,
                "reminder_time": reminder_time,
                "due_date": due_date
            }
        )

        if event_published:
            return {"status": "processed"}
        else:
            return {
                "status": "error",
                "message": "Failed to publish reminder event"
            }

    except Exception as e:
        logger.error(f"Error processing job trigger: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/health")
async def jobs_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for Jobs integration.

    Returns:
        Jobs service health status
    """
    return {
        "status": "healthy",
        "service": "jobs",
        "description": "Dapr Jobs callback handler"
    }
