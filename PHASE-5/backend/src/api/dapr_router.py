# ============================================================================
# Dapr Router
# ============================================================================
# Endpoints for Dapr sidecar integration:
# - /dapr/subscribe: Returns subscription configuration for Dapr Pub/Sub
# - /api/events/task-completed: Handles task.completed events
# - /api/events/reminder-triggered: Handles reminder events
#
# These endpoints are called by the Dapr sidecar to manage event-driven
# features like recurring tasks and reminders.
# ============================================================================

from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["dapr"])


@router.get("/dapr/subscribe")
async def dapr_subscribe() -> list[Dict[str, str]]:
    """
    Dapr subscription registration endpoint.

    Called by Dapr sidecar on startup to discover which topics and routes
    this service wants to subscribe to. The returned configuration tells
    Dapr to forward events from specified topics to our endpoints.

    Returns:
        List of subscription configurations with pubsubname, topic, and route

    Example:
        >>> # Dapr calls this on startup and gets:
        >>> [
        >>>     {"pubsubname": "pubsub.kafka", "topic": "task-events",
        >>>      "route": "/api/events/task-completed"},
        >>>     {"pubsubname": "pubsub.kafka", "topic": "reminders",
        >>>      "route": "/api/events/reminder-triggered"}
        >>> ]
    """
    subscriptions = [
        {
            "pubsubname": "pubsub.kafka",
            "topic": "task-events",
            "route": "/api/events/task-completed"
        },
        {
            "pubsubname": "pubsub.kafka",
            "topic": "reminders",
            "route": "/api/events/reminder-triggered"
        }
    ]

    logger.info("Dapr subscriptions requested", extra={"subscriptions": subscriptions})
    return subscriptions


@router.post("/events/task-completed")
async def handle_task_completed(request: Request) -> Dict[str, Any]:
    """
    Dapr subscriber endpoint for task.completed events.

    Called by Dapr when a task.completed event is published to the task-events topic.
    This endpoint is responsible for creating the next instance of a recurring task.

    Event Flow:
        1. User completes a recurring task via API
        2. Backend publishes task.completed event to Dapr Pub/Sub
        3. Dapr forwards event to this endpoint
        4. This endpoint creates the next task instance

    Request Body:
        {
            "event_type": "task.completed",
            "event_id": "uuid-v4",
            "timestamp": "ISO-8601 datetime",
            "task_id": integer,
            "user_id": integer,
            "task_data": {
                "title": string,
                "description": string | null,
                "priority": "HIGH" | "MEDIUM" | "LOW",
                "tags": [string],
                "recurrence_rule": "DAILY" | "WEEKLY" | "MONTHLY" | null,
                "due_date": "ISO-8601 datetime" | null
            }
        }

    Returns:
        {"status": "processed" | "ignored", "new_task_id": int | null, "reason": string | None}

    Note:
        - Returns {"status": "ignored"} if task has no recurrence_rule
        - Returns {"status": "processed"} if next task instance was created
    """
    try:
        event_data = await request.json()
        logger.info("Received task.completed event", extra={"event": event_data})

        # Extract task data
        task_id = event_data.get("task_id")
        task_data = event_data.get("task_data", {})
        recurrence_rule = task_data.get("recurrence_rule")

        # Ignore non-recurring tasks
        if not recurrence_rule:
            return {
                "status": "ignored",
                "reason": "not a recurring task",
                "new_task_id": None
            }

        # Import here to avoid circular dependency
        from ..services.recurrence_service import recurrence_service

        # Create next instance
        new_task = await recurrence_service.create_next_instance(
            task_id=task_id,
            event_data=event_data
        )

        return {
            "status": "processed",
            "new_task_id": new_task.id if new_task else None
        }

    except Exception as e:
        logger.error(f"Error processing task.completed event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/reminder-triggered")
async def handle_reminder_triggered(request: Request) -> Dict[str, Any]:
    """
    Dapr subscriber endpoint for reminder.triggered events.

    Called by Dapr when a reminder event is published to the reminders topic.
    This endpoint processes the reminder (currently just logs for verification).

    In future phases, this would deliver notifications (email, push, etc.).

    Request Body:
        {
            "event_type": "reminder.triggered",
            "event_id": "uuid-v4",
            "timestamp": "ISO-8601 datetime",
            "task_id": integer,
            "user_id": integer,
            "task_title": string,
            "reminder_time": "ISO-8601 datetime",
            "due_date": "ISO-8601 datetime" | null
        }

    Returns:
        {"status": "processed"}
    """
    try:
        event_data = await request.json()
        logger.info("Received reminder.triggered event", extra={"event": event_data})

        # Process reminder (currently just log)
        # In future: call notification service
        task_id = event_data.get("task_id")
        user_id = event_data.get("user_id")
        task_title = event_data.get("task_title", "Unknown")

        logger.info(
            f"Reminder triggered for task '{task_title}' (id: {task_id}, user: {user_id})"
        )

        return {"status": "processed"}

    except Exception as e:
        logger.error(f"Error processing reminder event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/dapr")
async def dapr_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for Dapr integration.

    Returns:
        Dapr service availability status
    """
    from ..services.dapr_service import dapr_service

    is_available = dapr_service.is_available()

    return {
        "status": "healthy" if is_available else "unhealthy",
        "dapr_available": is_available,
        "dapr_url": dapr_service.DAPR_BASE_URL
    }
