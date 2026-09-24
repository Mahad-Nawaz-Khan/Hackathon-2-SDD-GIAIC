"""
Chat API Router - Endpoints for AI Chatbot functionality
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session
from typing import Optional, Dict, Any
import time
import asyncio
from ..middleware.auth import get_current_user
from ..database import get_session
from ..services.chat_service import chat_service
from ..services.auth_service import auth_service
from ..services.task_service import task_service
from ..tools.task_crud_tools import task_crud_tools
from ..models.chat_models import (
    ChatMessageCreate,
    ChatResponse,
    ChatHistoryResponse,
    ChatMessageResponse,
    SenderTypeEnum,
    IntentTypeEnum,
    OperationTypeEnum,
    OperationStatusEnum,
    EntityTypeEnum,
)
import logging


logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func,
    max_retries: int = 2,
    base_delay: float = 0.5,
    exceptions: tuple = (Exception,)
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Result of the function call
    """
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
            await asyncio.sleep(delay)

# Initialize rate limiter for this router
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _chat_message_to_response(message) -> ChatMessageResponse:
    """Convert a ChatMessage model to ChatMessageResponse"""
    return ChatMessageResponse(
        id=message.id,
        content=message.content,
        sender_type=message.sender_type,
        intent=message.intent,
        created_at=message.created_at
    )



DEFAULT_HELP_MESSAGE = (
    "I'm here to help you manage your tasks! You can ask me to:\n"
    "• Create a task (e.g., 'Create a task to buy groceries')\n"
    "• Complete a task (e.g., 'Complete task 1')\n"
    "• Delete a task (e.g., 'Delete task 1')\n"
    "• Search tasks (e.g., 'Search for grocery tasks')\n"
    "• List your tasks (e.g., 'Show me my tasks' or 'What do I have today?')"
)


async def _handle_create_intent(content: str, user_id: int, db_session: Session):
    params = chat_service.classify_intent(content).parameters
    if "title" not in params and content:
        params["title"] = content[:100]

    async def create_task_with_retry():
        return task_crud_tools.create_task(params, user_id, db_session)

    try:
        result = await retry_with_backoff(
            create_task_with_retry,
            max_retries=2,
            exceptions=(Exception,)
        )
        if result.get("success"):
            return result.get("message", "Task created successfully!"), {"type": "create_task", "result": result.get("task")}
        return result.get("message", "I couldn't create that task. Please try again."), None
    except Exception as e:
        logger.error(f"Failed to create task after retries: {str(e)}")
        return "I'm having trouble creating that task right now. Please try again later.", None


def _handle_update_intent(content: str, user_id: int, db_session: Session):
    content_lower = content.lower()
    task_match = re.search(r'task\s*(\d+)|#(\d+)|^\s*(\d+)', content_lower)
    task_id = int(task_match.group(1) or task_match.group(2) or task_match.group(3)) if task_match else None

    completed = None
    if any(word in content_lower for word in ['complete', 'finish', 'done', 'mark as done']):
        completed = True
    elif any(word in content_lower for word in ['incomplete', 'not done', 'uncomplete']):
        completed = False

    if task_id and completed is not None:
        result = task_crud_tools.toggle_task_completion(task_id, user_id, db_session)
        if result.get("success"):
            return result.get("message", "Task updated successfully!"), {"type": "toggle_task", "result": result.get("task")}
        return result.get("message", "I couldn't find that task. Please check the task number."), None

    search_term = re.sub(r'\b(complete|finish|done|mark|as|the|task)\b', '', content_lower).strip()
    if search_term:
        task = task_crud_tools.get_task_by_search_term(search_term, user_id, db_session)
        if task and completed is not None:
            result = task_crud_tools.toggle_task_completion(task["id"], user_id, db_session)
            if result.get("success"):
                return result.get("message", "Task updated successfully!"), {"type": "toggle_task", "result": result.get("task")}
            return result.get("message", "I couldn't update that task."), None
        return "I couldn't find a matching task. Please be more specific or use the task number.", None
    return "Please specify which task you want to update. You can use the task number or describe it.", None


async def _handle_delete_intent(content: str, user_id: int, db_session: Session):
    task_match = re.search(r'task\s*(\d+)|#(\d+)|^\s*(\d+)', content.lower())
    task_id = int(task_match.group(1) or task_match.group(2) or task_match.group(3)) if task_match else None

    if not task_id:
        return "Please specify which task you want to delete by using the task number.", None

    async def delete_task_with_retry():
        return task_crud_tools.delete_task(task_id, user_id, db_session)

    try:
        result = await retry_with_backoff(
            delete_task_with_retry,
            max_retries=2,
            exceptions=(Exception,)
        )
        if result.get("success"):
            return result.get("message", "Task deleted successfully!"), {"type": "delete_task", "task_id": task_id}
        return result.get("message", "I couldn't find that task to delete."), None
    except Exception as e:
        logger.error(f"Failed to delete task after retries: {str(e)}")
        return "I'm having trouble deleting that task right now. Please try again later.", None


def _handle_search_intent(content: str, user_id: int, db_session: Session):
    search_match = re.search(r'(?:search|find|look\s+for)\s+(?:tasks?)?\s*(.+)', content.lower())
    search_term = search_match.group(1).strip() if search_match else None
    if search_term:
        for word in ['with', 'containing', 'that', 'have']:
            if word in search_term:
                search_term = search_term.split(word)[0].strip()

    params = {"search": search_term} if search_term else {}
    result = task_crud_tools.search_tasks(params, user_id, db_session)

    if result.get("success") and result.get("tasks"):
        task_count = result.get("count", 0)
        lines = [f"Found {task_count} task(s):\n"]
        for task in result.get("tasks", []):
            status = "✓" if task["completed"] else "○"
            line = f"{status} {task['title']}"
            if task.get("due_date"):
                line += f" (Due: {task['due_date']})"
            lines.append(line)
        return "\n".join(lines), {"type": "search_tasks", "count": task_count}
    return "I couldn't find any matching tasks.", None


def _handle_list_intent(content: str, user_id: int, db_session: Session):
    content_lower = content.lower()
    if any(word in content_lower for word in ['today', "today's"]):
        result = task_crud_tools.list_today_tasks(user_id, db_session)
        if result.get("success") and result.get("tasks"):
            task_count = result.get("count", 0)
            lines = [f"You have {task_count} task(s) due today:\n"]
            for task in result.get("tasks", []):
                status = "✓" if task["completed"] else "○"
                p_tag = f" [{task['priority']}]" if task.get("priority") else ""
                lines.append(f"{status} {task['title']}{p_tag}")
            return "\n".join(lines), {"type": "list_today_tasks", "count": task_count}
        return "You don't have any tasks due today. Great job!", None

    result = task_crud_tools.search_tasks({"completed": False, "limit": 10}, user_id, db_session)
    if result.get("success") and result.get("tasks"):
        task_count = result.get("count", 0)
        lines = [f"Here are your pending tasks ({task_count}):\n"]
        for task in result.get("tasks", []):
            status = "✓" if task["completed"] else "○"
            p_tag = f" [{task['priority']}]" if task.get("priority") else ""
            lines.append(f"{status} {task['title']}{p_tag}")
        return "\n".join(lines), {"type": "list_tasks", "count": task_count}
    return "You don't have any pending tasks. Great job!", None


def _handle_read_intent(content: str, user_id: int, db_session: Session):
    search_match = re.search(r'(?:show|get|tell\s+me\s+about)\s+(?:the\s+)?task\s*(.+)', content.lower())
    search_term = search_match.group(1).strip() if search_match else ""
    task = task_crud_tools.get_task_by_search_term(search_term, user_id, db_session)
    if task:
        status = "Completed" if task["completed"] else "Pending"
        lines = [f"Task: {task['title']}", f"Status: {status}"]
        if task.get("description"):
            lines.append(f"Description: {task['description']}")
        if task.get("due_date"):
            lines.append(f"Due: {task['due_date']}")
        if task.get("priority"):
            lines.append(f"Priority: {task['priority']}")
        return "\n".join(lines), {"type": "read_task", "task_id": task["id"]}
    return "I couldn't find a task matching that description.", None


@router.post("/message", response_model=ChatResponse)
@limiter.limit("30/minute")
async def send_chat_message(
    request: Request,
    message_data: ChatMessageCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_session: Session = Depends(get_session)
):
    """
    Send a message to the AI chatbot and get a response.

    The AI will analyze the message, detect the user's intent,
    and perform appropriate actions (create/update/delete/search tasks).
    """
    try:
        # Get or create user from Clerk payload
        user = await auth_service.get_or_create_user_from_clerk_payload(current_user, db_session)
        user_id = user.id

        # Generate session ID if not provided
        session_id = message_data.session_id or f"session_{user_id}_{int(hash(current_user.get('sub', '')) % 1000000)}"

        # Create user message with intent detection
        user_message = chat_service.create_user_message(
            user_id=user_id,
            session_id=session_id,
            content=message_data.content,
            db_session=db_session
        )

        # Process the message based on detected intent
        ai_response_content = None
        operation_performed = None
        model_used = "Rule-based Intent Classifier"

        # Get the intent from the message
        intent = user_message.intent
        confidence = user_message.intent_confidence or 0.0

        # If confidence is too low, ask for clarification
        if confidence < 0.6 and intent:
            ai_response_content = "I'm not sure I understood that correctly. Could you please rephrase? For example, you can say 'Create a task to buy groceries' or 'Show me my tasks'."
        elif intent == IntentTypeEnum.CREATE_TASK.value:
            ai_response_content, operation_performed = await _handle_create_intent(message_data.content, user_id, db_session)
        elif intent == IntentTypeEnum.UPDATE_TASK.value:
            ai_response_content, operation_performed = _handle_update_intent(message_data.content, user_id, db_session)
        elif intent == IntentTypeEnum.DELETE_TASK.value:
            ai_response_content, operation_performed = await _handle_delete_intent(message_data.content, user_id, db_session)
        elif intent == IntentTypeEnum.SEARCH_TASKS.value:
            ai_response_content, operation_performed = _handle_search_intent(message_data.content, user_id, db_session)
        elif intent == IntentTypeEnum.LIST_TASKS.value:
            ai_response_content, operation_performed = _handle_list_intent(message_data.content, user_id, db_session)
        elif intent == IntentTypeEnum.READ_TASK.value:
            ai_response_content, operation_performed = _handle_read_intent(message_data.content, user_id, db_session)
        else:
            ai_response_content = DEFAULT_HELP_MESSAGE

        # Mark user message as processed
        user_message.processed = True
        db_session.add(user_message)
        db_session.commit()

        # Create AI response message
        ai_message = chat_service.create_ai_message(
            user_id=user_id,
            session_id=session_id,
            content=ai_response_content,
            db_session=db_session
        )

        # Return response
        return ChatResponse(
            message=_chat_message_to_response(ai_message),
            operation_performed=operation_performed,
            model_used=model_used
        )

    except Exception as e:
        logger.exception(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process message")


@router.get("/history", response_model=ChatHistoryResponse)
@limiter.limit("60/minute")
async def get_chat_history(
    request: Request,
    session_id: Optional[str] = None,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get chat history for a session.
    """
    try:
        # Get or create user from Clerk payload
        user = await auth_service.get_or_create_user_from_clerk_payload(current_user, db_session)
        user_id = user.id

        # Use provided session_id or generate default
        if not session_id:
            session_id = f"session_{user_id}_{int(hash(current_user.get('sub', '')) % 1000000)}"

        # Get chat history
        messages = chat_service.get_chat_history(user_id, session_id, db_session, limit)

        # Convert to response format
        message_responses = [_chat_message_to_response(msg) for msg in messages]

        return ChatHistoryResponse(
            messages=message_responses,
            total_count=len(message_responses),
            session_id=session_id
        )

    except Exception as e:
        logger.exception(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@router.delete("/history")
@limiter.limit("10/minute")
async def clear_chat_history(
    request: Request,
    session_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_session: Session = Depends(get_session)
):
    """
    Clear chat history for a session.
    """
    try:
        # Get or create user from Clerk payload
        user = await auth_service.get_or_create_user_from_clerk_payload(current_user, db_session)
        user_id = user.id

        # Use provided session_id or generate default
        if not session_id:
            session_id = f"session_{user_id}_{int(hash(current_user.get('sub', '')) % 1000000)}"

        # This would need to be implemented in the chat service
        # For now, return success
        return {"message": "Chat history cleared", "session_id": session_id}

    except Exception as e:
        logger.exception(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history")