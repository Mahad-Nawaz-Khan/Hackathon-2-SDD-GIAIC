"""
Streaming Chat API Router - Endpoints for AI Chatbot with SSE streaming

This router provides Server-Sent Events (SSE) streaming for real-time
AI responses using the OpenAI Agents SDK.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session
from typing import Optional, Dict, Any, AsyncIterator
import json
import logging
import asyncio

from ..middleware.auth import get_current_user
from ..database import get_session
from ..services.chat_service import chat_service
from ..services.auth_service import auth_service
from ..services.agent_service import agent_service
from ..models.chat_models import (
    ChatMessageCreate,
    SenderTypeEnum,
)


logger = logging.getLogger(__name__)


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api/v1/chat", tags=["chat-streaming"])


async def _stream_response_generator(
    content: str,
    user_id: int,
    session_id: str,
    db_session: Session,
    conversation_history: Optional[list] = None
) -> AsyncIterator[str]:
    """
    Generator function that yields SSE formatted events.

    Args:
        content: User message content
        user_id: Internal user ID
        session_id: Session identifier
        db_session: Database session
        conversation_history: Optional conversation history

    Yields:
        SSE formatted strings
    """
    try:
        # Create user message first
        user_message = chat_service.create_user_message(
            user_id=user_id,
            session_id=session_id,
            content=content,
            db_session=db_session
        )

        # Send initial event
        yield f"event: message_created\ndata: {json.dumps({'id': user_message.id, 'content': content})}\n\n"

        # Check if agent service is available
        if not agent_service.is_available():
            # Fall back to rule-based processing
            from ..chat_router import router as chat_router
            # Get the response from the non-streaming endpoint
            response_data = await _process_with_rule_based(content, user_id, session_id, db_session, user_message)

            # Send as a single chunk
            yield f"event: content\ndata: {json.dumps({'content': response_data['content']})}\n\n"
            yield f"event: done\ndata: {json.dumps(response_data)}\n\n"
            return

        # Process with streaming agent
        full_response_content = ""
        operation_performed = None
        model_used = None

        async for event in agent_service.process_message_streamed(
            content=content,
            user_id=user_id,
            db_session=db_session,
            conversation_history=conversation_history
        ):
            if event["type"] == "content_delta":
                # Stream text content
                full_response_content += event["content"]
                yield f"event: content\ndata: {json.dumps({'content': event['content']})}\n\n"

            elif event["type"] == "tool_call":
                # Notify that a tool is being called
                yield f"event: tool_call\ndata: {json.dumps({'tool': event['tool_name'], 'args': event['tool_args']})}\n\n"

            elif event["type"] == "tool_output":
                # Tool result received
                yield f"event: tool_output\ndata: {json.dumps({'output': event['output']})}\n\n"

            elif event["type"] == "final":
                # Final response
                full_response_content = event.get("content", full_response_content)
                operation_performed = event.get("operation_performed")
                model_used = event.get("model_used")

                # Create AI message in database
                ai_message = chat_service.create_ai_message(
                    user_id=user_id,
                    session_id=session_id,
                    content=full_response_content,
                    db_session=db_session
                )

                # Send final event with complete response
                final_data = {
                    "content": full_response_content,
                    "operation_performed": operation_performed,
                    "model_used": model_used,
                    "message_id": ai_message.id
                }
                yield f"event: done\ndata: {json.dumps(final_data)}\n\n"
                return

            elif event["type"] == "error":
                # Error occurred
                yield f"event: error\ndata: {json.dumps({'error': event['content']})}\n\n"
                yield f"event: done\ndata: {json.dumps({'error': event['content']})}\n\n"
                return

    except Exception as e:
        logger.exception(f"Error in stream generator: {str(e)}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        yield f"event: done\ndata: {json.dumps({'error': str(e)})}\n\n"


async def _process_with_rule_based(
    content: str,
    user_id: int,
    session_id: str,
    db_session: Session,
    user_message
) -> Dict[str, Any]:
    """
    Fallback processing using rule-based intent classification.

    This is used when OpenAI Agents SDK is not available.
    """
    from ..models.chat_models import IntentTypeEnum
    from ..tools.task_crud_tools import task_crud_tools

    # Get the intent
    intent = user_message.intent
    confidence = user_message.intent_confidence or 0.0

    ai_response_content = None
    operation_performed = None
    model_used = "Rule-based Intent Classifier (fallback)"

    if confidence < 0.6 and intent:
        ai_response_content = "I'm not sure I understood that correctly. Could you please rephrase?"
    elif intent == IntentTypeEnum.CREATE_TASK.value:
        params = chat_service.classify_intent(content).parameters
        if "title" not in params and content:
            params["title"] = content[:100]

        result = task_crud_tools.create_task(params, user_id, db_session)
        if result.get("success"):
            ai_response_content = result.get("message", "Task created successfully!")
            operation_performed = {"type": "create_task", "result": result.get("task")}
        else:
            ai_response_content = result.get("message", "I couldn't create that task.")

    elif intent == IntentTypeEnum.LIST_TASKS.value:
        result = task_crud_tools.search_tasks({"completed": False, "limit": 10}, user_id, db_session)
        if result.get("success") and result.get("tasks"):
            task_count = result.get("count", 0)
            ai_response_content = f"Here are your pending tasks ({task_count}):\n\n"
            for task in result.get("tasks", []):
                status = "✓" if task["completed"] else "○"
                ai_response_content += f"{status} {task['title']}\n"
            operation_performed = {"type": "list_tasks", "count": task_count}
        else:
            ai_response_content = "You don't have any pending tasks. Great job!"
    else:
        ai_response_content = "I'm here to help you manage your tasks! You can ask me to create, list, complete, or delete tasks."

    # Create AI message
    ai_message = chat_service.create_ai_message(
        user_id=user_id,
        session_id=session_id,
        content=ai_response_content,
        db_session=db_session
    )

    return {
        "content": ai_response_content,
        "operation_performed": operation_performed,
        "model_used": model_used,
        "message_id": ai_message.id
    }


@router.post("/message/stream")
@limiter.limit("30/minute")
async def send_chat_message_stream(
    request: Request,
    message_data: ChatMessageCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_session: Session = Depends(get_session)
):
    """
    Send a message to the AI chatbot and get a streamed response using Server-Sent Events.

    The streaming endpoint provides real-time token-by-token updates as the AI generates
    its response. Connect with EventSource or similar SSE client.

    Example using JavaScript EventSource:
    ```javascript
    const eventSource = new EventSource('/api/v1/chat/message/stream?content=...' + session_id, {
        method: 'POST',
        headers: { 'Authorization': 'Bearer ...' }
    });

    eventSource.addEventListener('content', (e) => {
        const data = JSON.parse(e.data);
        console.log('Content delta:', data.content);
    });

    eventSource.addEventListener('done', (e) => {
        const data = JSON.parse(e.data);
        console.log('Complete:', data);
        eventSource.close();
    });
    ```
    """
    try:
        # Get or create user from Clerk payload
        user = await auth_service.get_or_create_user_from_clerk_payload(current_user, db_session)
        user_id = user.id

        # Generate session ID if not provided
        session_id = message_data.session_id or f"session_{user_id}_{int(hash(current_user.get('sub', '')) % 1000000)}"

        # Get conversation history for context
        messages = chat_service.get_chat_history(user_id, session_id, db_session, limit=10)
        conversation_history = [
            {
                "sender_type": msg.sender_type,
                "content": msg.content,
                "created_at": msg.created_at.isoformat()
            }
            for msg in messages
        ]

        # Return streaming response
        return StreamingResponse(
            _stream_response_generator(
                content=message_data.content,
                user_id=user_id,
                session_id=session_id,
                db_session=db_session,
                conversation_history=conversation_history
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )

    except Exception as e:
        logger.exception(f"Error processing streaming chat message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process message")
