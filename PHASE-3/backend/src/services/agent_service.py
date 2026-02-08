"""
OpenAI Agents SDK Integration Service

This service integrates the OpenAI Agents SDK for processing user messages
and managing task operations through natural language.

Uses Z.ai API via OpenAI-compatible endpoint.

Context is passed via global context to tools for database access.
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime
from dataclasses import dataclass

from sqlmodel import Session

from ..models.chat_models import (
    IntentDetectionResult,
    IntentTypeEnum,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Context Type for Tools
# ============================================================================

@dataclass
class ToolContext:
    """Context object passed to tools containing database session and user info."""
    db_session: Session
    user_id: int


# Global context (set during each agent run)
_tool_context: Optional[ToolContext] = None

# Track if any tool operation was performed during the current request
_operation_performed: Optional[Dict[str, Any]] = None


def _set_tool_context(db_session: Session, user_id: int):
    """Set the global tool context for the current request."""
    global _tool_context, _operation_performed
    _tool_context = ToolContext(db_session=db_session, user_id=user_id)
    _operation_performed = None  # Reset operations tracker


def _clear_tool_context():
    """Clear the global tool context."""
    global _tool_context, _operation_performed
    _tool_context = None
    _operation_performed = None


def _get_task_service():
    """Lazy import of task service to avoid circular imports."""
    from ..services.task_service import task_service
    return task_service


def _mark_operation_performed(op_type: str, details: Optional[Dict[str, Any]] = None):
    """Mark that an operation was performed by a tool."""
    global _operation_performed
    _operation_performed = {"type": op_type}
    if details:
        _operation_performed.update(details)


def _get_operation_performed() -> Optional[Dict[str, Any]]:
    """Get the operation that was performed and reset the tracker."""
    global _operation_performed
    op = _operation_performed
    return op


# ============================================================================
# Function Tool Implementations
# ============================================================================

def create_task_impl(title: str, description: str = "", priority: str = "MEDIUM", due_date: str = "") -> str:
    """
    Create a new task for the user.

    Args:
        title: The task title (required)
        description: Optional task description
        priority: Priority level (HIGH, MEDIUM, LOW) - default is MEDIUM
        due_date: Optional due date in YYYY-MM-DD format

    Returns:
        A message describing the result
    """
    global _tool_context
    if not _tool_context:
        return "I'm sorry, I couldn't create the task due to a server error."

    try:
        from ..schemas.task import TaskCreateRequest
        task_service = _get_task_service()

        task_data = TaskCreateRequest(
            title=title,
            description=description if description else None,
            priority=priority if priority else "MEDIUM",
            due_date=None
        )

        if due_date:
            try:
                task_data.due_date = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Failed to parse due_date: {due_date}")

        task = task_service.create_task(
            task_data,
            _tool_context.user_id,
            _tool_context.db_session
        )

        logger.info(f"Created task {task.id} for user {_tool_context.user_id}")
        # Mark operation for frontend refresh
        _mark_operation_performed("create_task", {"task_id": task.id})
        return f"✓ Task '{task.title}' created successfully!"
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        return f"Sorry, I couldn't create that task. Error: {str(e)}"


def update_task_impl(task_id: int, title: str = "", description: str = "", priority: str = "", completed: bool = None) -> str:
    """
    Update an existing task.

    Args:
        task_id: The ID of the task to update
        title: New task title (optional)
        description: New task description (optional)
        priority: New priority level - HIGH, MEDIUM, or LOW (optional)
        completed: Mark task as completed/uncompleted (optional)

    Returns:
        A message describing the result
    """
    global _tool_context
    if not _tool_context:
        return "I'm sorry, I couldn't update the task due to a server error."

    try:
        from ..schemas.task import TaskUpdateRequest
        task_service = _get_task_service()

        update_data = {}
        if title:
            update_data["title"] = title
        if description:
            update_data["description"] = description
        if priority:
            update_data["priority"] = priority
        if completed is not None:
            update_data["completed"] = completed

        if not update_data:
            return "Please provide at least one field to update."

        task_update = TaskUpdateRequest(**update_data)
        updated_task = task_service.update_task(
            task_id, task_update, _tool_context.user_id, _tool_context.db_session
        )

        if not updated_task:
            return f"Sorry, I couldn't find task #{task_id} to update."

        logger.info(f"Updated task {task_id} for user {_tool_context.user_id}")
        # Mark operation for frontend refresh
        _mark_operation_performed("update_task", {"task_id": task_id})
        return f"✓ Task '{updated_task.title}' updated successfully!"
    except Exception as e:
        logger.error(f"Error updating task: {str(e)}")
        return f"Sorry, I couldn't update that task. Error: {str(e)}"


def toggle_task_completion_impl(task_id: int) -> str:
    """
    Toggle the completion status of a task.

    Args:
        task_id: The ID of the task to toggle

    Returns:
        A message describing the result
    """
    global _tool_context
    if not _tool_context:
        return "I'm sorry, I couldn't update the task due to a server error."

    try:
        task_service = _get_task_service()
        task = task_service.toggle_task_completion(
            task_id, _tool_context.user_id, _tool_context.db_session
        )

        if not task:
            return f"Sorry, I couldn't find task #{task_id}."

        status = "completed" if task.completed else "not completed"
        logger.info(f"Toggled task {task_id} to {status} for user {_tool_context.user_id}")
        # Mark operation for frontend refresh
        _mark_operation_performed("toggle_task", {"task_id": task_id})
        return f"✓ Task '{task.title}' is now {status}!"
    except Exception as e:
        logger.error(f"Error toggling task completion: {str(e)}")
        return f"Sorry, I couldn't update that task. Error: {str(e)}"


def delete_task_impl(task_id: int) -> str:
    """
    Delete a task.

    Args:
        task_id: The ID of the task to delete

    Returns:
        A message describing the result
    """
    global _tool_context
    if not _tool_context:
        return "I'm sorry, I couldn't delete the task due to a server error."

    try:
        task_service = _get_task_service()

        from ..models.task import Task
        task = _tool_context.db_session.get(Task, task_id)
        if task and task.user_id != _tool_context.user_id:
            task = None

        if not task:
            return f"Sorry, I couldn't find task #{task_id} to delete."

        success = task_service.delete_task(
            task_id, _tool_context.user_id, _tool_context.db_session
        )

        if success:
            logger.info(f"Deleted task {task_id} for user {_tool_context.user_id}")
            # Mark operation for frontend refresh
            _mark_operation_performed("delete_task", {"task_id": task_id})
            return f"✓ Task '{task.title}' deleted successfully!"
        else:
            return f"Sorry, I couldn't delete task #{task_id}."
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        return f"Sorry, I couldn't delete that task. Error: {str(e)}"


def delete_tasks_by_search_impl(search_term: str) -> str:
    """
    Delete tasks that match a search term in their title or description.

    Args:
        search_term: The search term to match against task titles

    Returns:
        A message describing which tasks were deleted
    """
    global _tool_context
    if not _tool_context:
        return "I'm sorry, I couldn't delete tasks due to a server error."

    try:
        task_service = _get_task_service()
        from ..models.task import Task

        # Search for tasks matching the term
        tasks = task_service.get_tasks(
            user_id=_tool_context.user_id,
            db_session=_tool_context.db_session,
            search=search_term,
            limit=50
        )

        if not tasks:
            return f"No tasks found matching '{search_term}'. Nothing was deleted."

        deleted_count = 0
        deleted_titles = []
        for task in tasks:
            success = task_service.delete_task(
                task.id, _tool_context.user_id, _tool_context.db_session
            )
            if success:
                deleted_count += 1
                deleted_titles.append(f"'{task.title}'")

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} tasks matching '{search_term}' for user {_tool_context.user_id}")
            # Mark operation for frontend refresh
            _mark_operation_performed("delete_tasks", {"count": deleted_count})
            if deleted_count == 1:
                return f"✓ Deleted {deleted_titles[0]}!"
            else:
                return f"✓ Deleted {deleted_count} tasks: {', '.join(deleted_titles)}"
        else:
            return f"Found tasks but couldn't delete them. Please try again."

    except Exception as e:
        logger.error(f"Error deleting tasks by search: {str(e)}")
        return f"Sorry, I couldn't delete those tasks. Error: {str(e)}"


def search_tasks_impl(search: str = "", completed: bool = None, priority: str = "", limit: int = 10) -> str:
    """
    Search for tasks based on criteria.

    Args:
        search: Optional search term to match in title/description
        completed: Filter by completion status (true/false)
        priority: Filter by priority - HIGH, MEDIUM, or LOW
        limit: Maximum number of results to return

    Returns:
        A message with the search results
    """
    global _tool_context
    if not _tool_context:
        return "I'm sorry, I couldn't search tasks due to a server error."

    try:
        task_service = _get_task_service()

        tasks = task_service.get_tasks(
            user_id=_tool_context.user_id,
            db_session=_tool_context.db_session,
            search=search if search else None,
            completed=completed,
            priority=priority if priority else None,
            limit=limit
        )

        if not tasks:
            return "You don't have any matching tasks."

        result_lines = [f"Found {len(tasks)} task(s):"]
        for task in tasks:
            status = "✓" if task.completed else "○"
            priority_tag = f"[{task.priority}]" if task.priority else ""
            result_lines.append(f"{status} {task.title} {priority_tag}")
            if task.due_date:
                result_lines.append(f"  Due: {task.due_date.strftime('%Y-%m-%d')}")

        logger.info(f"Searched tasks for user {_tool_context.user_id}, found {len(tasks)} results")
        return "\n".join(result_lines)
    except Exception as e:
        logger.error(f"Error searching tasks: {str(e)}")
        return f"Sorry, I couldn't search tasks. Error: {str(e)}"


def list_tasks_impl(limit: int = 10) -> str:
    """
    List all pending tasks.

    Args:
        limit: Maximum number of tasks to return

    Returns:
        A message with the task list
    """
    global _tool_context
    if not _tool_context:
        return "I'm sorry, I couldn't retrieve tasks due to a server error."

    try:
        task_service = _get_task_service()

        tasks = task_service.get_tasks(
            user_id=_tool_context.user_id,
            db_session=_tool_context.db_session,
            completed=False,
            limit=limit
        )

        if not tasks:
            return "You don't have any pending tasks. Great job!"

        result_lines = [f"Here are your pending tasks ({len(tasks)}):"]
        for task in tasks:
            status = "✓" if task.completed else "○"
            priority_tag = f"[{task.priority}]" if task.priority else ""
            result_lines.append(f"{status} {task.title} {priority_tag}")

        logger.info(f"Listed tasks for user {_tool_context.user_id}")
        return "\n".join(result_lines)
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        return f"Sorry, I couldn't retrieve tasks. Error: {str(e)}"


def get_task_impl(task_id: int) -> str:
    """
    Get details of a specific task.

    Args:
        task_id: The ID of the task to retrieve

    Returns:
        A message with the task details
    """
    global _tool_context
    if not _tool_context:
        return "I'm sorry, I couldn't retrieve the task due to a server error."

    try:
        task_service = _get_task_service()

        task = task_service.get_task_by_id(
            task_id, _tool_context.user_id, _tool_context.db_session
        )

        if not task:
            return f"Sorry, I couldn't find task #{task_id}."

        status = "Completed" if task.completed else "Pending"
        result = f"Task: {task.title}\nStatus: {status}"
        if task.description:
            result += f"\nDescription: {task.description}"
        if task.due_date:
            result += f"\nDue: {task.due_date.strftime('%Y-%m-%d')}"
        if task.priority:
            result += f"\nPriority: {task.priority}"

        logger.info(f"Retrieved task {task_id} for user {_tool_context.user_id}")
        return result
    except Exception as e:
        logger.error(f"Error getting task: {str(e)}")
        return f"Sorry, I couldn't retrieve the task. Error: {str(e)}"


def show_conversation_summary_impl() -> str:
    """
    Show a summary of what has happened in our conversation so far.

    Returns:
        A summary of recent conversation activity
    """
    global _tool_context
    if not _tool_context:
        return "I'm sorry, I couldn't retrieve conversation history."

    try:
        from ..services.chat_service import chat_service

        # Get recent messages from all sessions for this user
        messages = chat_service.get_chat_history(
            user_id=_tool_context.user_id,
            session_id=None,
            db_session=_tool_context.db_session,
            limit=20
        )

        if not messages:
            return "This is the beginning of our conversation! How can I help you with your tasks today?"

        # Count message types
        user_msgs = [m for m in messages if m.sender_type == 'USER']
        ai_msgs = [m for m in messages if m.sender_type == 'AI']

        result_lines = [
            f"Here's what we've discussed ({len(messages)} messages):",
            f"- {len(user_msgs)} messages from you",
            f"- {len(ai_msgs)} responses from me",
            "",
            "Recent messages:"
        ]

        for msg in messages[-10:]:
            sender = "You" if msg.sender_type == 'USER' else "Me"
            content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            result_lines.append(f"- {sender}: {content_preview}")

        return "\n".join(result_lines)

    except Exception as e:
        logger.error(f"Error getting conversation summary: {str(e)}")
        return "Sorry, I couldn't retrieve the conversation summary."


# ============================================================================
# Agent Service Class
# ============================================================================

class AgentService:
    """
    Service for managing OpenAI Agents SDK integration.

    Uses Gemini API (free tier) via OpenAI-compatible endpoint.
    """

    def __init__(self):
        self._initialized = False
        self._agent = None
        self._Runner = None
        self._run_config = None
        self._tools = []

        # Z.ai API configuration (also supports GEMINI_API_KEY as fallback)
        self._gemini_api_key = os.getenv("Z_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
        self._model_name = os.getenv("Z_AI_MODEL", "gpt-4o")

    def initialize(self):
        """Initialize the OpenAI Agents SDK with Z.ai API."""
        if self._initialized:
            return

        try:
            from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, function_tool
            from openai import AsyncOpenAI

            # Use Z.AI_API_KEY for Z.ai (fallback to GEMINI_API_KEY for backward compatibility)
            api_key = os.getenv("Z_AI_API_KEY") or self._gemini_api_key

            if not api_key:
                logger.warning("Z_AI_API_KEY or GEMINI_API_KEY not found, OpenAI Agents SDK will not be available")
                return

            # Create external OpenAI client for Z.ai
            external_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.z.ai/api/coding/paas/v4"
            )

            # Create the model wrapper (use gpt-4o or compatible model)
            model = OpenAIChatCompletionsModel(
                model=os.getenv("Z_AI_MODEL", "gpt-4o"),
                openai_client=external_client
            )

            # Create run config with increased max turns
            self._run_config = RunConfig(
                model=model,
                model_provider=external_client,
                tracing_disabled=True,
                max_turns=50  # Increase from default 10 to 50
            )

            # Decorate the implementation functions as tools
            create_task_tool = function_tool(create_task_impl)
            update_task_tool = function_tool(update_task_impl)
            toggle_task_tool = function_tool(toggle_task_completion_impl)
            delete_task_tool = function_tool(delete_task_impl)
            delete_by_search_tool = function_tool(delete_tasks_by_search_impl)
            search_tasks_tool = function_tool(search_tasks_impl)
            list_tasks_tool = function_tool(list_tasks_impl)
            get_task_tool = function_tool(get_task_impl)
            show_conversation_tool = function_tool(show_conversation_summary_impl)

            self._tools = [
                create_task_tool,
                update_task_tool,
                toggle_task_tool,
                delete_task_tool,
                delete_by_search_tool,
                search_tasks_tool,
                list_tasks_tool,
                get_task_tool,
                show_conversation_tool,
            ]

            # Create the agent with tools
            self._agent = Agent(
                name="TaskManager",
                instructions=(
                    "You are a friendly task management assistant. Help users manage tasks efficiently.\n\n"
                    "TASK CREATION:\n"
                    "- When users mention things to do, create a task with a SHORT, clear title.\n"
                    "- Examples: 'eat potato' not 'can you add a task for eating potato'\n"
                    "- Set priority to HIGH if urgent, MEDIUM otherwise.\n\n"
                    "TASK DELETION:\n"
                    "- Use delete_tasks_by_search for descriptions like 'delete potato tasks'\n"
                    "- Use delete_task only when user gives an ID number.\n\n"
                    "CONVERSATION:\n"
                    "- You CAN see conversation history - it's provided as context.\n"
                    "- Use show_conversation_summary tool when asked about conversation.\n\n"
                    "IMPORTANT: After completing any action, STOP and respond to the user. Do NOT call multiple tools unless explicitly asked."
                ),
                tools=self._tools
            )

            self._Runner = Runner
            self._initialized = True
            logger.info("OpenAI Agents SDK initialized successfully with Z.ai API")

        except ImportError as e:
            logger.warning(f"OpenAI Agents SDK not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI Agents SDK: {e}")

    def is_available(self) -> bool:
        """Check if the OpenAI Agents SDK is available and initialized."""
        return self._initialized and self._agent is not None

    async def process_message(
        self,
        content: str,
        user_id: int,
        db_session: Session,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_info: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message using the OpenAI Agents SDK.

        Args:
            content: The user's message content
            user_id: The internal user ID
            db_session: Database session
            conversation_history: Optional conversation history for context
            user_info: Optional user information for personalization

        Returns:
            Dictionary with the response content and any operations performed
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "OpenAI Agents SDK not available",
                "content": "I'm sorry, the AI service is not available right now. Please try again later."
            }

        try:
            # Set the tool context for this request
            _set_tool_context(db_session, user_id)

            # Get user name for personalization (only if we have a real name)
            user_name = None
            if user_info:
                name = user_info.get("name") or user_info.get("first_name")
                if name and name.lower() not in ("there", "friend"):
                    user_name = name

            # Build the input with user context and conversation history
            input_text = content

            # Prepend context if available
            context_parts = []
            if user_name:
                context_parts.append(f"User's name: {user_name}")

            if conversation_history and len(conversation_history) > 0:
                history_parts = []
                for msg in conversation_history[-5:]:
                    sender = "User" if msg.get("sender_type") == "USER" else "Assistant"
                    history_parts.append(f"{sender}: {msg.get('content', '')}")

                if history_parts:
                    context_parts.append("Recent conversation:")
                    context_parts.extend(history_parts)

            if context_parts:
                input_text = "\n".join(context_parts) + f"\n\nCurrent message: {content}"

            # Run the agent
            result = await self._Runner.run(
                self._agent,
                input=input_text,
                run_config=self._run_config
            )

            response_content = result.final_output if result.final_output else "I'm sorry, I couldn't process that request."
            operation_performed = self._extract_operations(result)

            logger.info(f"Agent processed message for user {user_id}")

            return {
                "success": True,
                "content": response_content,
                "operation_performed": operation_performed,
                "model_used": "OpenAI Agents SDK (Z.ai)"
            }

        except Exception as e:
            logger.error(f"Error processing message with OpenAI Agents SDK: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": "I'm sorry, I encountered an error processing your request. Please try again."
            }
        finally:
            _clear_tool_context()

    async def process_message_streamed(
        self,
        content: str,
        user_id: int,
        db_session: Session,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_info: Optional[Dict[str, str]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a user message with streaming response.

        Args:
            content: The user's message content
            user_id: The internal user ID
            db_session: Database session
            conversation_history: Optional conversation history
            user_info: Optional user information for personalization

        Yields:
            Dictionary with streaming events
        """
        if not self.is_available():
            yield {
                "type": "error",
                "content": "I'm sorry, the AI service is not available right now. Please try again later."
            }
            return

        try:
            _set_tool_context(db_session, user_id)

            # Get user name for personalization (only if we have a real name)
            user_name = None
            if user_info:
                name = user_info.get("name") or user_info.get("first_name")
                if name and name.lower() not in ("there", "friend"):
                    user_name = name

            # Build the input with user context and conversation history
            input_text = content

            # Prepend context if available
            context_parts = []
            if user_name:
                context_parts.append(f"User's name: {user_name}")

            if conversation_history and len(conversation_history) > 0:
                history_parts = []
                for msg in conversation_history[-5:]:
                    sender = "User" if msg.get("sender_type") == "USER" else "Assistant"
                    history_parts.append(f"{sender}: {msg.get('content', '')}")

                if history_parts:
                    context_parts.append("Recent conversation:")
                    context_parts.extend(history_parts)

            if context_parts:
                input_text = "\n".join(context_parts) + f"\n\nCurrent message: {content}"

            result = await self._Runner.run(
                self._agent,
                input=input_text,
                run_config=self._run_config
            )

            final_output = result.final_output if result.final_output else "I'm sorry, I couldn't process that request."

            # Simulate streaming
            chunk_size = 10
            for i in range(0, len(final_output), chunk_size):
                chunk = final_output[i:i + chunk_size]
                yield {
                    "type": "content_delta",
                    "content": chunk
                }
                await asyncio.sleep(0.02)

            operation_performed = self._extract_operations(result)

            yield {
                "type": "final",
                "content": final_output,
                "operation_performed": operation_performed,
                "model_used": "OpenAI Agents SDK (Z.ai)"
            }

            logger.info(f"Agent processed message (streamed) for user {user_id}")

        except Exception as e:
            logger.error(f"Error processing message with OpenAI Agents SDK (streamed): {e}")
            yield {
                "type": "error",
                "content": "I'm sorry, I encountered an error processing your request. Please try again."
            }
        finally:
            _clear_tool_context()

    def _extract_operations(self, result) -> Optional[Dict[str, Any]]:
        """Extract information about operations performed from the agent result."""
        # First check if any tool marked an operation as performed
        global _operation_performed
        if _operation_performed:
            return _operation_performed

        try:
            # Check various possible structures from OpenAI Agents SDK
            # The result structure may vary depending on SDK version

            # Method 1: Check for new_items (older SDK versions)
            if hasattr(result, 'new_items') and result.new_items:
                for item in result.new_items:
                    if hasattr(item, 'type') and 'tool_call' in str(item.type):
                        return {
                            "type": "tool_call",
                            "tool_used": getattr(item, 'name', 'unknown')
                        }

            # Method 2: Check for raw_responses or context
            if hasattr(result, 'raw_responses') and result.raw_responses:
                # Tool calls were made
                return {"type": "tool_call", "count": len(result.raw_responses)}

            # Method 3: Check if final_output contains task operation keywords
            if hasattr(result, 'final_output') and result.final_output:
                output = result.final_output
                if any(keyword in output for keyword in ['✓ Task', 'created successfully!', 'updated successfully!', 'deleted successfully!', 'is now', 'Deleted']):
                    return {"type": "task_operation", "indicated_by": "response_content"}

            # Method 4: Check context for tool calls
            if hasattr(result, 'context') and result.context:
                context = result.context
                if hasattr(context, 'tool_calls') and context.tool_calls:
                    return {"type": "tool_call", "count": len(context.tool_calls)}

            logger.debug(f"Could not extract operations from result type: {type(result)}, attributes: {dir(result)}")
        except Exception as e:
            logger.debug(f"Could not extract operations: {e}")
        return None

    def classify_intent(self, message: str) -> IntentDetectionResult:
        """
        Classify the intent from a user message using keyword matching.

        This is a simplified fallback method.
        """
        import re
        message_lower = message.lower().strip()

        intent_patterns = {
            IntentTypeEnum.CREATE_TASK: [
                r'\b(create|add|make|new)\s+(a\s+)?task',
                r'\b(remind\s+me\s+(to|about))',
                r'\b(need\s+to|should|have\s+to|gotta)\s+',
            ],
            IntentTypeEnum.UPDATE_TASK: [
                r'\b(update|change|edit|modify)\s+(the\s+)?task',
                r'\b(mark|set|change)\s+(the\s+)?task\s*\d*\s+as\s+(completed|done|finished)',
                r'\b(complete|finish|done)\s+(the\s+)?task\s*\d*',
            ],
            IntentTypeEnum.DELETE_TASK: [
                r'\b(delete|remove)\s+(the\s+)?task',
            ],
            IntentTypeEnum.SEARCH_TASKS: [
                r'\b(search|find|look\s+for)\s+(tasks?)',
                r'\b(show\s+me)\s*(tasks?)\s*(with|containing)',
            ],
            IntentTypeEnum.LIST_TASKS: [
                r'\b(today|tomorrow|this\s+week)\s*',
                r'\b(show|list|display|what\s+are)\s*(all\s+)?(my\s+)?tasks?',
                r'\b(get|see|view)\s*(all\s+)?(my\s+)?tasks?',
            ],
            IntentTypeEnum.READ_TASK: [
                r'\b(show|get|tell\s+me\s+about)\s+(the\s+)?task\s*\d+',
            ],
        }

        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return IntentDetectionResult(
                        intent=intent,
                        confidence=0.7,
                        parameters={}
                    )

        return IntentDetectionResult(
            intent=IntentTypeEnum.UNKNOWN,
            confidence=0.0,
            parameters={}
        )


# Singleton instance
agent_service = AgentService()
