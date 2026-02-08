"""
OpenAI Agents SDK Integration Service

This service integrates the OpenAI Agents SDK for processing user messages
and managing task operations through natural language.

Uses Gemini API (free tier) via OpenAI-compatible endpoint.

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


def _set_tool_context(db_session: Session, user_id: int):
    """Set the global tool context for the current request."""
    global _tool_context
    _tool_context = ToolContext(db_session=db_session, user_id=user_id)


def _clear_tool_context():
    """Clear the global tool context."""
    global _tool_context
    _tool_context = None


def _get_task_service():
    """Lazy import of task service to avoid circular imports."""
    from ..services.task_service import task_service
    return task_service


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
            return f"✓ Task '{task.title}' deleted successfully!"
        else:
            return f"Sorry, I couldn't delete task #{task_id}."
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        return f"Sorry, I couldn't delete that task. Error: {str(e)}"


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

        # Gemini API configuration
        self._gemini_api_key = os.getenv("GEMINI_API_KEY")
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    def initialize(self):
        """Initialize the OpenAI Agents SDK with Gemini API."""
        if self._initialized:
            return

        try:
            from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel
            from agents import function_tool, AsyncOpenAI

            if not self._gemini_api_key:
                logger.warning("GEMINI_API_KEY not found, OpenAI Agents SDK will not be available")
                return

            # Create external OpenAI client for Gemini
            external_client = AsyncOpenAI(
                api_key=self._gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

            # Create the model wrapper
            model = OpenAIChatCompletionsModel(
                model=self._model_name,
                openai_client=external_client
            )

            # Create run config
            self._run_config = RunConfig(
                model=model,
                model_provider=external_client,
                tracing_disabled=True
            )

            # Decorate the implementation functions as tools
            create_task_tool = function_tool(create_task_impl)
            update_task_tool = function_tool(update_task_impl)
            toggle_task_tool = function_tool(toggle_task_completion_impl)
            delete_task_tool = function_tool(delete_task_impl)
            search_tasks_tool = function_tool(search_tasks_impl)
            list_tasks_tool = function_tool(list_tasks_impl)
            get_task_tool = function_tool(get_task_impl)

            self._tools = [
                create_task_tool,
                update_task_tool,
                toggle_task_tool,
                delete_task_tool,
                search_tasks_tool,
                list_tasks_tool,
                get_task_tool,
            ]

            # Create the agent with tools
            self._agent = Agent(
                name="TaskManager",
                instructions=(
                    "You are a friendly and helpful task management assistant. "
                    "Your job is to help users manage their tasks through natural conversation.\n\n"
                    "TASK CREATION RULES:\n"
                    "- When users mention things they need to do, want to do, or should remember, ALWAYS create a task for them.\n"
                    "- Extract a clear, concise task title from their message. Don't use their entire message as the title.\n"
                    "  Examples:\n"
                    "  - User: 'can you add a task for eating potato' → Task title: 'Eat potato'\n"
                    "  - User: 'remind me to buy groceries tomorrow' → Task title: 'Buy groceries'\n"
                    "  - User: 'I need to call mom' → Task title: 'Call mom'\n"
                    "  - User: 'add a task to finish my homework' → Task title: 'Finish homework'\n"
                    "- If the user mentions a deadline (tomorrow, next week, etc.), set the due_date.\n"
                    "- If the user indicates high importance, set priority to HIGH.\n\n"
                    "GENERAL BEHAVIOR:\n"
                    "- Be conversational and friendly. Use natural language like 'Sure!', 'I've got that', 'Done!', etc.\n"
                    "- After creating a task, confirm what you did in a friendly way.\n"
                    "- If users ask to see their tasks, list them clearly.\n"
                    "- If users ask to complete/update/delete a task, do it and confirm.\n\n"
                    "Remember: You're here to make task management effortless through natural conversation!"
                ),
                tools=self._tools
            )

            self._Runner = Runner
            self._initialized = True
            logger.info(f"OpenAI Agents SDK initialized successfully with Gemini model: {self._model_name}")

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
                if name and name.lower() != "there":
                    user_name = name

            # Build input with conversation history for context
            # Use the messages directly as input history for the Runner
            messages_for_context = []

            # Add conversation history
            if conversation_history and len(conversation_history) > 0:
                recent_messages = conversation_history[-5:]
                for msg in recent_messages:
                    role = "user" if msg.get("sender_type") == "USER" else "assistant"
                    messages_for_context.append({
                        "role": role,
                        "content": msg.get('content', '')
                    })

            # Add the current message
            current_msg = content
            if user_name:
                current_msg = f"(Your name is {user_name}) {content}"

            messages_for_context.append({
                "role": "user",
                "content": current_msg
            })

            # Run the agent
            result = await self._Runner.run(
                self._agent,
                input=content,
                messages=messages_for_context[:-1],  # Pass history as previous messages
                run_config=self._run_config
            )

            response_content = result.final_output if result.final_output else "I'm sorry, I couldn't process that request."
            operation_performed = self._extract_operations(result)

            logger.info(f"Agent processed message for user {user_id}")

            return {
                "success": True,
                "content": response_content,
                "operation_performed": operation_performed,
                "model_used": f"OpenAI Agents SDK (Gemini: {self._model_name})"
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
                if name and name.lower() != "there":
                    user_name = name

            # Build input with conversation history for context
            # Use the messages directly as input history for the Runner
            messages_for_context = []

            # Add conversation history
            if conversation_history and len(conversation_history) > 0:
                recent_messages = conversation_history[-5:]
                for msg in recent_messages:
                    role = "user" if msg.get("sender_type") == "USER" else "assistant"
                    messages_for_context.append({
                        "role": role,
                        "content": msg.get('content', '')
                    })

            # Add the current message
            current_msg = content
            if user_name:
                current_msg = f"(Your name is {user_name}) {content}"

            messages_for_context.append({
                "role": "user",
                "content": current_msg
            })

            result = await self._Runner.run(
                self._agent,
                input=content,
                messages=messages_for_context[:-1],  # Pass history as previous messages
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
                "model_used": f"OpenAI Agents SDK (Gemini: {self._model_name})"
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
        try:
            if hasattr(result, 'new_items') and result.new_items:
                for item in result.new_items:
                    if hasattr(item, 'type') and 'tool_call' in item.type:
                        return {
                            "type": getattr(item.agent, 'name', 'unknown') if hasattr(item, 'agent') else 'tool',
                            "tool_used": item.name if hasattr(item, 'name') else 'unknown'
                        }
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
