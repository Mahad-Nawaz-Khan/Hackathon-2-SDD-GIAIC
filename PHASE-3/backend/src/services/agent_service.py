"""
OpenAI Agents SDK Integration Service

This service integrates the OpenAI Agents SDK for processing user messages
and managing task operations through natural language.

Uses Gemini API (free tier) via OpenAI-compatible interface.
"""

import logging
import os
import json
import asyncio
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime

from sqlmodel import Session

from ..models.chat_models import (
    ChatMessage,
    IntentDetectionResult,
    IntentTypeEnum,
)
from ..mcp.server import (
    set_task_context,
    clear_task_context,
)


logger = logging.getLogger(__name__)


class AgentService:
    """
    Service for managing OpenAI Agents SDK integration.

    This service uses the OpenAI Agents SDK to process user messages,
    detect intent, and execute task operations using MCP tools.

    Uses Gemini API (free tier) via OpenAI-compatible interface.
    """

    def __init__(self):
        self._initialized = False
        self._agent = None
        self._Runner = None
        self._RunConfig = None
        self._run_config = None

        # Gemini API configuration
        self._gemini_api_key = os.getenv("GEMINI_API_KEY")
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    def initialize(self):
        """
        Initialize the OpenAI Agents SDK with Gemini API.

        This should be called once at application startup.
        """
        if self._initialized:
            return

        try:
            # Import the agents library
            from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI

            # Check for Gemini API key
            if not self._gemini_api_key:
                logger.warning("GEMINI_API_KEY not found, OpenAI Agents SDK will not be available")
                self._initialized = False
                return

            # Import MCP tools
            from ..mcp.server import (
                create_task,
                update_task,
                toggle_task_completion,
                delete_task,
                search_tasks,
                list_today_tasks,
                get_task,
            )

            # Create external OpenAI client configured for Gemini
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

            # Create the task management agent
            self._agent = Agent(
                name="TaskManager",
                instructions=(
                    "You are a helpful task management assistant. "
                    "You help users create, read, update, delete, and search for tasks. "
                    "When a user asks to do something with tasks, use the available tools to perform the operation. "
                    "Always provide clear, friendly responses about what actions you've taken. "
                    "If you need more information, ask the user for clarification. "
                    "Be concise and helpful in your responses."
                )
            )

            # Register MCP tools with the agent
            # The tools from MCP server need to be wrapped as agent tools
            self._tools = [
                self._wrap_mcp_tool(create_task, "create_task"),
                self._wrap_mcp_tool(update_task, "update_task"),
                self._wrap_mcp_tool(toggle_task_completion, "toggle_task_completion"),
                self._wrap_mcp_tool(delete_task, "delete_task"),
                self._wrap_mcp_tool(search_tasks, "search_tasks"),
                self._wrap_mcp_tool(list_today_tasks, "list_today_tasks"),
                self._wrap_mcp_tool(get_task, "get_task"),
            ]

            self._Runner = Runner
            self._RunConfig = RunConfig
            self._initialized = True
            logger.info(f"OpenAI Agents SDK initialized successfully with Gemini model: {self._model_name}")

        except ImportError as e:
            logger.warning(f"OpenAI Agents SDK not available: {e}")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI Agents SDK: {e}")
            self._initialized = False

    def _wrap_mcp_tool(self, tool_func, tool_name: str):
        """
        Wrap an MCP tool function for use with the OpenAI Agents SDK.

        Args:
            tool_func: The MCP tool function to wrap
            tool_name: Name for the tool

        Returns:
            Wrapped function compatible with OpenAI Agents SDK
        """
        from agents import function_tool

        # Get the function's docstring for description
        description = tool_func.__doc__ or f"MCP tool: {tool_name}"

        @function_tool
        async def wrapped_tool(**kwargs):
            """Execute the MCP tool with current context."""
            try:
                result = tool_func(**kwargs)
                # Ensure result is a dict
                if not isinstance(result, dict):
                    result = {"result": result}
                return result
            except Exception as e:
                logger.error(f"Error executing MCP tool {tool_name}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to execute {tool_name}"
                }

        # Set the name and description
        wrapped_tool.__name__ = tool_name
        wrapped_tool.__doc__ = description

        return wrapped_tool

    def is_available(self) -> bool:
        """Check if the OpenAI Agents SDK is available and initialized."""
        return self._initialized and self._agent is not None

    async def process_message(
        self,
        content: str,
        user_id: int,
        db_session: Session,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message using the OpenAI Agents SDK.

        Args:
            content: The user's message content
            user_id: The internal user ID
            db_session: Database session
            conversation_history: Optional conversation history for context

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
            # Set the task context for MCP tools
            set_task_context(db_session, user_id)

            # Build input with conversation history
            input_text = content
            if conversation_history:
                # Add recent conversation context
                recent_messages = conversation_history[-5:]  # Last 5 messages
                context_parts = []
                for msg in recent_messages:
                    sender = "User" if msg.get("sender_type") == "USER" else "Assistant"
                    context_parts.append(f"{sender}: {msg.get('content', '')}")
                if context_parts:
                    input_text = "Previous conversation:\n" + "\n".join(context_parts) + f"\n\nCurrent user message: {content}"

            # Run the agent with sync method (Gemini doesn't support async streaming in the same way)
            result = self._Runner.run_sync(
                self._agent,
                input=input_text,
                run_config=self._run_config
            )

            # Get the final output
            response_content = result.final_output if result.final_output else "I'm sorry, I couldn't process that request."

            # Extract any tool calls from the result
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
            # Clear the task context
            clear_task_context()

    async def process_message_streamed(
        self,
        content: str,
        user_id: int,
        db_session: Session,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a user message using the OpenAI Agents SDK with streaming.

        Note: Gemini API has limited streaming support, so this uses sync run
        with simulated streaming for better UX.

        Args:
            content: The user's message content
            user_id: The internal user ID
            db_session: Database session
            conversation_history: Optional conversation history for context

        Yields:
            Dictionary with streaming events including content deltas and final response
        """
        if not self.is_available():
            yield {
                "type": "error",
                "content": "I'm sorry, the AI service is not available right now. Please try again later."
            }
            return

        try:
            # Set the task context for MCP tools
            set_task_context(db_session, user_id)

            # Build input with conversation history
            input_text = content
            if conversation_history:
                recent_messages = conversation_history[-5:]
                context_parts = []
                for msg in recent_messages:
                    sender = "User" if msg.get("sender_type") == "USER" else "Assistant"
                    context_parts.append(f"{sender}: {msg.get('content', '')}")
                if context_parts:
                    input_text = "Previous conversation:\n" + "\n".join(context_parts) + f"\n\nCurrent user message: {content}"

            # For Gemini, use run_sync and simulate streaming
            # This is because Gemini's OpenAI-compatible interface has limited streaming support
            result = self._Runner.run_sync(
                self._agent,
                input=input_text,
                run_config=self._run_config
            )

            # Get the final output
            final_output = result.final_output if result.final_output else "I'm sorry, I couldn't process that request."

            # Simulate streaming by yielding chunks of the response
            # This provides better UX even with non-streaming Gemini API
            chunk_size = 10  # Characters per chunk
            for i in range(0, len(final_output), chunk_size):
                chunk = final_output[i:i + chunk_size]
                yield {
                    "type": "content_delta",
                    "content": chunk
                }
                # Small delay to simulate streaming
                await asyncio.sleep(0.02)

            # Extract any tool calls from the result
            operation_performed = self._extract_operations(result)

            # Yield the final response
            yield {
                "type": "final",
                "content": final_output,
                "operation_performed": operation_performed,
                "model_used": f"OpenAI Agents SDK (Gemini: {self._model_name})"
            }

            logger.info(f"Agent processed message (simulated streaming) for user {user_id}")

        except Exception as e:
            logger.error(f"Error processing message with OpenAI Agents SDK (streamed): {e}")
            yield {
                "type": "error",
                "content": "I'm sorry, I encountered an error processing your request. Please try again."
            }
        finally:
            # Clear the task context
            clear_task_context()

    def _extract_operations(self, result) -> Optional[Dict[str, Any]]:
        """
        Extract information about operations performed from the agent result.

        Args:
            result: The agent run result

        Returns:
            Dictionary with operation information or None
        """
        try:
            # Check if any tools were called
            # This is a simplified check - in production you'd analyze the result more carefully
            if hasattr(result, 'context') and hasattr(result.context, 'items'):
                for item in result.context.items:
                    if item.type == "tool_call_item":
                        return {
                            "type": item.name,
                            "tool_used": item.name
                        }
        except Exception as e:
            logger.debug(f"Could not extract operations: {e}")

        return None

    def classify_intent(self, message: str) -> IntentDetectionResult:
        """
        Classify the intent from a user message using the OpenAI Agents SDK.

        This is a fallback method that uses simple keyword matching
        when the full agent isn't needed.

        Args:
            message: The user's message text

        Returns:
            IntentDetectionResult with detected intent and parameters
        """
        # This is a simplified version - in production we'd use the agent for this
        message_lower = message.lower().strip()

        # Simple keyword-based intent detection
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

        import re

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
