"""
Nano Agent - MCP Server Tool with OpenAI Agent SDK.

This module implements the nano agent using OpenAI's Agent SDK for
autonomous task execution with file system tools.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
from pathlib import Path
import json
import asyncio

# OpenAI Agent SDK imports (required)
from agents import Agent, Runner, RunConfig, ModelSettings
from agents.lifecycle import RunHooksBase

# Rich logging imports
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

# Token tracking
from .token_tracking import TokenTracker, format_token_count, format_cost

from .data_types import (
    PromptNanoAgentRequest,
    PromptNanoAgentResponse,
    AgentConfig,
    AgentExecution
)

from .constants import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    MAX_AGENT_TURNS,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS,
    AVAILABLE_TOOLS,
    AVAILABLE_MODELS,
    NANO_AGENT_SYSTEM_PROMPT,
    ERROR_NO_API_KEY,
    ERROR_PROVIDER_NOT_SUPPORTED,
    SUCCESS_AGENT_COMPLETE,
    VERSION,
    PROVIDER_REQUIREMENTS
)

# Import tools from nano_agent_tools
from .nano_agent_tools import get_nano_agent_tools

# Import provider configuration
from .provider_config import ProviderConfig

# Initialize logger and rich console
logger = logging.getLogger(__name__)

# Create a quiet console for MCP server mode
class QuietConsole:
    """A console that doesn't output anything - used when running as MCP server."""
    def print(self, *args, **kwargs):
        pass

# Detect if running as MCP server by checking if we're being imported by MCP modules
import sys
IS_MCP_SERVER = any('mcp' in module_name.lower() for module_name in sys.modules.keys())

# Use quiet console when running as MCP server to avoid JSON parsing errors
console = QuietConsole() if IS_MCP_SERVER else Console()


class RichLoggingHooks(RunHooksBase):
    """Custom lifecycle hooks for rich logging of tool calls and token tracking."""
    
    def __init__(self, token_tracker: Optional[TokenTracker] = None):
        """Initialize the hooks with a console instance and optional token tracker.
        
        Args:
            token_tracker: Optional TokenTracker for monitoring usage
        """
        self.tool_call_count = 0
        self.tool_call_map = {}  # Map tool call number to tool name
        self.token_tracker = token_tracker
    
    async def on_agent_start(self, context, agent):
        """Called when the agent starts."""
        console.print(Panel(
            Text(f"Agent: {agent.name}", style="bold cyan"),
            title="🚀 Agent Started",
            border_style="blue"
        ))
        
        # Track initial token usage if available
        if self.token_tracker and hasattr(context, 'usage'):
            self.token_tracker.update(context.usage)
            logger.debug(f"Initial tokens: {context.usage.total_tokens}")
    
    def _truncate_value(self, value, max_length=100):
        """Truncate a value and add ellipsis if needed."""
        str_value = str(value)
        if len(str_value) > max_length:
            return str_value[:max_length-3] + "..."
        return str_value
    
    def _format_tool_args(self, tool_name):
        """Format tool arguments for display."""
        # For now, return empty dict since we can't access args directly
        # In real implementation, we'd extract from context
        return {}
    
    async def on_tool_start(self, context, agent, tool):
        """Called before a tool is invoked."""
        self.tool_call_count += 1
        
        # Extract tool name 
        tool_name = getattr(tool, 'name', 'Unknown Tool')
        
        # Store mapping for later use
        self.tool_call_map[self.tool_call_count] = tool_name
        self.current_tool_number = self.tool_call_count
        self.current_tool_name = tool_name
        self.current_tool_start_time = time.time()
        
        # Try multiple methods to get tool arguments
        tool_args = {}
        
        # Method 1: Check if tool has call attributes
        if hasattr(tool, 'call') and hasattr(tool.call, 'function'):
            try:
                if hasattr(tool.call.function, 'arguments'):
                    tool_args = json.loads(tool.call.function.arguments)
                    logger.debug(f"Found args in tool.call.function: {tool_args}")
            except Exception as e:
                logger.debug(f"Failed to get args from tool.call: {e}")
        
        # Method 2: Look in context for recent tool calls
        if not tool_args and hasattr(context, 'messages'):
            try:
                # Get the last few messages and look for tool calls
                for msg in reversed(list(context.messages)[-3:]):
                    if hasattr(msg, 'tool_calls'):
                        for tc in msg.tool_calls:
                            if hasattr(tc, 'function') and tc.function.name == tool_name:
                                if hasattr(tc.function, 'arguments'):
                                    tool_args = json.loads(tc.function.arguments)
                                    logger.debug(f"Found args in context messages: {tool_args}")
                                    break
                        if tool_args:
                            break
            except Exception as e:
                logger.debug(f"Failed to get args from context: {e}")
        
        # Method 3: Check the tool object directly
        if not tool_args:
            # Log what the tool object contains for debugging
            logger.debug(f"Tool type: {type(tool)}, dir: {[x for x in dir(tool) if not x.startswith('_')][:10]}")
            if hasattr(tool, '__dict__'):
                logger.debug(f"Tool dict: {list(tool.__dict__.keys())[:10]}")
        
        # Display the tool call with or without arguments
        if tool_args:
            # Truncate values that are too long
            formatted_args = {}
            for key, value in tool_args.items():
                formatted_args[key] = self._truncate_value(value, 100)
            
            args_str = json.dumps(formatted_args, indent=2)
            display_text = f"{tool_name}(\n{args_str}\n)"
            
            console.print(Panel(
                Syntax(display_text, "python", theme="monokai", line_numbers=False),
                title=f"🔧 Tool Call #{self.tool_call_count}",
                border_style="cyan"
            ))
        else:
            # Fallback to simple display
            console.print(Panel(
                Text(f"Invoking: {tool_name}", style="cyan"),
                title=f"🔧 Tool Call #{self.tool_call_count}",
                border_style="cyan"
            ))
        
        # Store args for later use if found
        self.current_tool_args = tool_args
    
    async def on_tool_end(self, context, agent, tool, result):
        """Called after a tool is invoked."""
        tool_name = getattr(tool, 'name', 'Unknown Tool')
        tool_number = getattr(self, 'current_tool_number', 0)
        
        # Calculate execution time
        exec_time = time.time() - getattr(self, 'current_tool_start_time', time.time())
        
        # Try to get the captured arguments from our tools module
        tool_args = {}
        try:
            from .nano_agent_tools import _last_tool_args
            if tool_name in _last_tool_args:
                tool_args = _last_tool_args[tool_name]
                # Clear after use to avoid showing stale args
                del _last_tool_args[tool_name]
        except:
            pass
        
        # Process the result for display
        result_str = str(result)
        
        # Determine if result was truncated
        was_truncated = False
        max_result_length = 200
        
        # Format the result based on its content
        truncation_note = ""
        if "Error:" in result_str:
            # Display errors prominently
            if len(result_str) > max_result_length:
                display_result = result_str[:max_result_length-3] + "..."
                truncation_note = f" (truncated, {len(result_str)} chars total)"
                was_truncated = True
            else:
                display_result = result_str
            result_color = "red"
        elif result_str.startswith('{') and result_str.endswith('}'):
            # Try to parse as JSON for better formatting
            try:
                json_result = json.loads(result_str)
                formatted_json = json.dumps(json_result, indent=2)
                if len(formatted_json) > max_result_length:
                    # Truncate JSON intelligently
                    display_result = formatted_json[:max_result_length-3] + "..."
                    truncation_note = f" (truncated, {len(result_str)} chars total)"
                    was_truncated = True
                else:
                    display_result = formatted_json
                result_color = "green"
            except:
                if len(result_str) > max_result_length:
                    display_result = result_str[:max_result_length-3] + "..."
                    truncation_note = f" (truncated, {len(result_str)} chars total)"
                    was_truncated = True
                else:
                    display_result = result_str
                result_color = "green"
        else:
            # Regular text result
            if len(result_str) > max_result_length:
                display_result = result_str[:max_result_length-3] + "..."
                truncation_note = f" (truncated, {len(result_str)} chars total)"
                was_truncated = True
            else:
                display_result = result_str
            result_color = "green"
        
        # Format the function call with arguments and return value
        if tool_args:
            # Truncate argument values that are too long
            formatted_args = {}
            for key, value in tool_args.items():
                formatted_args[key] = self._truncate_value(value, 100)
            
            args_str = json.dumps(formatted_args, indent=2)
            call_display = f"{tool_name}({args_str}) -> {display_result}{truncation_note}"
        else:
            call_display = f"{tool_name}() -> {display_result}{truncation_note}"
        
        # Create panel with tool call number and execution time (no per-tool tokens)
        console.print(Panel(
            Syntax(call_display, "python", theme="monokai", line_numbers=False) if tool_args else Text(call_display, style=result_color),
            title=f"✅ Tool Call #{tool_number} ({exec_time:.2f}s)",
            border_style="green" if result_color == "green" else "red"
        ))
    
    async def on_agent_end(self, context, agent, output):
        """Called when the agent produces final output."""
        # Track final token usage
        if self.token_tracker and hasattr(context, 'usage'):
            self.token_tracker.update(context.usage)
            
            # Show usage summary
            report = self.token_tracker.generate_report()
            usage_text = (
                f"Tokens: {format_token_count(report.total_tokens)} | "
                f"Cost: {format_cost(report.total_cost)}"
            )
            console.print(Panel(
                Text(f"Agent completed successfully\n{usage_text}", style="bold green"),
                title="🎯 Agent Finished",
                border_style="green"
            ))
        else:
            console.print(Panel(
                Text("Agent completed successfully", style="bold green"),
                title="🎯 Agent Finished",
                border_style="green"
            ))


async def _execute_nano_agent_async(request: PromptNanoAgentRequest, enable_rich_logging: bool = True) -> PromptNanoAgentResponse:
    """
    Execute the nano agent using OpenAI Agent SDK (async version).
    
    This method uses the OpenAI Agent SDK for a robust agent experience
    with better tool handling and conversation management.
    
    Args:
        request: The validated request containing prompt and configuration
        enable_rich_logging: Whether to enable rich console logging for tool calls
        
    Returns:
        Response with execution results or error information
    """
    start_time = time.time()
    
    try:
        logger.info(f"Executing nano agent with Agent SDK: {request.agentic_prompt[:100]}...")
        logger.debug(f"Model: {request.model}, Provider: {request.provider}")
        
        # Validate provider and model combination
        is_valid, error_msg = ProviderConfig.validate_provider_setup(
            request.provider, 
            request.model,
            AVAILABLE_MODELS,
            PROVIDER_REQUIREMENTS
        )
        if not is_valid:
            return PromptNanoAgentResponse(
                success=False,
                error=error_msg,
                execution_time_seconds=time.time() - start_time
            )
        
        # Setup provider-specific configurations
        ProviderConfig.setup_provider(request.provider)
        
        # Get tools for the agent
        tools = get_nano_agent_tools()
        
        # Configure model settings based on model capabilities
        base_settings = {
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }
        
        # Get filtered settings for the specific model
        model_settings = ProviderConfig.get_model_settings(
            model=request.model,
            provider=request.provider,
            base_settings=base_settings
        )
        
        # Create agent using the provider configuration
        agent = ProviderConfig.create_agent(
            name="NanoAgent",
            instructions=NANO_AGENT_SYSTEM_PROMPT,
            tools=tools,
            model=request.model,
            provider=request.provider,
            model_settings=model_settings
        )
        
        # Create token tracker and hooks for rich logging if enabled
        token_tracker = TokenTracker(model=request.model, provider=request.provider) if enable_rich_logging else None
        hooks = RichLoggingHooks(token_tracker=token_tracker) if enable_rich_logging else None
        
        # Run the agent asynchronously
        result = await Runner.run(
            agent,
            request.agentic_prompt,
            max_turns=MAX_AGENT_TURNS,
            run_config=RunConfig(
                workflow_name="nano_agent_task",
                trace_metadata={
                    "model": request.model,
                    "provider": request.provider,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            hooks=hooks
        )
        
        execution_time = time.time() - start_time
        
        # Extract the final output
        final_output = result.final_output if hasattr(result, 'final_output') else str(result)
        
        # Check if result has usage information
        if hasattr(result, 'usage') and token_tracker:
            token_tracker.add_usage(
                input_tokens=result.usage.get('prompt_tokens', 0),
                output_tokens=result.usage.get('completion_tokens', 0)
            )
        
        # Prepare metadata
        metadata = {
            "model": request.model,
            "provider": request.provider,
            "turns": len(result.messages) if hasattr(result, 'messages') else 0,
        }
        
        # Add token usage if available
        if token_tracker:
            metadata["token_usage"] = token_tracker.get_summary()
        
        logger.info(f"Agent completed successfully in {execution_time:.2f}s")
        
        return PromptNanoAgentResponse(
            success=True,
            result=final_output,
            metadata=metadata,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        logger.error(f"Agent SDK execution failed: {str(e)}\nFull traceback:\n{full_traceback}")
        execution_time = time.time() - start_time
        
        return PromptNanoAgentResponse(
            success=False,
            error=f"Agent SDK execution failed: {str(e)}",
            metadata={
                "model": request.model,
                "provider": request.provider,
                "error_type": type(e).__name__
            },
            execution_time_seconds=execution_time
        )


def _execute_nano_agent(request: PromptNanoAgentRequest, enable_rich_logging: bool = True) -> PromptNanoAgentResponse:
    """
    Execute the nano agent using OpenAI Agent SDK.
    
    This method uses the OpenAI Agent SDK for a robust agent experience
    with better tool handling and conversation management.
    
    Args:
        request: The validated request containing prompt and configuration
        enable_rich_logging: Whether to enable rich console logging for tool calls
        
    Returns:
        Response with execution results or error information
    """
    start_time = time.time()
    
    try:
        logger.info(f"Executing nano agent with Agent SDK: {request.agentic_prompt[:100]}...")
        logger.debug(f"Model: {request.model}, Provider: {request.provider}")
        
        # Validate provider and model combination
        is_valid, error_msg = ProviderConfig.validate_provider_setup(
            request.provider, 
            request.model,
            AVAILABLE_MODELS,
            PROVIDER_REQUIREMENTS
        )
        if not is_valid:
            return PromptNanoAgentResponse(
                success=False,
                error=error_msg,
                execution_time_seconds=time.time() - start_time
            )
        
        # Setup provider-specific configurations
        ProviderConfig.setup_provider(request.provider)
        
        # Configure model settings based on model capabilities
        base_settings = {
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }
        
        # Get filtered settings for the specific model
        model_settings = ProviderConfig.get_model_settings(
            model=request.model,
            provider=request.provider,
            base_settings=base_settings
        )
        
        # Create agent with provider-specific configuration
        agent = ProviderConfig.create_agent(
            name="NanoAgent",
            instructions=NANO_AGENT_SYSTEM_PROMPT,
            tools=get_nano_agent_tools(),
            model=request.model,
            provider=request.provider,
            model_settings=model_settings
        )
        
        # Create token tracker and hooks for rich logging if enabled
        token_tracker = TokenTracker(model=request.model, provider=request.provider) if enable_rich_logging else None
        hooks = RichLoggingHooks(token_tracker=token_tracker) if enable_rich_logging else None
        
        # Run the agent synchronously (we'll handle async in the wrapper)
        result = Runner.run_sync(
            agent,
            request.agentic_prompt,
            max_turns=MAX_AGENT_TURNS,
            run_config=RunConfig(
                workflow_name="nano_agent_task",
                trace_metadata={
                    "model": request.model,
                    "provider": request.provider,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            hooks=hooks
        )
        
        execution_time = time.time() - start_time
        
        # Extract the final output
        final_output = result.final_output if hasattr(result, 'final_output') else str(result)
        
        # Check if result has usage information
        if hasattr(result, 'usage') and token_tracker:
            logger.debug(f"Result has usage: {result.usage}")
            token_tracker.update(result.usage)
        
        # Build metadata including token usage
        metadata = {
            "model": request.model,
            "provider": request.provider,
            "timestamp": datetime.now().isoformat(),
            "agent_sdk": True,
            "turns_used": len(result.messages) if hasattr(result, 'messages') else None,
        }
        
        # Add token usage information if available
        if token_tracker:
            report = token_tracker.generate_report()
            metadata["token_usage"] = {
                "total_tokens": report.total_tokens,
                "input_tokens": report.total_input_tokens,
                "output_tokens": report.total_output_tokens,
                "cached_tokens": report.cached_input_tokens,
                "total_cost": round(report.total_cost, 4),
            }
        
        response = PromptNanoAgentResponse(
            success=True,
            result=final_output,
            metadata=metadata,
            execution_time_seconds=execution_time
        )
        
        logger.info(f"Agent SDK execution completed successfully in {execution_time:.2f} seconds")
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Agent SDK execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return PromptNanoAgentResponse(
            success=False,
            error=error_msg,
            metadata={
                "error_type": type(e).__name__,
                "model": request.model,
                "provider": request.provider,
                "agent_sdk": True,
            },
            execution_time_seconds=execution_time
        )


async def prompt_nano_agent(
    agentic_prompt: str,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    ctx: Any = None  # Context will be injected by FastMCP when registered
) -> Dict[str, Any]:
    """
    Execute an autonomous agent with REAL FILE SYSTEM OPERATIONS using LangGraph.

    This tool creates an AI agent using LangGraph's create_react_agent (like DeepAgents)
    but performs ACTUAL file system operations instead of virtual ones. Key features:
    - Uses LangGraph create_react_agent architecture (NO OpenAI SDK)
    - Performs REAL file operations with comprehensive verification
    - Every operation is verified on the actual file system
    - Eliminates phantom operations completely
    - Based on DeepAgents architecture but with real file access

    The agent uses direct file system tools that perform actual disk operations
    and verify every single operation to ensure it really happened.

    Args:
        agentic_prompt: Natural language description of the work to be done.
                       Be specific and detailed for best results.
                       Examples:
                       - "Read all Python files in src/ and create a summary document"
                       - "Generate unit tests for the data_processing module"
                       - "Create a REST API with CRUD operations for a todo list"

        model: The LLM model to use for the agent. Options vary by provider:
               OpenAI: gpt-5-mini (default), gpt-5-nano, gpt-5, gpt-4o
               Anthropic: claude-opus-4-1-20250805, claude-sonnet-4-20250514, etc.
               Ollama: gpt-oss:20b, gpt-oss:120b (local models)

        provider: The LLM provider. Options:
                 - "openai" (default): OpenAI's GPT models
                 - "anthropic": Anthropic's Claude models via LiteLLM
                 - "ollama": Local models via Ollama

        ctx: MCP context (automatically injected)

    Returns:
        Dictionary containing:
        - success: Whether the agent completed successfully with verification
        - result: The agent's execution result with verification details
        - error: Error message if execution or verification failed
        - metadata: Comprehensive reliability and verification information
        - execution_time_seconds: Total time taken including all checks

    Reliability Features:
        - All file operations use verified_* tools with dual verification
        - Preflight checks ensure file system access before execution
        - Environment detection identifies potential restrictions
        - Comprehensive error handling with specific failure details
        - No phantom operations - either works verifiably or fails clearly

    Examples:
        >>> await prompt_nano_agent(
        ...     "Create a Python function that calculates fibonacci numbers"
        ... )
        {"success": True, "result": "✅ RELIABLE EXECUTION COMPLETED\n...✅ VERIFIED SUCCESS: Wrote and verified..."}

        >>> await prompt_nano_agent(
        ...     "Test file system access and create a simple file",
        ...     model="gpt-5"
        ... )
        {"success": True, "result": "✅ RELIABLE EXECUTION COMPLETED\n...VERIFICATION SUMMARY: All operations verified"}
    """
    try:
        # Report progress if context is available
        if ctx:
            await ctx.report_progress(0.1, 1.0, "Initializing reliable agent with preflight checks...")

        # Create and validate request
        request = PromptNanoAgentRequest(
            agentic_prompt=agentic_prompt,
            model=model,
            provider=provider
        )

        if ctx:
            await ctx.report_progress(0.2, 1.0, "Initializing REAL file system agent...")

        # Import and use the REAL agent executor (LangGraph + real file ops)
        from .real_agent import execute_real_nano_agent

        if ctx:
            await ctx.report_progress(0.4, 1.0, "Executing REAL agent with LangGraph and verified file operations...")

        # Execute using the REAL file system architecture (NO OpenAI SDK)
        response = await execute_real_nano_agent(request)

        if ctx:
            await ctx.report_progress(1.0, 1.0, "REAL agent execution completed")
            if response.success:
                tool_uses = response.metadata.get("tool_uses", 0)
                verifications = response.metadata.get("verifications", 0)
                await ctx.info(f"REAL agent completed successfully - {tool_uses} tools used, {verifications} verifications passed in {response.execution_time_seconds:.3f}s")
            else:
                await ctx.error(f"REAL agent failed: {response.error}")

        # Convert response to dictionary for MCP protocol
        return response.model_dump()

    except Exception as e:
        logger.error(f"Error in reliable prompt_nano_agent: {str(e)}", exc_info=True)

        if ctx:
            await ctx.error(f"Reliable execution failed: {str(e)}")

        # Return error response with reliability metadata
        error_response = PromptNanoAgentResponse(
            success=False,
            error=f"Reliable agent system error: {str(e)}",
            metadata={
                "error_type": type(e).__name__,
                "reliability_system": "error",
                "system_component": "reliable_agent_wrapper"
            }
        )
        return error_response.model_dump()


# Additional utility functions

async def get_agent_status() -> Dict[str, Any]:
    """
    Get the current status of the nano agent system with reliability information.

    This utility function provides comprehensive status including reliability system status.
    """
    try:
        # Import reliability status
        from .reliable_agent import get_reliability_status

        # Get base status
        base_status = {
            "status": "operational",
            "version": VERSION,
            "available_models": AVAILABLE_MODELS,
            "available_providers": list(AVAILABLE_MODELS.keys()),
            "tools_available": AVAILABLE_TOOLS,
            "agent_sdk": True,
            "agent_sdk_version": "0.2.5",  # From openai-agents package
            "reliability_enabled": True
        }

        # Add reliability status
        reliability_status = get_reliability_status()
        base_status.update(reliability_status)

        return base_status

    except Exception as e:
        return {
            "status": "error",
            "version": VERSION,
            "error": f"Status check failed: {str(e)}",
            "reliability_enabled": False
        }


async def get_reliability_report() -> Dict[str, Any]:
    """
    Get a comprehensive reliability report for the nano agent system.

    Returns detailed information about environment, preflight checks, and verification capabilities.
    """
    try:
        from .reliable_agent import get_reliability_status
        from .environment_detector import detect_environment
        from .preflight import run_preflight_checks

        # Gather comprehensive reliability information
        reliability_status = get_reliability_status()
        environment_info = detect_environment()
        preflight_results = run_preflight_checks()

        return {
            "timestamp": datetime.now().isoformat(),
            "reliability_system": reliability_status,
            "environment_analysis": environment_info,
            "preflight_checks": preflight_results,
            "overall_operational": (
                reliability_status.get("file_system_operational", False) and
                environment_info.get("risk_level") != "CRITICAL" and
                preflight_results.get("is_operational", False)
            ),
            "recommendations": _generate_reliability_recommendations(
                reliability_status, environment_info, preflight_results
            )
        }

    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": f"Reliability report generation failed: {str(e)}",
            "error_type": type(e).__name__
        }


def _generate_reliability_recommendations(reliability_status: Dict, env_info: Dict, preflight: Dict) -> List[str]:
    """Generate recommendations based on reliability analysis."""
    recommendations = []

    # Check critical issues
    if not reliability_status.get("file_system_operational", False):
        recommendations.append("CRITICAL: File system access is blocked - check container mounts and permissions")

    if env_info.get("risk_level") == "CRITICAL":
        recommendations.append("CRITICAL: Environment risk level is critical - review environment restrictions")

    if not preflight.get("is_operational", False):
        recommendations.append("WARNING: Preflight checks failed - review file system permissions")

    # Check warnings
    if env_info.get("execution_context", {}).get("containerized", False):
        recommendations.append("INFO: Running in container - ensure proper volume mounts for file operations")

    if env_info.get("execution_context", {}).get("sandboxed", False):
        recommendations.append("WARNING: Sandbox environment detected - file operations may be restricted")

    # Performance recommendations
    fs_access = env_info.get("file_system_access", {})
    write_success_rate = fs_access.get("write_success_rate", 0)
    if write_success_rate < 100 and write_success_rate > 0:
        recommendations.append(f"INFO: Limited file system access ({write_success_rate:.1f}% success rate)")

    if not recommendations:
        recommendations.append("✅ System is operating optimally for reliable file operations")

    return recommendations


def validate_model_provider_combination(model: str, provider: str) -> bool:
    """
    Validate that the model and provider combination is supported.
    
    Args:
        model: The model identifier
        provider: The provider name
        
    Returns:
        True if the combination is valid, False otherwise
    """
    return provider in AVAILABLE_MODELS and model in AVAILABLE_MODELS[provider]


# Export raw tools for direct use in CLI (these are the decorated versions)
from .nano_agent_tools import (
    read_file_raw as read_file,
    write_file_raw as write_file,
    list_directory_raw as list_directory,
    get_file_info_raw as get_file_info
)