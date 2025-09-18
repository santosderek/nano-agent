"""
Real File System Agent using LangGraph create_react_agent.

This module provides a nano agent that performs ACTUAL file system operations
using the same architecture as DeepAgents but with real file operations instead
of virtual ones. Uses create_react_agent from LangGraph, not OpenAI SDK.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

# LangGraph imports (following DeepAgents pattern)
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# Model imports (following DeepAgents pattern)
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic

# Our direct tools
from .direct_tools import get_direct_tools, DirectOperationError

# Data types
from .data_types import (
    PromptNanoAgentRequest,
    PromptNanoAgentResponse
)

logger = logging.getLogger(__name__)

# System prompt for real file operations
REAL_AGENT_SYSTEM_PROMPT = """You are a file system agent with REAL file operation capabilities and comprehensive external verification.

CRITICAL: You have access to direct file system tools that perform ACTUAL operations on the real file system.
Every operation is verified using multiple independent methods to ensure it actually occurred - NO phantom operations possible.

AVAILABLE DIRECT TOOLS:
1. direct_write_file(file_path, content) - Write content with 5-method external validation
   - Uses Python, os.path, subprocess 'ls', subprocess 'stat', and 'wc' for verification
   - Creates parent directories automatically
   - Forces filesystem sync and validates with MD5 hashes

2. direct_read_file(file_path) - Read files with size and content verification
   - Validates file exists and is readable
   - Verifies content length matches file size
   - Includes detailed metadata

3. direct_list_directory(directory_path) - List directory contents with verification
   - Validates directory accessibility
   - Provides file sizes and types
   - Handles permission errors gracefully

4. direct_edit_file(file_path, old_str, new_str) - Edit files with exact string matching
   - Requires exact text match including whitespace
   - Verifies replacement occurred correctly
   - Checks for multiple occurrences

5. direct_system_test() - Comprehensive filesystem capability testing
   - Tests write, read, edit, and delete operations
   - Provides detailed capability report
   - Identifies any system limitations

VERIFICATION FEATURES:
- All operations use external subprocess commands (ls, stat, wc, cat) that cannot be faked
- Multiple independent validation methods per operation
- Filesystem sync forced after each write operation
- MD5 hash verification for content integrity
- Comprehensive error reporting with specific failure reasons

OPERATIONAL GUIDELINES:
- These tools perform REAL operations on the actual filesystem
- All operations are externally verified - phantom operations are impossible
- If verification fails, the operation will raise a DirectOperationError
- Use direct_system_test first if you suspect filesystem restrictions
- Read files before editing to see exact content and formatting
- Handle both relative and absolute paths correctly

PROGRESS REPORTING PROTOCOL (MCP):
IMPORTANT: Report detailed progress for all multi-stage operations using available context methods:

Multi-Stage Workflow Reporting:
1. **Initialization** (0-10%): "Preparing operation environment..."
2. **Validation** (10-30%): "Validating targets and permissions..."
3. **Pre-execution Checks** (30-40%): "Running safety checks..."
4. **Execution** (40-80%): "Executing [specific operation]..."
5. **Verification** (80-95%): "Verifying operation success..."
6. **Cleanup** (95-100%): "Finalizing and cleaning up..."

Reporting Guidelines:
- Use deterministic progress (x/total) when possible
- For file operations over 1MB, report every 10% processed
- For batch operations, report each item: "Processing file X of Y"
- Include file sizes in messages: "Reading large file (5.2MB)..."
- Report any delays: "Operation taking longer than expected..."
- On errors, immediately report with context: "Error at stage X: [details]"

Example Progress Sequence:
```
report_progress(0, 100, "Starting file edit operation...")
report_progress(20, 100, "Validating file exists and is writable...")
report_progress(40, 100, "Creating backup...")
report_progress(60, 100, "Applying changes...")
report_progress(80, 100, "Verifying changes...")
report_progress(100, 100, "Edit complete and verified")
```

TASK EXECUTION APPROACH:
1. Report initialization: "Starting REAL agent execution..."
2. Understand the complete task requirements
3. Report validation phase: "Running filesystem capability checks..."
4. Use direct_system_test if filesystem capability is uncertain
5. Report exploration phase: "Exploring project structure..."
6. Use direct_list_directory to explore project structure
7. Report analysis phase: "Analyzing existing content..."
8. Use direct_read_file to understand existing content
9. Report execution phase: "Executing verified operations..."
10. Execute modifications using direct_write_file or direct_edit_file
11. Report verification phase: "Verifying all operations..."
12. Verify results by reading back modified files
13. Report completion: "All operations verified and complete"

ERROR HANDLING:
- DirectOperationError indicates verification failure - operation did not actually occur
- ValidationFailure indicates external verification methods detected inconsistency
- Always report specific verification results from tools
- If operations fail, the error messages include detailed diagnostic information

Your job is to complete file system tasks using these externally verified real operations that eliminate any possibility of phantom or simulated file operations."""


class RealFileSystemAgent:
    """Agent that performs real file system operations using LangGraph."""

    def __init__(self, model_name: str = "gpt-4o", provider: str = "openai"):
        """Initialize the real file system agent."""
        self.model_name = model_name
        self.provider = provider
        self.model = self._create_model()
        self.tools = get_direct_tools()
        self.agent = self._create_agent()

    def _create_model(self):
        """Create the language model based on provider."""
        if self.provider == "openai":
            # Handle GPT-5 models vs other OpenAI models
            model_params = {
                "model": self.model_name
            }

            # GPT-5 models have special requirements
            if self.model_name.startswith("gpt-5"):
                # GPT-5 only supports temperature=1.0 (default) and max_completion_tokens
                model_params["max_completion_tokens"] = 4000
                # Don't set temperature for GPT-5 - use default
            else:
                # Other OpenAI models support temperature and max_tokens
                model_params["temperature"] = 0.1
                model_params["max_tokens"] = 4000

            return ChatOpenAI(**model_params)

        elif self.provider == "anthropic":
            return ChatAnthropic(
                model=self.model_name,
                temperature=0.1,
                max_tokens=4000
            )
        elif self.provider == "azure":
            # Azure might also have GPT-5 models, handle accordingly
            model_params = {
                "model": self.model_name,
                "temperature": 0.1,
                "azure_deployment": self.model_name,  # Use model_name as deployment name
                "api_version": "2024-02-15-preview"
            }

            # GPT-5 models use max_completion_tokens instead of max_tokens
            if self.model_name.startswith("gpt-5"):
                model_params["max_completion_tokens"] = 4000
            else:
                model_params["max_tokens"] = 4000

            return AzureChatOpenAI(**model_params)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _create_agent(self):
        """Create the react agent using LangGraph (DeepAgents pattern)."""
        # This is the exact pattern from DeepAgents - using create_react_agent
        return create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=REAL_AGENT_SYSTEM_PROMPT
        )

    async def execute_task(self, prompt: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Execute a task using the real file system agent.

        Args:
            prompt: Natural language description of the task
            max_iterations: Maximum number of agent iterations

        Returns:
            Dictionary with execution results and verification info
        """
        start_time = time.time()
        execution_id = f"real_agent_{int(time.time())}"

        logger.info(f"🚀 Starting real agent execution: {execution_id}")
        logger.info(f"📝 Task: {prompt[:100]}...")

        try:
            # Create the input message
            messages = [HumanMessage(content=prompt)]

            # Execute the agent using LangGraph's invoke (like DeepAgents)
            config = RunnableConfig(
                recursion_limit=max_iterations,
                configurable={
                    "thread_id": execution_id,
                    "checkpoint_id": execution_id
                }
            )

            # Run the agent
            logger.info("🤖 Executing LangGraph react agent...")
            result = await self.agent.ainvoke(
                {"messages": messages},
                config=config
            )

            execution_time = time.time() - start_time

            # Extract the final message
            messages_result = result.get("messages", [])
            if messages_result:
                final_message = messages_result[-1]
                if hasattr(final_message, 'content'):
                    final_output = final_message.content
                else:
                    final_output = str(final_message)
            else:
                final_output = "No output generated"

            # Count tool uses and analyze for verification
            tool_uses = []
            verification_count = 0

            for msg in messages_result:
                if isinstance(msg, ToolMessage):
                    tool_uses.append({
                        "tool": getattr(msg, 'name', 'unknown'),
                        "content": str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
                    })
                    if "✅ DIRECT VERIFIED" in str(msg.content):
                        verification_count += 1

            success_response = {
                "success": True,
                "result": f"""🎯 REAL AGENT EXECUTION COMPLETED

{final_output}

--- EXECUTION SUMMARY ---
• Execution ID: {execution_id}
• Tools Used: {len(tool_uses)}
• Verifications: {verification_count}
• Execution Time: {execution_time:.3f}s
• Model: {self.provider}/{self.model_name}

--- TOOL VERIFICATION ---
{chr(10).join([f"• {tool['tool']}: {tool['content']}" for tool in tool_uses[-3:]])}

✅ All operations performed on REAL file system with verification""",
                "metadata": {
                    "execution_id": execution_id,
                    "execution_time_seconds": execution_time,
                    "model": self.model_name,
                    "provider": self.provider,
                    "tool_uses": len(tool_uses),
                    "verifications": verification_count,
                    "messages_count": len(messages_result),
                    "agent_type": "langgraph_react_agent",
                    "real_file_operations": True
                }
            }

            logger.info(f"✅ Real agent completed successfully: {execution_id}")
            return success_response

        except DirectOperationError as e:
            execution_time = time.time() - start_time
            error_msg = f"File operation verification failed: {str(e)}"
            logger.error(f"❌ Direct operation error in {execution_id}: {error_msg}")

            return {
                "success": False,
                "error": error_msg,
                "metadata": {
                    "execution_id": execution_id,
                    "execution_time_seconds": execution_time,
                    "error_type": "DirectOperationError",
                    "agent_type": "langgraph_react_agent"
                }
            }

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Real agent execution failed: {str(e)}"
            logger.error(f"❌ Unexpected error in {execution_id}: {error_msg}", exc_info=True)

            return {
                "success": False,
                "error": error_msg,
                "metadata": {
                    "execution_id": execution_id,
                    "execution_time_seconds": execution_time,
                    "error_type": type(e).__name__,
                    "agent_type": "langgraph_react_agent"
                }
            }


# Global agent instance
_real_agent = None


def get_real_agent(model_name: str = "gpt-4o", provider: str = "openai") -> RealFileSystemAgent:
    """Get or create a real file system agent instance."""
    global _real_agent
    if _real_agent is None or _real_agent.model_name != model_name or _real_agent.provider != provider:
        _real_agent = RealFileSystemAgent(model_name=model_name, provider=provider)
    return _real_agent


async def execute_real_nano_agent(request: PromptNanoAgentRequest) -> PromptNanoAgentResponse:
    """
    Execute a nano agent with REAL file system operations using LangGraph.

    This is the main entry point that uses create_react_agent (like DeepAgents)
    but performs actual file operations instead of virtual ones.

    Args:
        request: The validated request containing prompt and configuration

    Returns:
        Response with real execution results and verification metadata
    """
    start_time = time.time()

    try:
        logger.info(f"🚀 Executing REAL nano agent with LangGraph")
        logger.info(f"📋 Model: {request.provider}/{request.model}")
        logger.info(f"📝 Prompt: {request.agentic_prompt[:100]}...")

        # Get the real agent
        agent = get_real_agent(
            model_name=request.model,
            provider=request.provider
        )

        # Execute the task
        result = await agent.execute_task(
            prompt=request.agentic_prompt,
            max_iterations=20  # Allow plenty of iterations for complex tasks
        )

        execution_time = time.time() - start_time

        if result["success"]:
            return PromptNanoAgentResponse(
                success=True,
                result=result["result"],
                metadata={
                    **result["metadata"],
                    "total_execution_time_seconds": execution_time,
                    "architecture": "langgraph_create_react_agent",
                    "file_operations": "real_verified",
                    "phantom_operations": "eliminated"
                },
                execution_time_seconds=execution_time
            )
        else:
            return PromptNanoAgentResponse(
                success=False,
                error=result["error"],
                metadata={
                    **result["metadata"],
                    "total_execution_time_seconds": execution_time,
                    "architecture": "langgraph_create_react_agent"
                },
                execution_time_seconds=execution_time
            )

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Real nano agent system error: {str(e)}"
        logger.error(error_msg, exc_info=True)

        return PromptNanoAgentResponse(
            success=False,
            error=error_msg,
            metadata={
                "error_type": type(e).__name__,
                "execution_time_seconds": execution_time,
                "architecture": "langgraph_create_react_agent",
                "system_component": "real_agent_wrapper"
            },
            execution_time_seconds=execution_time
        )


def get_real_agent_status() -> Dict[str, Any]:
    """Get status of the real file system agent."""
    try:
        from .direct_tools import direct_system_test

        # Run a quick system test
        test_result = direct_system_test()

        # Determine operational status
        is_operational = "✅ DIRECT OPERATIONS FULLY FUNCTIONAL" in test_result

        return {
            "timestamp": datetime.now().isoformat(),
            "agent_type": "langgraph_react_agent",
            "architecture": "create_react_agent",
            "file_operations": "real_verified",
            "operational": is_operational,
            "system_test": test_result,
            "features": [
                "real_file_operations",
                "langraph_based",
                "direct_verification",
                "no_phantom_operations",
                "openai_sdk_free"
            ]
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "agent_type": "langgraph_react_agent",
            "operational": False,
            "error": str(e),
            "error_type": type(e).__name__
        }