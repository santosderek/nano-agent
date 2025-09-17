"""
Reliable Agent Execution Module for Nano Agent.

This module provides a robust agent execution system with mandatory
verification, preflight checks, and comprehensive error handling
to eliminate phantom operations and ensure actual file system changes.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

# OpenAI Agent SDK imports
from agents import Agent, Runner, RunConfig, ModelSettings
from agents.lifecycle import RunHooksBase

# Import our reliability components
from .verified_tools import get_verified_nano_agent_tools, VerificationError, FileSystemAccessError
from .preflight import run_preflight_checks, is_file_system_operational
from .environment_detector import detect_environment, is_environment_safe_for_file_operations
from .provider_config import ProviderConfig

# Import data types and constants
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
    AVAILABLE_MODELS,
    PROVIDER_REQUIREMENTS,
    SUCCESS_AGENT_COMPLETE,
    VERSION
)

# Token tracking
from .token_tracking import TokenTracker, format_token_count, format_cost

logger = logging.getLogger(__name__)

# Enhanced system prompt for reliable operations
RELIABLE_NANO_AGENT_SYSTEM_PROMPT = """You are a highly reliable file system agent with mandatory verification capabilities.

CRITICAL RELIABILITY REQUIREMENTS:
- You MUST use ONLY the verified_* tools for ALL file operations
- Every file operation MUST be verified before reporting success
- If ANY verification fails, STOP immediately and report the failure
- Use verified_file_system_test() if you suspect any access issues
- Always provide detailed verification results in your responses

AVAILABLE VERIFIED TOOLS:
- verified_write_file(file_path, content): Write with mandatory verification
- verified_read_file(file_path): Read with existence verification
- verified_list_directory(directory_path): List with access verification
- verified_edit_file(file_path, old_str, new_str): Edit with content verification
- verified_file_system_test(): Test file system access capabilities

VERIFICATION PROTOCOL:
1. Each tool performs the operation AND verifies it occurred
2. Tools return "✅ VERIFIED SUCCESS" messages with details
3. If verification fails, tools raise VerificationError with specific details
4. You must report verification status in all responses

FAILURE HANDLING:
- If any tool raises VerificationError, STOP and report the exact error
- Do not attempt workarounds for verification failures
- Do not continue with additional operations if verification fails
- Clearly state what was verified vs what failed

Your job is to complete tasks reliably with full verification, or fail clearly with specific error details."""


class ReliabilityError(Exception):
    """Raised when reliability checks fail."""
    pass


class ReliableAgentHooks(RunHooksBase):
    """Custom lifecycle hooks for reliable agent execution with verification monitoring."""

    def __init__(self, token_tracker: Optional[TokenTracker] = None):
        """Initialize reliability hooks with token tracking."""
        self.token_tracker = token_tracker
        self.verification_failures = []
        self.verification_successes = []
        self.tool_call_count = 0

    async def on_agent_start(self, context, agent):
        """Called when the agent starts."""
        logger.info(f"🚀 Reliable agent started: {agent.name}")

        # Track initial token usage if available
        if self.token_tracker and hasattr(context, 'usage'):
            self.token_tracker.update(context.usage)

    async def on_tool_start(self, context, agent, tool):
        """Called before a tool is invoked."""
        self.tool_call_count += 1
        tool_name = getattr(tool, 'name', 'Unknown Tool')

        logger.info(f"🔧 Tool #{self.tool_call_count}: {tool_name}")

        # Ensure only verified tools are being used
        if not tool_name.startswith('verified_'):
            logger.error(f"❌ NON-VERIFIED TOOL DETECTED: {tool_name}")
            raise ReliabilityError(f"Attempted to use non-verified tool: {tool_name}")

    async def on_tool_end(self, context, agent, tool, result):
        """Called after a tool is invoked."""
        tool_name = getattr(tool, 'name', 'Unknown Tool')
        result_str = str(result)

        # Check if verification was successful
        if "✅ VERIFIED SUCCESS" in result_str:
            self.verification_successes.append({
                "tool": tool_name,
                "result": result_str[:200] + "..." if len(result_str) > 200 else result_str
            })
            logger.info(f"✅ Tool #{self.tool_call_count} verification passed: {tool_name}")
        elif "VerificationError" in result_str or "CRITICAL:" in result_str:
            self.verification_failures.append({
                "tool": tool_name,
                "error": result_str[:500] + "..." if len(result_str) > 500 else result_str
            })
            logger.error(f"❌ Tool #{self.tool_call_count} verification failed: {tool_name}")
        else:
            # Unexpected result format
            logger.warning(f"⚠️ Tool #{self.tool_call_count} unexpected result format: {tool_name}")

    async def on_agent_end(self, context, agent, output):
        """Called when the agent produces final output."""
        # Track final token usage
        if self.token_tracker and hasattr(context, 'usage'):
            self.token_tracker.update(context.usage)

        # Log verification summary
        total_verifications = len(self.verification_successes) + len(self.verification_failures)
        success_rate = (len(self.verification_successes) / total_verifications * 100) if total_verifications > 0 else 100

        if self.verification_failures:
            logger.error(f"🚨 Agent completed with {len(self.verification_failures)} verification failures")
            for failure in self.verification_failures:
                logger.error(f"  - {failure['tool']}: {failure['error'][:100]}...")
        else:
            logger.info(f"✅ Agent completed successfully - all {len(self.verification_successes)} verifications passed")


class ReliableAgentExecutor:
    """Executes nano agents with comprehensive reliability guarantees."""

    def __init__(self):
        self.execution_history = []

    async def execute_reliable_agent(self, request: PromptNanoAgentRequest) -> PromptNanoAgentResponse:
        """
        Execute nano agent with comprehensive reliability checks.

        Args:
            request: The validated request containing prompt and configuration

        Returns:
            Response with execution results, verification status, and reliability metadata
        """
        start_time = time.time()
        execution_id = f"exec_{int(time.time())}_{id(request)}"

        logger.info(f"🚀 Starting reliable agent execution: {execution_id}")

        try:
            # Phase 1: Environment Detection
            logger.info("📊 Phase 1: Environment Detection")
            env_info = detect_environment()

            if env_info["risk_level"] == "CRITICAL":
                return PromptNanoAgentResponse(
                    success=False,
                    error=f"CRITICAL: Environment is not safe for file operations. Risk Level: {env_info['risk_level']}. Summary: {env_info['summary']}",
                    metadata={
                        "phase_failed": "environment_detection",
                        "environment_info": env_info,
                        "execution_id": execution_id
                    },
                    execution_time_seconds=time.time() - start_time
                )

            logger.info(f"✅ Environment check passed - Risk Level: {env_info['risk_level']}")

            # Phase 2: Preflight Checks
            logger.info("🔍 Phase 2: Preflight Checks")
            preflight_results = run_preflight_checks()

            if not preflight_results["is_operational"]:
                return PromptNanoAgentResponse(
                    success=False,
                    error=f"CRITICAL: Preflight checks failed. {preflight_results['checks_failed']} of {preflight_results['checks_total']} checks failed. Critical errors: {'; '.join(preflight_results['critical_errors'])}",
                    metadata={
                        "phase_failed": "preflight_checks",
                        "preflight_results": preflight_results,
                        "environment_info": env_info,
                        "execution_id": execution_id
                    },
                    execution_time_seconds=time.time() - start_time
                )

            logger.info(f"✅ Preflight checks passed - {preflight_results['checks_passed']}/{preflight_results['checks_total']} checks successful")

            # Phase 3: Provider Validation
            logger.info("🔧 Phase 3: Provider Validation")
            is_valid, error_msg = ProviderConfig.validate_provider_setup(
                request.provider,
                request.model,
                AVAILABLE_MODELS,
                PROVIDER_REQUIREMENTS
            )

            if not is_valid:
                return PromptNanoAgentResponse(
                    success=False,
                    error=f"Provider validation failed: {error_msg}",
                    metadata={
                        "phase_failed": "provider_validation",
                        "model": request.model,
                        "provider": request.provider,
                        "execution_id": execution_id
                    },
                    execution_time_seconds=time.time() - start_time
                )

            # Setup provider-specific configurations
            ProviderConfig.setup_provider(request.provider)
            logger.info(f"✅ Provider validated: {request.provider}/{request.model}")

            # Phase 4: Agent Creation and Execution
            logger.info("🤖 Phase 4: Agent Creation and Execution")

            # Get verified tools
            verified_tools = get_verified_nano_agent_tools()
            logger.info(f"📦 Loaded {len(verified_tools)} verified tools")

            # Configure model settings
            base_settings = {
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": MAX_TOKENS
            }

            model_settings = ProviderConfig.get_model_settings(
                model=request.model,
                provider=request.provider,
                base_settings=base_settings
            )

            # Create reliable agent
            agent = ProviderConfig.create_agent(
                name="ReliableNanoAgent",
                instructions=RELIABLE_NANO_AGENT_SYSTEM_PROMPT,
                tools=verified_tools,
                model=request.model,
                provider=request.provider,
                model_settings=model_settings
            )

            # Create token tracker and reliability hooks
            token_tracker = TokenTracker(model=request.model, provider=request.provider)
            hooks = ReliableAgentHooks(token_tracker=token_tracker)

            # Execute the agent
            logger.info(f"🎯 Executing agent with prompt: {request.agentic_prompt[:100]}...")

            result = await Runner.run(
                agent,
                request.agentic_prompt,
                max_turns=MAX_AGENT_TURNS,
                run_config=RunConfig(
                    workflow_name="reliable_nano_agent_task",
                    trace_metadata={
                        "model": request.model,
                        "provider": request.provider,
                        "timestamp": datetime.now().isoformat(),
                        "execution_id": execution_id,
                        "reliability_mode": True
                    }
                ),
                hooks=hooks
            )

            execution_time = time.time() - start_time

            # Phase 5: Result Validation and Verification Summary
            logger.info("✅ Phase 5: Result Validation")

            # Extract final output
            final_output = result.final_output if hasattr(result, 'final_output') else str(result)

            # Check for verification failures
            if hooks.verification_failures:
                logger.error(f"❌ Agent execution completed but with {len(hooks.verification_failures)} verification failures")

                failure_summary = "; ".join([f"{f['tool']}: {f['error'][:100]}" for f in hooks.verification_failures])

                return PromptNanoAgentResponse(
                    success=False,
                    error=f"Agent execution failed verification checks: {failure_summary}",
                    metadata={
                        "phase_failed": "verification_checks",
                        "verification_failures": hooks.verification_failures,
                        "verification_successes": hooks.verification_successes,
                        "environment_info": env_info,
                        "preflight_results": preflight_results,
                        "execution_id": execution_id,
                        "agent_turns": len(result.messages) if hasattr(result, 'messages') else 0
                    },
                    execution_time_seconds=execution_time
                )

            # Build comprehensive metadata
            metadata = {
                "model": request.model,
                "provider": request.provider,
                "timestamp": datetime.now().isoformat(),
                "execution_id": execution_id,
                "reliability_verified": True,
                "environment_info": {
                    "risk_level": env_info["risk_level"],
                    "summary": env_info["summary"],
                    "platform": env_info["platform_info"].get("system", "unknown")
                },
                "preflight_results": {
                    "checks_passed": preflight_results["checks_passed"],
                    "checks_total": preflight_results["checks_total"],
                    "success_rate": preflight_results["success_rate"]
                },
                "verification_summary": {
                    "total_tools_used": hooks.tool_call_count,
                    "verifications_passed": len(hooks.verification_successes),
                    "verifications_failed": len(hooks.verification_failures),
                    "verification_success_rate": (len(hooks.verification_successes) / hooks.tool_call_count * 100) if hooks.tool_call_count > 0 else 100
                },
                "agent_turns": len(result.messages) if hasattr(result, 'messages') else 0
            }

            # Add token usage information
            if token_tracker:
                report = token_tracker.generate_report()
                metadata["token_usage"] = {
                    "total_tokens": report.total_tokens,
                    "input_tokens": report.total_input_tokens,
                    "output_tokens": report.total_output_tokens,
                    "cached_tokens": report.cached_input_tokens,
                    "total_cost": round(report.total_cost, 4)
                }

            # Format final response
            verified_response = f"""✅ RELIABLE EXECUTION COMPLETED

{final_output}

VERIFICATION SUMMARY:
- Total Tools Used: {hooks.tool_call_count}
- Verifications Passed: {len(hooks.verification_successes)}
- Verification Success Rate: {metadata['verification_summary']['verification_success_rate']:.1f}%
- Execution Time: {execution_time:.3f}s
- Environment Risk Level: {env_info['risk_level']}

All file operations have been verified and confirmed successful."""

            logger.info(f"✅ Reliable agent execution completed successfully: {execution_id}")

            return PromptNanoAgentResponse(
                success=True,
                result=verified_response,
                metadata=metadata,
                execution_time_seconds=execution_time
            )

        except VerificationError as e:
            execution_time = time.time() - start_time
            error_msg = f"Verification Error: {str(e)}"
            logger.error(f"❌ Verification error in {execution_id}: {error_msg}")

            return PromptNanoAgentResponse(
                success=False,
                error=error_msg,
                metadata={
                    "error_type": "VerificationError",
                    "execution_id": execution_id,
                    "phase_failed": "agent_execution"
                },
                execution_time_seconds=execution_time
            )

        except FileSystemAccessError as e:
            execution_time = time.time() - start_time
            error_msg = f"File System Access Error: {str(e)}"
            logger.error(f"❌ File system access error in {execution_id}: {error_msg}")

            return PromptNanoAgentResponse(
                success=False,
                error=error_msg,
                metadata={
                    "error_type": "FileSystemAccessError",
                    "execution_id": execution_id,
                    "phase_failed": "agent_execution"
                },
                execution_time_seconds=execution_time
            )

        except ReliabilityError as e:
            execution_time = time.time() - start_time
            error_msg = f"Reliability Error: {str(e)}"
            logger.error(f"❌ Reliability error in {execution_id}: {error_msg}")

            return PromptNanoAgentResponse(
                success=False,
                error=error_msg,
                metadata={
                    "error_type": "ReliabilityError",
                    "execution_id": execution_id,
                    "phase_failed": "agent_execution"
                },
                execution_time_seconds=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error during reliable agent execution: {str(e)}"
            logger.error(f"❌ Unexpected error in {execution_id}: {error_msg}", exc_info=True)

            return PromptNanoAgentResponse(
                success=False,
                error=error_msg,
                metadata={
                    "error_type": type(e).__name__,
                    "execution_id": execution_id,
                    "phase_failed": "unknown"
                },
                execution_time_seconds=execution_time
            )


# Global reliable executor instance
_reliable_executor = ReliableAgentExecutor()


async def execute_reliable_nano_agent(request: PromptNanoAgentRequest) -> PromptNanoAgentResponse:
    """
    Execute a nano agent with comprehensive reliability guarantees.

    This function provides the main entry point for reliable agent execution
    with full verification, preflight checks, and error handling.

    Args:
        request: The validated request containing prompt and configuration

    Returns:
        Response with execution results and comprehensive reliability metadata
    """
    return await _reliable_executor.execute_reliable_agent(request)


def get_reliability_status() -> Dict[str, Any]:
    """
    Get the current status of the reliability system.

    Returns:
        Dictionary with reliability system status information
    """
    try:
        # Run quick checks
        env_safe = is_environment_safe_for_file_operations()
        fs_operational, critical_errors = is_file_system_operational()

        return {
            "timestamp": datetime.now().isoformat(),
            "reliability_system": "operational",
            "version": VERSION,
            "environment_safe": env_safe,
            "file_system_operational": fs_operational,
            "critical_errors": critical_errors,
            "verified_tools_available": len(get_verified_nano_agent_tools()),
            "reliability_features": [
                "dual_verification",
                "preflight_checks",
                "environment_detection",
                "fail_fast_mechanisms",
                "comprehensive_error_handling"
            ]
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "reliability_system": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }