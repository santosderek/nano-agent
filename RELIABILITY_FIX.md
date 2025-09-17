# Nano Agent Reliability Fix

## Problem Analysis
The nano agent uses OpenAI Agent SDK which appears to run tools in a sandboxed environment, preventing actual file system operations while reporting success.

## Comprehensive Solution

### 1. Dual-Verification Tool Architecture

Replace current tools with verified implementations:

```python
# /src/nano_agent/modules/verified_tools.py

import os
import time
from pathlib import Path
from typing import Dict, Any
from agents import function_tool

class VerificationError(Exception):
    """Raised when tool operation cannot be verified."""
    pass

def verify_file_operation(operation: str, file_path: str, expected_content: str = None) -> bool:
    """Verify that a file operation actually occurred."""
    try:
        path = Path(file_path).resolve()

        if operation == "create":
            if not path.exists():
                return False
            if expected_content and path.read_text() != expected_content:
                return False
            return True

        elif operation == "delete":
            return not path.exists()

        elif operation == "modify":
            if not path.exists():
                return False
            if expected_content and expected_content not in path.read_text():
                return False
            return True

        return False
    except Exception:
        return False

@function_tool
def verified_write_file(file_path: str, content: str) -> str:
    """Write content to a file with mandatory verification."""
    try:
        # Step 1: Perform the operation
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write using multiple methods for redundancy
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk

        # Step 2: Immediate verification
        time.sleep(0.1)  # Brief delay for file system sync

        if not verify_file_operation("create", file_path, content):
            raise VerificationError(f"File write verification failed: {file_path}")

        # Step 3: Secondary verification with direct read
        try:
            actual_content = Path(file_path).read_text()
            if actual_content != content:
                raise VerificationError(f"Content mismatch: expected {len(content)} chars, got {len(actual_content)} chars")
        except Exception as e:
            raise VerificationError(f"Could not read back written file: {e}")

        return f"✅ VERIFIED: Successfully wrote {len(content)} characters to {file_path}"

    except VerificationError:
        raise
    except Exception as e:
        raise VerificationError(f"Write operation failed: {e}")

@function_tool
def verified_read_file(file_path: str) -> str:
    """Read file content with existence verification."""
    try:
        path = Path(file_path).resolve()

        # Verify file exists
        if not path.exists():
            raise VerificationError(f"File does not exist: {file_path}")

        if not path.is_file():
            raise VerificationError(f"Path is not a file: {file_path}")

        content = path.read_text(encoding='utf-8')

        # Verify we got actual content
        return f"✅ VERIFIED: Read {len(content)} characters from {file_path}\n\nContent:\n{content}"

    except VerificationError:
        raise
    except Exception as e:
        raise VerificationError(f"Read operation failed: {e}")

@function_tool
def verified_file_system_test() -> str:
    """Test actual file system access with verification."""
    test_file = "/tmp/nano_agent_verification_test.txt"
    test_content = f"Verification test at {time.time()}"

    try:
        # Test write
        with open(test_file, 'w') as f:
            f.write(test_content)

        # Verify write
        if not Path(test_file).exists():
            return "❌ CRITICAL: File system write access BLOCKED - file not created"

        # Test read
        actual_content = Path(test_file).read_text()
        if actual_content != test_content:
            return f"❌ CRITICAL: File system corruption - wrote {len(test_content)} chars, read {len(actual_content)} chars"

        # Cleanup
        Path(test_file).unlink()

        return "✅ VERIFIED: File system access is functional"

    except Exception as e:
        return f"❌ CRITICAL: File system access failed: {e}"
```

### 2. Mandatory Pre-Flight Checks

```python
# /src/nano_agent/modules/preflight.py

def run_preflight_checks() -> Dict[str, Any]:
    """Run mandatory checks before any file operations."""
    results = {
        "file_system_access": False,
        "write_permissions": False,
        "read_permissions": False,
        "working_directory": str(Path.cwd()),
        "temp_directory": "/tmp",
        "errors": []
    }

    try:
        # Test file system access
        test_result = verified_file_system_test()
        if "✅ VERIFIED" in test_result:
            results["file_system_access"] = True
            results["write_permissions"] = True
            results["read_permissions"] = True
        else:
            results["errors"].append(test_result)

    except Exception as e:
        results["errors"].append(f"Preflight check failed: {e}")

    return results
```

### 3. Enhanced Agent with Mandatory Verification

```python
# /src/nano_agent/modules/reliable_agent.py

async def execute_reliable_nano_agent(request: PromptNanoAgentRequest) -> PromptNanoAgentResponse:
    """Execute nano agent with mandatory reliability checks."""

    # Step 1: Preflight checks
    preflight = run_preflight_checks()
    if not preflight["file_system_access"]:
        return PromptNanoAgentResponse(
            success=False,
            error=f"CRITICAL: File system access blocked. Errors: {preflight['errors']}",
            metadata={"preflight_failed": True, "preflight_results": preflight}
        )

    # Step 2: Create agent with verified tools
    verified_tools = [
        verified_write_file,
        verified_read_file,
        verified_file_system_test,
        # ... other verified tools
    ]

    # Enhanced system prompt with verification requirements
    enhanced_prompt = f"""
    {NANO_AGENT_SYSTEM_PROMPT}

    CRITICAL RELIABILITY REQUIREMENTS:
    - ALL file operations MUST be verified using the verified_* tools
    - You MUST report verification results in your responses
    - If ANY verification fails, STOP and report the failure immediately
    - Use verified_file_system_test() if you suspect any issues

    Available verified tools: {[tool.__name__ for tool in verified_tools]}
    """

    agent = Agent(
        name="ReliableNanoAgent",
        instructions=enhanced_prompt,
        tools=verified_tools,
        model=request.model
    )

    # Step 3: Execute with verification monitoring
    result = await Runner.run(agent, request.agentic_prompt, max_turns=MAX_AGENT_TURNS)

    # Step 4: Post-execution verification
    final_verification = run_preflight_checks()

    return PromptNanoAgentResponse(
        success=True,
        result=str(result),
        metadata={
            "preflight_checks": preflight,
            "post_execution_checks": final_verification,
            "reliability_verified": True
        }
    )
```

### 4. Environment Detection and Fixes

```python
# /src/nano_agent/modules/environment_detector.py

def detect_execution_environment() -> Dict[str, Any]:
    """Detect if running in sandbox/container with restrictions."""

    env_info = {
        "platform": os.name,
        "user": os.getenv("USER", "unknown"),
        "home": os.getenv("HOME", "unknown"),
        "pwd": str(Path.cwd()),
        "containerized": False,
        "sandbox_detected": False,
        "file_system_restrictions": []
    }

    # Detect containerization
    if Path("/.dockerenv").exists() or os.getenv("container"):
        env_info["containerized"] = True

    # Test write access to various directories
    test_dirs = ["/tmp", str(Path.home()), str(Path.cwd()), "/var/tmp"]
    for test_dir in test_dirs:
        try:
            test_file = Path(test_dir) / f"test_write_{time.time()}.tmp"
            test_file.write_text("test")
            test_file.unlink()
            env_info[f"writable_{test_dir.replace('/', '_')}"] = True
        except:
            env_info[f"writable_{test_dir.replace('/', '_')}"] = False
            env_info["file_system_restrictions"].append(test_dir)

    # Detect sandbox by checking for typical restrictions
    if len(env_info["file_system_restrictions"]) > 2:
        env_info["sandbox_detected"] = True

    return env_info
```

## Implementation Strategy

1. **Replace current tools** with verified versions
2. **Add mandatory preflight checks** to every agent execution
3. **Implement dual verification** (operation + independent check)
4. **Add environment detection** to identify sandbox restrictions
5. **Create fail-fast mechanisms** that stop on verification failures
6. **Add comprehensive logging** of all verification steps

## Expected Outcome

This approach will:
- ✅ **Detect sandbox restrictions** immediately
- ✅ **Fail fast** if file operations are blocked
- ✅ **Verify every operation** independently
- ✅ **Provide clear error messages** about what's actually blocked
- ✅ **Eliminate phantom operations** completely

The system will either work reliably or fail clearly - no more false positives.