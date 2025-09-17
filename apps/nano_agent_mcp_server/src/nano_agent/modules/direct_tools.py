"""
Direct File System Tools - NO SDK DEPENDENCIES.

This module provides real file system operations using native Python
with LangChain tool decorators, following the DeepAgents pattern but
for ACTUAL file operations instead of virtual ones.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Union, Any
from datetime import datetime
import logging

# LangChain imports (NOT OpenAI SDK)
from langchain_core.tools import tool

# Our verification utilities
from .files import resolve_path, ensure_parent_exists, format_path_for_display

logger = logging.getLogger(__name__)

# Verification constants
VERIFICATION_RETRIES = 3
SYNC_DELAY = 0.1


class DirectOperationError(Exception):
    """Raised when direct file operations fail verification."""
    pass


def _force_sync_filesystem():
    """Force filesystem synchronization using multiple methods."""
    try:
        # Method 1: Python os.sync()
        if hasattr(os, 'sync'):
            os.sync()
    except:
        pass

    try:
        # Method 2: subprocess sync command
        subprocess.run(['sync'], check=False, capture_output=True, timeout=2)
    except:
        pass

    # Always include a delay
    time.sleep(SYNC_DELAY)


def _verify_operation_with_retries(verification_func, *args, **kwargs) -> bool:
    """Retry verification with filesystem sync between attempts."""
    for attempt in range(VERIFICATION_RETRIES):
        _force_sync_filesystem()

        if verification_func(*args, **kwargs):
            return True

        if attempt < VERIFICATION_RETRIES - 1:
            logger.debug(f"Verification attempt {attempt + 1} failed, retrying...")
            time.sleep(SYNC_DELAY * (attempt + 1))

    return False


@tool
def direct_write_file(file_path: str, content: str) -> str:
    """
    Write content to a file using direct filesystem operations.

    This tool performs REAL file operations with comprehensive verification.
    It will either succeed with verification or fail with specific errors.

    Args:
        file_path: Path to write the file (relative or absolute)
        content: Content to write to the file

    Returns:
        Success message with verification details

    Raises:
        DirectOperationError: If operation cannot be verified
    """
    start_time = time.time()

    try:
        # Resolve path
        path = resolve_path(file_path)
        display_path = format_path_for_display(path)

        logger.info(f"🔧 DIRECT WRITE: {display_path}")

        # Ensure parent directory exists
        ensure_parent_exists(path)

        # Write with multiple sync methods
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            # Force write to disk
            os.fsync(f.fileno())

        # Immediate verification with retries
        def verify_exists():
            return path.exists() and path.is_file()

        def verify_content():
            try:
                actual = path.read_text(encoding='utf-8')
                return actual == content
            except:
                return False

        # Verify file exists
        if not _verify_operation_with_retries(verify_exists):
            raise DirectOperationError(f"CRITICAL: File was not created: {file_path}")

        # Verify content matches
        if not _verify_operation_with_retries(verify_content):
            raise DirectOperationError(f"CRITICAL: Content verification failed: {file_path}")

        # Final integrity check
        actual_size = path.stat().st_size
        expected_size = len(content.encode('utf-8'))

        if actual_size != expected_size:
            raise DirectOperationError(
                f"CRITICAL: Size mismatch - expected {expected_size} bytes, got {actual_size} bytes"
            )

        execution_time = time.time() - start_time

        success_msg = (
            f"✅ DIRECT VERIFIED: Wrote {len(content)} chars to {display_path} "
            f"(size: {actual_size} bytes, time: {execution_time:.3f}s)"
        )

        logger.info(success_msg)
        return success_msg

    except DirectOperationError:
        raise
    except Exception as e:
        error_msg = f"Direct write failed: {str(e)}"
        logger.error(error_msg)
        raise DirectOperationError(error_msg)


@tool
def direct_read_file(file_path: str) -> str:
    """
    Read file content using direct filesystem operations.

    Args:
        file_path: Path to read (relative or absolute)

    Returns:
        File content with verification details

    Raises:
        DirectOperationError: If file cannot be read or verified
    """
    start_time = time.time()

    try:
        path = resolve_path(file_path)
        display_path = format_path_for_display(path)

        logger.info(f"🔧 DIRECT READ: {display_path}")

        # Verify file exists with retries
        def verify_readable():
            return path.exists() and path.is_file()

        if not _verify_operation_with_retries(verify_readable):
            raise DirectOperationError(f"File not found or not readable: {file_path}")

        # Read content
        content = path.read_text(encoding='utf-8')

        # Verify we got content for non-empty files
        file_size = path.stat().st_size
        if file_size > 0 and len(content) == 0:
            raise DirectOperationError(f"File size {file_size} bytes but read 0 characters")

        execution_time = time.time() - start_time

        result = (
            f"✅ DIRECT VERIFIED: Read {len(content)} chars from {display_path} "
            f"(size: {file_size} bytes, time: {execution_time:.3f}s)\n\n"
            f"--- CONTENT ---\n{content}"
        )

        logger.info(f"Successfully read {len(content)} chars from {display_path}")
        return result

    except DirectOperationError:
        raise
    except Exception as e:
        error_msg = f"Direct read failed: {str(e)}"
        logger.error(error_msg)
        raise DirectOperationError(error_msg)


@tool
def direct_list_directory(directory_path: Optional[str] = None) -> str:
    """
    List directory contents using direct filesystem operations.

    Args:
        directory_path: Directory to list (default: current directory)

    Returns:
        Directory listing with verification

    Raises:
        DirectOperationError: If directory cannot be accessed
    """
    start_time = time.time()

    try:
        if directory_path is None:
            path = Path.cwd()
            display_path = str(path)
        else:
            path = resolve_path(directory_path)
            display_path = format_path_for_display(path)

        logger.info(f"🔧 DIRECT LIST: {display_path}")

        # Verify directory exists and is accessible
        def verify_directory():
            return path.exists() and path.is_dir()

        if not _verify_operation_with_retries(verify_directory):
            raise DirectOperationError(f"Directory not found or not accessible: {directory_path}")

        # List contents
        try:
            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"[DIR]  {item.name}/")
                else:
                    try:
                        size = item.stat().st_size
                        items.append(f"[FILE] {item.name} ({size} bytes)")
                    except:
                        items.append(f"[FILE] {item.name} (size unknown)")
        except PermissionError:
            raise DirectOperationError(f"Permission denied accessing directory: {directory_path}")
        except Exception as e:
            raise DirectOperationError(f"Failed to list directory: {str(e)}")

        execution_time = time.time() - start_time

        result = (
            f"✅ DIRECT VERIFIED: Listed {display_path} "
            f"({len(items)} items, time: {execution_time:.3f}s)\n\n"
            f"--- CONTENTS ---\n"
        )

        if items:
            result += "\n".join(items)
        else:
            result += "Directory is empty"

        logger.info(f"Successfully listed {len(items)} items in {display_path}")
        return result

    except DirectOperationError:
        raise
    except Exception as e:
        error_msg = f"Direct directory listing failed: {str(e)}"
        logger.error(error_msg)
        raise DirectOperationError(error_msg)


@tool
def direct_edit_file(file_path: str, old_str: str, new_str: str) -> str:
    """
    Edit file by replacing text using direct filesystem operations.

    Args:
        file_path: Path to file to edit
        old_str: Text to find and replace (must match exactly)
        new_str: Replacement text

    Returns:
        Success message with verification

    Raises:
        DirectOperationError: If edit cannot be verified
    """
    start_time = time.time()

    try:
        path = resolve_path(file_path)
        display_path = format_path_for_display(path)

        logger.info(f"🔧 DIRECT EDIT: {display_path}")

        # Verify file exists
        if not path.exists() or not path.is_file():
            raise DirectOperationError(f"File not found: {file_path}")

        # Read current content
        original_content = path.read_text(encoding='utf-8')

        # Check if old_str exists
        if old_str not in original_content:
            raise DirectOperationError(f"Text not found in file: {old_str[:100]}...")

        # Check for multiple occurrences
        occurrences = original_content.count(old_str)
        if occurrences > 1:
            raise DirectOperationError(
                f"Found {occurrences} occurrences. Please provide more specific text."
            )

        # Perform replacement
        new_content = original_content.replace(old_str, new_str, 1)

        # Write new content with sync
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            f.flush()
            os.fsync(f.fileno())

        # Verify edit was successful
        def verify_edit():
            try:
                actual = path.read_text(encoding='utf-8')
                return actual == new_content and new_str in actual and old_str not in actual
            except:
                return False

        if not _verify_operation_with_retries(verify_edit):
            raise DirectOperationError(f"Edit verification failed for {file_path}")

        execution_time = time.time() - start_time

        success_msg = (
            f"✅ DIRECT VERIFIED: Edited {display_path} - "
            f"replaced {len(old_str)} chars with {len(new_str)} chars "
            f"(time: {execution_time:.3f}s)"
        )

        logger.info(success_msg)
        return success_msg

    except DirectOperationError:
        raise
    except Exception as e:
        error_msg = f"Direct edit failed: {str(e)}"
        logger.error(error_msg)
        raise DirectOperationError(error_msg)


@tool
def direct_system_test() -> str:
    """
    Test direct filesystem operations with comprehensive verification.

    Returns:
        Detailed test report showing actual filesystem capabilities
    """
    start_time = time.time()
    test_file = f"/tmp/direct_test_{int(time.time())}_{os.getpid()}.txt"
    test_content = f"Direct operation test at {datetime.now().isoformat()}"

    results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "errors": []
    }

    try:
        logger.info("🔧 DIRECT SYSTEM TEST: Starting comprehensive verification")

        # Test 1: Write
        results["tests_run"] += 1
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
                f.flush()
                os.fsync(f.fileno())

            _force_sync_filesystem()

            if Path(test_file).exists():
                results["tests_passed"] += 1
                logger.info("✅ Write test passed")
            else:
                results["tests_failed"] += 1
                results["errors"].append("Write test: file not created")
                logger.error("❌ Write test failed")
        except Exception as e:
            results["tests_failed"] += 1
            results["errors"].append(f"Write test: {str(e)}")
            logger.error(f"❌ Write test failed: {e}")

        # Test 2: Read
        results["tests_run"] += 1
        try:
            if Path(test_file).exists():
                actual_content = Path(test_file).read_text(encoding='utf-8')
                if actual_content == test_content:
                    results["tests_passed"] += 1
                    logger.info("✅ Read test passed")
                else:
                    results["tests_failed"] += 1
                    results["errors"].append(f"Read test: content mismatch")
                    logger.error("❌ Read test failed")
            else:
                results["tests_failed"] += 1
                results["errors"].append("Read test: file doesn't exist")
        except Exception as e:
            results["tests_failed"] += 1
            results["errors"].append(f"Read test: {str(e)}")
            logger.error(f"❌ Read test failed: {e}")

        # Test 3: Edit
        results["tests_run"] += 1
        try:
            if Path(test_file).exists():
                modified_content = test_content + "\nMODIFIED"
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                    f.flush()
                    os.fsync(f.fileno())

                _force_sync_filesystem()

                actual = Path(test_file).read_text(encoding='utf-8')
                if actual == modified_content:
                    results["tests_passed"] += 1
                    logger.info("✅ Edit test passed")
                else:
                    results["tests_failed"] += 1
                    results["errors"].append("Edit test: verification failed")
                    logger.error("❌ Edit test failed")
            else:
                results["tests_failed"] += 1
                results["errors"].append("Edit test: file doesn't exist")
        except Exception as e:
            results["tests_failed"] += 1
            results["errors"].append(f"Edit test: {str(e)}")
            logger.error(f"❌ Edit test failed: {e}")

        # Test 4: Delete
        results["tests_run"] += 1
        try:
            if Path(test_file).exists():
                Path(test_file).unlink()
                _force_sync_filesystem()

                if not Path(test_file).exists():
                    results["tests_passed"] += 1
                    logger.info("✅ Delete test passed")
                else:
                    results["tests_failed"] += 1
                    results["errors"].append("Delete test: file still exists")
                    logger.error("❌ Delete test failed")
            else:
                results["tests_failed"] += 1
                results["errors"].append("Delete test: file doesn't exist")
        except Exception as e:
            results["tests_failed"] += 1
            results["errors"].append(f"Delete test: {str(e)}")
            logger.error(f"❌ Delete test failed: {e}")

    finally:
        # Cleanup
        try:
            if Path(test_file).exists():
                Path(test_file).unlink()
        except:
            pass

    execution_time = time.time() - start_time
    success_rate = (results["tests_passed"] / results["tests_run"] * 100) if results["tests_run"] > 0 else 0

    if results["tests_failed"] == 0:
        status = "✅ DIRECT OPERATIONS FULLY FUNCTIONAL"
        logger.info("🎉 All direct operation tests passed")
    elif results["tests_passed"] == 0:
        status = "❌ CRITICAL: ALL DIRECT OPERATIONS BLOCKED"
        logger.error("🚨 All direct operation tests failed")
    else:
        status = f"⚠️ PARTIAL: {success_rate:.1f}% success rate"
        logger.warning(f"🔧 Mixed results: {success_rate:.1f}% success")

    report = f"""{status}

DIRECT OPERATION TEST REPORT (time: {execution_time:.3f}s):
- Tests Run: {results["tests_run"]}
- Tests Passed: {results["tests_passed"]}
- Tests Failed: {results["tests_failed"]}
- Success Rate: {success_rate:.1f}%

CAPABILITIES:
- File Write: {'✅' if 'Write test' not in str(results["errors"]) else '❌'}
- File Read: {'✅' if 'Read test' not in str(results["errors"]) else '❌'}
- File Edit: {'✅' if 'Edit test' not in str(results["errors"]) else '❌'}
- File Delete: {'✅' if 'Delete test' not in str(results["errors"]) else '❌'}

Test File: {test_file}
Timestamp: {datetime.now().isoformat()}"""

    if results["errors"]:
        report += f"\n\nERRORS:\n" + "\n".join(f"- {error}" for error in results["errors"])

    return report


def get_direct_tools():
    """Get all direct file operation tools."""
    return [
        direct_write_file,
        direct_read_file,
        direct_list_directory,
        direct_edit_file,
        direct_system_test
    ]