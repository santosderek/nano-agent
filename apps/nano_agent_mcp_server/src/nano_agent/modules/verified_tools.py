"""
Verified Tools for Nano Agent with Dual-Verification Architecture.

This module provides file operation tools that include mandatory verification
to prevent phantom operations and ensure actual file system changes occur.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import function_tool decorator from agents SDK
try:
    from agents import function_tool
except ImportError:
    # Fallback if agents SDK not available
    def function_tool(func):
        return func

from .files import resolve_path, ensure_parent_exists, format_path_for_display

logger = logging.getLogger(__name__)

# Constants for verification
VERIFICATION_RETRY_COUNT = 3
VERIFICATION_DELAY = 0.1  # seconds
MAX_CONTENT_LENGTH_LOG = 200


class VerificationError(Exception):
    """Raised when tool operation cannot be verified."""
    pass


class FileSystemAccessError(Exception):
    """Raised when file system access is completely blocked."""
    pass


def _sync_file_system():
    """Force file system synchronization."""
    try:
        # Try to sync file system writes
        os.sync()
    except (OSError, AttributeError):
        # sync() not available on all platforms
        pass

    # Small delay to allow file system operations to complete
    time.sleep(VERIFICATION_DELAY)


def _verify_file_exists(file_path: str) -> bool:
    """Verify that a file actually exists on the file system."""
    try:
        path = Path(file_path).resolve()
        exists = path.exists() and path.is_file()
        logger.debug(f"File existence check for {file_path}: {exists}")
        return exists
    except Exception as e:
        logger.debug(f"File existence check failed for {file_path}: {e}")
        return False


def _verify_file_content(file_path: str, expected_content: str) -> bool:
    """Verify that a file contains the expected content."""
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            logger.debug(f"Content verification failed - file doesn't exist: {file_path}")
            return False

        actual_content = path.read_text(encoding='utf-8')
        matches = actual_content == expected_content

        if not matches:
            logger.debug(f"Content mismatch in {file_path}: expected {len(expected_content)} chars, got {len(actual_content)} chars")
            if len(expected_content) < 100 and len(actual_content) < 100:
                logger.debug(f"Expected: {repr(expected_content)}")
                logger.debug(f"Actual: {repr(actual_content)}")

        return matches
    except Exception as e:
        logger.debug(f"Content verification failed for {file_path}: {e}")
        return False


def _verify_directory_exists(dir_path: str) -> bool:
    """Verify that a directory actually exists on the file system."""
    try:
        path = Path(dir_path).resolve()
        exists = path.exists() and path.is_dir()
        logger.debug(f"Directory existence check for {dir_path}: {exists}")
        return exists
    except Exception as e:
        logger.debug(f"Directory existence check failed for {dir_path}: {e}")
        return False


def _perform_verification_with_retry(verification_func, *args, **kwargs) -> bool:
    """Perform verification with retries and file system sync."""
    for attempt in range(VERIFICATION_RETRY_COUNT):
        _sync_file_system()

        if verification_func(*args, **kwargs):
            return True

        if attempt < VERIFICATION_RETRY_COUNT - 1:
            logger.debug(f"Verification attempt {attempt + 1} failed, retrying...")
            time.sleep(VERIFICATION_DELAY * (attempt + 1))

    return False


@function_tool
def verified_write_file(file_path: str, content: str) -> str:
    """
    Write content to a file with mandatory verification.

    This tool writes content to a file and then verifies that the operation
    actually occurred by reading the file back and checking the content.

    Args:
        file_path: Path to the file to write (relative or absolute)
        content: Content to write to the file

    Returns:
        Success message with verification status, or raises VerificationError

    Raises:
        VerificationError: If the write operation cannot be verified
        FileSystemAccessError: If file system access is completely blocked
    """
    start_time = time.time()

    try:
        # Resolve to absolute path
        path = resolve_path(file_path)
        display_path = format_path_for_display(path)

        logger.info(f"Starting verified write to: {display_path}")

        # Ensure parent directories exist
        ensure_parent_exists(path)

        # Step 1: Perform the write operation with multiple methods for reliability
        try:
            # Method 1: Standard write with explicit sync
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            logger.debug(f"Write operation completed for {display_path}")

        except Exception as e:
            raise VerificationError(f"Write operation failed for {file_path}: {str(e)}")

        # Step 2: Verify the write operation with retries
        logger.debug(f"Starting verification for {display_path}")

        # Verify file exists
        if not _perform_verification_with_retry(_verify_file_exists, str(path)):
            raise VerificationError(f"CRITICAL: File was not created: {file_path}")

        # Verify content matches
        if not _perform_verification_with_retry(_verify_file_content, str(path), content):
            raise VerificationError(f"CRITICAL: File content verification failed: {file_path}")

        # Step 3: Additional integrity check - re-read and compare sizes
        try:
            actual_content = path.read_text(encoding='utf-8')
            if len(actual_content) != len(content):
                raise VerificationError(
                    f"CRITICAL: Content length mismatch: expected {len(content)} chars, "
                    f"got {len(actual_content)} chars in {file_path}"
                )
        except Exception as e:
            raise VerificationError(f"CRITICAL: Could not re-read file for integrity check: {e}")

        execution_time = time.time() - start_time

        success_msg = (
            f"✅ VERIFIED SUCCESS: Wrote and verified {len(content)} characters to {display_path} "
            f"(completed in {execution_time:.3f}s)"
        )

        logger.info(success_msg)
        return success_msg

    except VerificationError:
        # Re-raise verification errors as-is
        raise
    except Exception as e:
        # Convert other exceptions to verification errors
        error_msg = f"Write operation failed for {file_path}: {str(e)}"
        logger.error(error_msg)
        raise VerificationError(error_msg)


@function_tool
def verified_read_file(file_path: str) -> str:
    """
    Read file content with existence verification.

    This tool reads a file and verifies that the file actually exists
    and is readable before returning the content.

    Args:
        file_path: Path to the file to read (relative or absolute)

    Returns:
        File content with verification status

    Raises:
        VerificationError: If the file cannot be verified or read
    """
    start_time = time.time()

    try:
        # Resolve to absolute path
        path = resolve_path(file_path)
        display_path = format_path_for_display(path)

        logger.info(f"Starting verified read from: {display_path}")

        # Step 1: Verify file exists
        if not _perform_verification_with_retry(_verify_file_exists, str(path)):
            raise VerificationError(f"CRITICAL: File does not exist: {file_path}")

        # Step 2: Check if it's actually a file (not directory)
        if not path.is_file():
            raise VerificationError(f"CRITICAL: Path is not a file: {file_path}")

        # Step 3: Read the file content
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError as e:
            raise VerificationError(f"CRITICAL: File encoding error in {file_path}: {str(e)}")
        except Exception as e:
            raise VerificationError(f"CRITICAL: Could not read file {file_path}: {str(e)}")

        # Step 4: Verify we got actual content (not empty due to access issues)
        file_size = path.stat().st_size
        if file_size > 0 and len(content) == 0:
            raise VerificationError(f"CRITICAL: File size is {file_size} bytes but read 0 characters from {file_path}")

        execution_time = time.time() - start_time

        success_msg = (
            f"✅ VERIFIED SUCCESS: Read and verified {len(content)} characters from {display_path} "
            f"(completed in {execution_time:.3f}s)"
        )

        logger.info(success_msg)

        # Return success message and content
        return f"{success_msg}\n\n--- FILE CONTENT ---\n{content}"

    except VerificationError:
        # Re-raise verification errors as-is
        raise
    except Exception as e:
        # Convert other exceptions to verification errors
        error_msg = f"Read operation failed for {file_path}: {str(e)}"
        logger.error(error_msg)
        raise VerificationError(error_msg)


@function_tool
def verified_list_directory(directory_path: Optional[str] = None) -> str:
    """
    List directory contents with verification.

    Args:
        directory_path: Path to directory (default: current working directory)

    Returns:
        Directory listing with verification status

    Raises:
        VerificationError: If directory cannot be verified or listed
    """
    start_time = time.time()

    try:
        # Default to current working directory if no path provided
        if directory_path is None:
            path = Path.cwd()
            display_path = str(path)
        else:
            path = resolve_path(directory_path)
            display_path = format_path_for_display(path)

        logger.info(f"Starting verified directory listing: {display_path}")

        # Step 1: Verify directory exists
        if not _perform_verification_with_retry(_verify_directory_exists, str(path)):
            raise VerificationError(f"CRITICAL: Directory does not exist: {directory_path or 'current directory'}")

        # Step 2: Check if it's actually a directory
        if not path.is_dir():
            raise VerificationError(f"CRITICAL: Path is not a directory: {directory_path or 'current directory'}")

        # Step 3: List directory contents
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
            raise VerificationError(f"CRITICAL: Permission denied accessing directory: {directory_path}")
        except Exception as e:
            raise VerificationError(f"CRITICAL: Could not list directory {directory_path}: {str(e)}")

        execution_time = time.time() - start_time

        # Format result
        result = f"✅ VERIFIED SUCCESS: Listed directory {display_path} (completed in {execution_time:.3f}s)\n"
        result += f"Total items: {len(items)}\n\n"
        result += "\n".join(items) if items else "Directory is empty"

        logger.info(f"Successfully listed {len(items)} items in {display_path}")
        return result

    except VerificationError:
        # Re-raise verification errors as-is
        raise
    except Exception as e:
        # Convert other exceptions to verification errors
        error_msg = f"Directory listing failed for {directory_path}: {str(e)}"
        logger.error(error_msg)
        raise VerificationError(error_msg)


@function_tool
def verified_edit_file(file_path: str, old_str: str, new_str: str) -> str:
    """
    Edit a file by replacing text with verification.

    Args:
        file_path: Path to the file to edit
        old_str: Text to find and replace (must match exactly)
        new_str: Text to replace with

    Returns:
        Success message with verification status

    Raises:
        VerificationError: If the edit operation cannot be verified
    """
    start_time = time.time()

    try:
        # Resolve to absolute path
        path = resolve_path(file_path)
        display_path = format_path_for_display(path)

        logger.info(f"Starting verified edit of: {display_path}")

        # Step 1: Verify file exists
        if not _perform_verification_with_retry(_verify_file_exists, str(path)):
            raise VerificationError(f"CRITICAL: File does not exist: {file_path}")

        # Step 2: Read current content
        try:
            original_content = path.read_text(encoding='utf-8')
        except Exception as e:
            raise VerificationError(f"CRITICAL: Could not read file for editing: {str(e)}")

        # Step 3: Check if old_str exists in the file
        if old_str not in original_content:
            raise VerificationError(f"CRITICAL: Text to replace not found in file: {old_str[:100]}...")

        # Check for multiple occurrences
        occurrences = original_content.count(old_str)
        if occurrences > 1:
            raise VerificationError(
                f"CRITICAL: Found {occurrences} occurrences of text. "
                f"Please provide more specific text to ensure unique replacement."
            )

        # Step 4: Perform the replacement
        new_content = original_content.replace(old_str, new_str, 1)

        # Step 5: Write the new content using verified write
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            raise VerificationError(f"CRITICAL: Could not write edited content: {str(e)}")

        # Step 6: Verify the edit was successful
        if not _perform_verification_with_retry(_verify_file_content, str(path), new_content):
            raise VerificationError(f"CRITICAL: Edit verification failed for {file_path}")

        # Step 7: Verify the replacement actually occurred
        try:
            verified_content = path.read_text(encoding='utf-8')
            if new_str not in verified_content:
                raise VerificationError(f"CRITICAL: New text not found in edited file")
            if old_str in verified_content:
                raise VerificationError(f"CRITICAL: Old text still present in edited file")
        except Exception as e:
            raise VerificationError(f"CRITICAL: Could not verify edit results: {str(e)}")

        execution_time = time.time() - start_time

        success_msg = (
            f"✅ VERIFIED SUCCESS: Edited {display_path} - replaced {len(old_str)} chars with {len(new_str)} chars "
            f"(completed in {execution_time:.3f}s)"
        )

        logger.info(success_msg)
        return success_msg

    except VerificationError:
        # Re-raise verification errors as-is
        raise
    except Exception as e:
        # Convert other exceptions to verification errors
        error_msg = f"Edit operation failed for {file_path}: {str(e)}"
        logger.error(error_msg)
        raise VerificationError(error_msg)


@function_tool
def verified_file_system_test() -> str:
    """
    Test actual file system access with comprehensive verification.

    This tool performs a comprehensive test of file system capabilities
    to detect if operations are being blocked or sandboxed.

    Returns:
        Detailed report of file system access capabilities
    """
    start_time = time.time()
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "test_file": f"/tmp/nano_agent_verification_{int(time.time())}.txt",
        "tests_passed": 0,
        "tests_failed": 0,
        "errors": [],
        "capabilities": {}
    }

    test_content = f"File system verification test\nTimestamp: {datetime.now().isoformat()}\nRandom: {time.time()}"

    try:
        logger.info("Starting comprehensive file system verification test")

        # Test 1: Write access
        try:
            with open(test_results["test_file"], 'w', encoding='utf-8') as f:
                f.write(test_content)
                f.flush()
                os.fsync(f.fileno())
            test_results["capabilities"]["write"] = True
            test_results["tests_passed"] += 1
            logger.debug("✅ Write test passed")
        except Exception as e:
            test_results["capabilities"]["write"] = False
            test_results["tests_failed"] += 1
            test_results["errors"].append(f"Write test failed: {str(e)}")
            logger.debug(f"❌ Write test failed: {e}")

        # Test 2: File existence verification
        try:
            exists = Path(test_results["test_file"]).exists()
            test_results["capabilities"]["file_exists_check"] = exists
            if exists:
                test_results["tests_passed"] += 1
                logger.debug("✅ File existence test passed")
            else:
                test_results["tests_failed"] += 1
                test_results["errors"].append("File was written but does not exist")
                logger.debug("❌ File existence test failed")
        except Exception as e:
            test_results["capabilities"]["file_exists_check"] = False
            test_results["tests_failed"] += 1
            test_results["errors"].append(f"File existence test failed: {str(e)}")
            logger.debug(f"❌ File existence test failed: {e}")

        # Test 3: Read access
        try:
            if test_results["capabilities"].get("file_exists_check", False):
                read_content = Path(test_results["test_file"]).read_text(encoding='utf-8')
                content_matches = read_content == test_content
                test_results["capabilities"]["read"] = True
                test_results["capabilities"]["content_integrity"] = content_matches

                if content_matches:
                    test_results["tests_passed"] += 1
                    logger.debug("✅ Read and content integrity test passed")
                else:
                    test_results["tests_failed"] += 1
                    test_results["errors"].append(
                        f"Content mismatch: wrote {len(test_content)} chars, read {len(read_content)} chars"
                    )
                    logger.debug("❌ Content integrity test failed")
            else:
                test_results["capabilities"]["read"] = False
                test_results["tests_failed"] += 1
                test_results["errors"].append("Cannot test read - file does not exist")
        except Exception as e:
            test_results["capabilities"]["read"] = False
            test_results["tests_failed"] += 1
            test_results["errors"].append(f"Read test failed: {str(e)}")
            logger.debug(f"❌ Read test failed: {e}")

        # Test 4: File modification
        try:
            if test_results["capabilities"].get("read", False):
                modified_content = test_content + "\nModification test"
                with open(test_results["test_file"], 'a', encoding='utf-8') as f:
                    f.write("\nModification test")
                    f.flush()
                    os.fsync(f.fileno())

                # Verify modification
                _sync_file_system()
                final_content = Path(test_results["test_file"]).read_text(encoding='utf-8')
                modification_successful = final_content == modified_content

                test_results["capabilities"]["modify"] = modification_successful
                if modification_successful:
                    test_results["tests_passed"] += 1
                    logger.debug("✅ Modification test passed")
                else:
                    test_results["tests_failed"] += 1
                    test_results["errors"].append("File modification verification failed")
                    logger.debug("❌ Modification test failed")
            else:
                test_results["capabilities"]["modify"] = False
                test_results["tests_failed"] += 1
                test_results["errors"].append("Cannot test modify - read test failed")
        except Exception as e:
            test_results["capabilities"]["modify"] = False
            test_results["tests_failed"] += 1
            test_results["errors"].append(f"Modification test failed: {str(e)}")
            logger.debug(f"❌ Modification test failed: {e}")

        # Test 5: File deletion
        try:
            if Path(test_results["test_file"]).exists():
                Path(test_results["test_file"]).unlink()
                _sync_file_system()
                deletion_successful = not Path(test_results["test_file"]).exists()

                test_results["capabilities"]["delete"] = deletion_successful
                if deletion_successful:
                    test_results["tests_passed"] += 1
                    logger.debug("✅ Deletion test passed")
                else:
                    test_results["tests_failed"] += 1
                    test_results["errors"].append("File deletion verification failed")
                    logger.debug("❌ Deletion test failed")
            else:
                test_results["capabilities"]["delete"] = False
                test_results["tests_failed"] += 1
                test_results["errors"].append("Cannot test delete - file does not exist")
        except Exception as e:
            test_results["capabilities"]["delete"] = False
            test_results["tests_failed"] += 1
            test_results["errors"].append(f"Deletion test failed: {str(e)}")
            logger.debug(f"❌ Deletion test failed: {e}")

    except Exception as e:
        test_results["errors"].append(f"Test framework error: {str(e)}")
        logger.error(f"File system test framework error: {e}")

    # Cleanup any remaining test file
    try:
        test_file_path = Path(test_results["test_file"])
        if test_file_path.exists():
            test_file_path.unlink()
    except:
        pass

    execution_time = time.time() - start_time
    test_results["execution_time_seconds"] = execution_time

    # Generate summary
    total_tests = test_results["tests_passed"] + test_results["tests_failed"]
    success_rate = (test_results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0

    if test_results["tests_failed"] == 0:
        status = "✅ VERIFIED: File system access is fully functional"
        logger.info("File system verification: ALL TESTS PASSED")
    elif test_results["tests_passed"] == 0:
        status = "❌ CRITICAL: File system access is completely blocked"
        logger.error("File system verification: ALL TESTS FAILED")
    else:
        status = f"⚠️  WARNING: Partial file system access ({success_rate:.1f}% success rate)"
        logger.warning(f"File system verification: PARTIAL FAILURE ({success_rate:.1f}% success)")

    # Format detailed report
    report = f"""{status}

VERIFICATION REPORT (completed in {execution_time:.3f}s):
- Tests Passed: {test_results["tests_passed"]}
- Tests Failed: {test_results["tests_failed"]}
- Success Rate: {success_rate:.1f}%

CAPABILITIES:
- Write Access: {'✅' if test_results["capabilities"].get("write", False) else '❌'}
- Read Access: {'✅' if test_results["capabilities"].get("read", False) else '❌'}
- File Existence Check: {'✅' if test_results["capabilities"].get("file_exists_check", False) else '❌'}
- Content Integrity: {'✅' if test_results["capabilities"].get("content_integrity", False) else '❌'}
- File Modification: {'✅' if test_results["capabilities"].get("modify", False) else '❌'}
- File Deletion: {'✅' if test_results["capabilities"].get("delete", False) else '❌'}"""

    if test_results["errors"]:
        report += f"\n\nERRORS DETECTED:\n" + "\n".join(f"- {error}" for error in test_results["errors"])

    report += f"\n\nTest File Used: {test_results['test_file']}"
    report += f"\nTimestamp: {test_results['timestamp']}"

    return report


def get_verified_nano_agent_tools():
    """
    Get all verified tools for the nano agent.

    Returns:
        List of verified tool functions with dual-verification
    """
    return [
        verified_write_file,
        verified_read_file,
        verified_list_directory,
        verified_edit_file,
        verified_file_system_test
    ]