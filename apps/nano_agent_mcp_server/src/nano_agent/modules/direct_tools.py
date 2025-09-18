"""
Direct File System Tools with FORCED SELF-VALIDATION.

This module provides real file system operations using native Python
with mandatory external validation that cannot be faked or simulated.
Every operation must prove it actually occurred using multiple verification methods.
"""

import os
import sys
import time
import subprocess
import hashlib
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


class ValidationFailure(Exception):
    """Raised when external validation proves an operation didn't occur."""
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


def _external_validation_file_exists(file_path: str) -> bool:
    """FORCED external validation using multiple independent methods."""
    path = Path(file_path).resolve()

    # Method 1: Python pathlib
    exists_pathlib = path.exists() and path.is_file()

    # Method 2: os.path
    exists_os = os.path.exists(str(path)) and os.path.isfile(str(path))

    # Method 3: subprocess ls (cannot be faked)
    try:
        result = subprocess.run(['ls', '-la', str(path)], capture_output=True, text=True, timeout=5)
        exists_subprocess = result.returncode == 0 and str(path) in result.stdout
    except:
        exists_subprocess = False

    # Method 4: stat command (another external check)
    try:
        stat_result = subprocess.run(['stat', str(path)], capture_output=True, text=True, timeout=5)
        exists_stat = stat_result.returncode == 0
    except:
        exists_stat = False

    # ALL methods must agree - no faking allowed
    consensus = exists_pathlib and exists_os and exists_subprocess and exists_stat

    logger.info(f"EXTERNAL FILE VALIDATION {file_path}: pathlib={exists_pathlib}, os={exists_os}, ls={exists_subprocess}, stat={exists_stat} → CONSENSUS={consensus}")

    return consensus


def _external_validation_content_matches(file_path: str, expected_content: str) -> bool:
    """FORCED external content validation using multiple methods."""
    if not _external_validation_file_exists(file_path):
        return False

    path = Path(file_path)

    try:
        # Method 1: Python read
        content_python = path.read_text(encoding='utf-8')

        # Method 2: subprocess cat (external command - cannot be faked)
        cat_result = subprocess.run(['cat', str(path)], capture_output=True, text=True, timeout=10)
        content_subprocess = cat_result.stdout if cat_result.returncode == 0 else ""

        # Method 3: File size verification
        expected_size = len(expected_content.encode('utf-8'))
        actual_size = path.stat().st_size
        size_matches = expected_size == actual_size

        # Method 4: MD5 hash verification (cannot be faked)
        expected_hash = hashlib.md5(expected_content.encode('utf-8')).hexdigest()
        actual_hash = hashlib.md5(content_python.encode('utf-8')).hexdigest()
        hash_matches = expected_hash == actual_hash

        # Method 5: subprocess wc (external word count - cannot be faked)
        wc_result = subprocess.run(['wc', '-c', str(path)], capture_output=True, text=True, timeout=5)
        wc_size = int(wc_result.stdout.split()[0]) if wc_result.returncode == 0 else -1
        wc_matches = wc_size == expected_size

        # ALL methods must agree
        all_match = (
            content_python == expected_content and
            content_subprocess == expected_content and
            size_matches and
            hash_matches and
            wc_matches
        )

        logger.info(f"EXTERNAL CONTENT VALIDATION {file_path}: python_match={content_python == expected_content}, cat_match={content_subprocess == expected_content}, size_match={size_matches}, hash_match={hash_matches}, wc_match={wc_matches} → CONSENSUS={all_match}")

        return all_match

    except Exception as e:
        logger.error(f"External content validation error for {file_path}: {e}")
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

        # FORCED EXTERNAL VALIDATION - cannot be faked
        logger.info(f"🔍 FORCING EXTERNAL VALIDATION for {display_path}")

        # External validation 1: File exists using multiple methods
        if not _external_validation_file_exists(str(path)):
            raise ValidationFailure(f"EXTERNAL VALIDATION FAILED: File was not actually created: {file_path}")

        # External validation 2: Content matches using multiple methods
        if not _external_validation_content_matches(str(path), content):
            raise ValidationFailure(f"EXTERNAL VALIDATION FAILED: Content verification failed: {file_path}")

        # External validation 3: Additional subprocess checks
        try:
            # Use file command to verify it's a text file
            file_result = subprocess.run(['file', str(path)], capture_output=True, text=True, timeout=5)
            if file_result.returncode != 0:
                raise ValidationFailure(f"EXTERNAL VALIDATION FAILED: File command failed for {file_path}")

            # Use wc to verify byte count externally
            wc_result = subprocess.run(['wc', '-c', str(path)], capture_output=True, text=True, timeout=5)
            if wc_result.returncode == 0:
                external_size = int(wc_result.stdout.split()[0])
                expected_size = len(content.encode('utf-8'))
                if external_size != expected_size:
                    raise ValidationFailure(f"EXTERNAL VALIDATION FAILED: External size check failed - expected {expected_size}, got {external_size}")

        except ValidationFailure:
            raise
        except Exception as e:
            logger.warning(f"Additional external validation checks failed: {e}")
            # Don't fail on these, but log the issue

        execution_time = time.time() - start_time

        # Get final file size for success message
        final_size = path.stat().st_size

        success_msg = (
            f"✅ EXTERNALLY VALIDATED: Wrote and verified {len(content)} chars to {display_path} "
            f"using 5 independent validation methods (Python, os.path, ls, stat, wc) "
            f"- Size: {final_size} bytes, Time: {execution_time:.3f}s - NO PHANTOM OPERATIONS POSSIBLE"
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