"""
Preflight Checks System for Nano Agent.

This module provides comprehensive preflight checks to verify file system
access and detect potential restrictions before agent execution begins.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants for preflight checks
PREFLIGHT_TIMEOUT = 10  # seconds
TEST_DIRECTORIES = ["/tmp", "/var/tmp"]
REQUIRED_PERMISSIONS = ["read", "write", "create", "delete"]


class PreflightCheckResult:
    """Result of a preflight check operation."""

    def __init__(self, name: str, passed: bool, message: str, details: Dict[str, Any] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()


class PreflightChecker:
    """Performs comprehensive preflight checks for file system operations."""

    def __init__(self):
        self.results: List[PreflightCheckResult] = []
        self.start_time = None
        self.end_time = None

    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all preflight checks and return comprehensive results.

        Returns:
            Dictionary with check results and overall status
        """
        self.start_time = time.time()
        self.results = []

        logger.info("Starting comprehensive preflight checks...")

        # Run individual checks
        self._check_basic_file_system_access()
        self._check_directory_permissions()
        self._check_file_operations()
        self._check_environment_restrictions()
        self._check_working_directory_access()

        self.end_time = time.time()
        execution_time = self.end_time - self.start_time

        # Compile results
        passed_checks = [r for r in self.results if r.passed]
        failed_checks = [r for r in self.results if not r.passed]

        overall_status = "PASS" if len(failed_checks) == 0 else "FAIL"
        is_operational = len(failed_checks) == 0

        results = {
            "overall_status": overall_status,
            "is_operational": is_operational,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat(),
            "checks_total": len(self.results),
            "checks_passed": len(passed_checks),
            "checks_failed": len(failed_checks),
            "success_rate": len(passed_checks) / len(self.results) * 100 if self.results else 0,
            "passed_checks": [{"name": r.name, "message": r.message} for r in passed_checks],
            "failed_checks": [{"name": r.name, "message": r.message, "details": r.details} for r in failed_checks],
            "critical_errors": [r.message for r in failed_checks if "CRITICAL" in r.message],
            "warnings": [r.message for r in failed_checks if "WARNING" in r.message]
        }

        # Log summary
        if is_operational:
            logger.info(f"✅ All preflight checks passed ({len(passed_checks)}/{len(self.results)}) in {execution_time:.3f}s")
        else:
            logger.error(f"❌ Preflight checks failed ({len(failed_checks)}/{len(self.results)} failures) in {execution_time:.3f}s")
            for check in failed_checks:
                logger.error(f"  - {check.name}: {check.message}")

        return results

    def _check_basic_file_system_access(self):
        """Check if basic file system operations are available."""
        check_name = "Basic File System Access"

        try:
            # Test basic Path operations
            current_dir = Path.cwd()
            if not current_dir.exists():
                self.results.append(PreflightCheckResult(
                    check_name, False,
                    "CRITICAL: Cannot access current working directory",
                    {"current_dir": str(current_dir)}
                ))
                return

            # Test basic file system queries
            try:
                list(current_dir.iterdir())
                can_list = True
            except Exception as e:
                can_list = False
                list_error = str(e)

            if can_list:
                self.results.append(PreflightCheckResult(
                    check_name, True,
                    "✅ Basic file system access is available"
                ))
            else:
                self.results.append(PreflightCheckResult(
                    check_name, False,
                    f"CRITICAL: Cannot list current directory: {list_error}",
                    {"list_error": list_error}
                ))

        except Exception as e:
            self.results.append(PreflightCheckResult(
                check_name, False,
                f"CRITICAL: Basic file system access failed: {str(e)}",
                {"error": str(e), "error_type": type(e).__name__}
            ))

    def _check_directory_permissions(self):
        """Check permissions for common directories."""
        check_name = "Directory Permissions"

        accessible_dirs = []
        inaccessible_dirs = []

        # Test current working directory
        try:
            cwd = Path.cwd()
            self._test_directory_access(cwd, "current_working_directory")
            accessible_dirs.append(str(cwd))
        except Exception as e:
            inaccessible_dirs.append(f"current_working_directory: {str(e)}")

        # Test common temp directories
        for test_dir in TEST_DIRECTORIES:
            try:
                dir_path = Path(test_dir)
                if dir_path.exists():
                    self._test_directory_access(dir_path, test_dir)
                    accessible_dirs.append(test_dir)
                else:
                    inaccessible_dirs.append(f"{test_dir}: directory does not exist")
            except Exception as e:
                inaccessible_dirs.append(f"{test_dir}: {str(e)}")

        # Test home directory if available
        try:
            home_dir = Path.home()
            if home_dir.exists():
                self._test_directory_access(home_dir, "home_directory")
                accessible_dirs.append(str(home_dir))
        except Exception as e:
            inaccessible_dirs.append(f"home_directory: {str(e)}")

        if accessible_dirs and not inaccessible_dirs:
            self.results.append(PreflightCheckResult(
                check_name, True,
                f"✅ All tested directories are accessible ({len(accessible_dirs)} directories)",
                {"accessible_directories": accessible_dirs}
            ))
        elif accessible_dirs:
            self.results.append(PreflightCheckResult(
                check_name, True,
                f"⚠️  Some directories accessible ({len(accessible_dirs)} accessible, {len(inaccessible_dirs)} restricted)",
                {"accessible_directories": accessible_dirs, "inaccessible_directories": inaccessible_dirs}
            ))
        else:
            self.results.append(PreflightCheckResult(
                check_name, False,
                "CRITICAL: No directories are accessible for file operations",
                {"inaccessible_directories": inaccessible_dirs}
            ))

    def _test_directory_access(self, dir_path: Path, dir_name: str) -> bool:
        """Test access permissions for a specific directory."""
        if not dir_path.exists():
            raise Exception(f"Directory does not exist: {dir_path}")

        if not dir_path.is_dir():
            raise Exception(f"Path is not a directory: {dir_path}")

        # Test read access
        try:
            list(dir_path.iterdir())
        except PermissionError:
            raise Exception(f"Read permission denied")
        except Exception as e:
            raise Exception(f"Read access failed: {str(e)}")

        # Test write access with a temporary file
        test_file = dir_path / f"preflight_test_{int(time.time())}.tmp"
        try:
            test_file.write_text("preflight test")
            if not test_file.exists():
                raise Exception("Write succeeded but file not created")

            # Test read back
            content = test_file.read_text()
            if content != "preflight test":
                raise Exception("Content verification failed")

            # Test delete
            test_file.unlink()
            if test_file.exists():
                raise Exception("Delete succeeded but file still exists")

        except PermissionError:
            raise Exception("Write permission denied")
        except Exception as e:
            raise Exception(f"Write access test failed: {str(e)}")
        finally:
            # Cleanup
            try:
                if test_file.exists():
                    test_file.unlink()
            except:
                pass

        return True

    def _check_file_operations(self):
        """Check if file operations work correctly with verification."""
        check_name = "File Operations Verification"

        try:
            # Import the verified file system test
            from .verified_tools import verified_file_system_test

            # Run the comprehensive file system test
            test_result = verified_file_system_test()

            # Parse the result to determine if it passed
            if "✅ VERIFIED: File system access is fully functional" in test_result:
                self.results.append(PreflightCheckResult(
                    check_name, True,
                    "✅ All file operations verified and working correctly",
                    {"test_result": test_result}
                ))
            elif "⚠️  WARNING: Partial file system access" in test_result:
                self.results.append(PreflightCheckResult(
                    check_name, True,
                    "⚠️  File operations partially working - some restrictions detected",
                    {"test_result": test_result}
                ))
            elif "❌ CRITICAL: File system access is completely blocked" in test_result:
                self.results.append(PreflightCheckResult(
                    check_name, False,
                    "CRITICAL: File system operations are completely blocked",
                    {"test_result": test_result}
                ))
            else:
                self.results.append(PreflightCheckResult(
                    check_name, False,
                    "WARNING: File operations test returned unexpected result",
                    {"test_result": test_result}
                ))

        except Exception as e:
            self.results.append(PreflightCheckResult(
                check_name, False,
                f"CRITICAL: File operations test failed to execute: {str(e)}",
                {"error": str(e), "error_type": type(e).__name__}
            ))

    def _check_environment_restrictions(self):
        """Check for environment restrictions that might affect operations."""
        check_name = "Environment Restrictions"

        restrictions = []
        warnings = []

        # Check if running in a container
        if Path("/.dockerenv").exists():
            warnings.append("Running in Docker container - mount restrictions may apply")

        # Check environment variables that might indicate restrictions
        if os.getenv("SANDBOX"):
            restrictions.append("SANDBOX environment variable detected")

        if os.getenv("RESTRICTED_ENV"):
            restrictions.append("RESTRICTED_ENV environment variable detected")

        # Check for common container indicators
        container_indicators = ["container", "CONTAINER_ID", "KUBERNETES_SERVICE_HOST"]
        for indicator in container_indicators:
            if os.getenv(indicator):
                warnings.append(f"Container environment detected: {indicator}")

        # Check user permissions
        try:
            current_user = os.getenv("USER", "unknown")
            if current_user == "root":
                warnings.append("Running as root user - unexpected in sandboxed environments")
            elif current_user == "unknown":
                warnings.append("User identity unknown - may indicate restricted environment")
        except:
            warnings.append("Cannot determine user identity")

        if restrictions:
            self.results.append(PreflightCheckResult(
                check_name, False,
                f"WARNING: Environment restrictions detected: {', '.join(restrictions)}",
                {"restrictions": restrictions, "warnings": warnings}
            ))
        elif warnings:
            self.results.append(PreflightCheckResult(
                check_name, True,
                f"⚠️  Environment warnings detected: {', '.join(warnings)}",
                {"warnings": warnings}
            ))
        else:
            self.results.append(PreflightCheckResult(
                check_name, True,
                "✅ No obvious environment restrictions detected"
            ))

    def _check_working_directory_access(self):
        """Check if the working directory has full access permissions."""
        check_name = "Working Directory Access"

        try:
            cwd = Path.cwd()
            cwd_str = str(cwd)

            # Check if we can create subdirectories
            test_subdir = cwd / f"preflight_subdir_test_{int(time.time())}"
            can_create_dirs = False

            try:
                test_subdir.mkdir()
                can_create_dirs = test_subdir.exists()
                if can_create_dirs:
                    test_subdir.rmdir()
            except Exception as e:
                logger.debug(f"Cannot create subdirectories in working directory: {e}")

            # Check if we can create files in working directory
            test_file = cwd / f"preflight_cwd_test_{int(time.time())}.tmp"
            can_create_files = False

            try:
                test_file.write_text("test")
                can_create_files = test_file.exists()
                if can_create_files:
                    test_file.unlink()
            except Exception as e:
                logger.debug(f"Cannot create files in working directory: {e}")

            # Evaluate results
            if can_create_files and can_create_dirs:
                self.results.append(PreflightCheckResult(
                    check_name, True,
                    f"✅ Full access to working directory: {cwd_str}",
                    {"working_directory": cwd_str, "can_create_files": True, "can_create_dirs": True}
                ))
            elif can_create_files:
                self.results.append(PreflightCheckResult(
                    check_name, True,
                    f"⚠️  Limited access to working directory (files only): {cwd_str}",
                    {"working_directory": cwd_str, "can_create_files": True, "can_create_dirs": False}
                ))
            else:
                self.results.append(PreflightCheckResult(
                    check_name, False,
                    f"CRITICAL: No write access to working directory: {cwd_str}",
                    {"working_directory": cwd_str, "can_create_files": False, "can_create_dirs": False}
                ))

        except Exception as e:
            self.results.append(PreflightCheckResult(
                check_name, False,
                f"CRITICAL: Cannot access working directory: {str(e)}",
                {"error": str(e), "error_type": type(e).__name__}
            ))


def run_preflight_checks() -> Dict[str, Any]:
    """
    Run comprehensive preflight checks for nano agent file operations.

    Returns:
        Dictionary containing check results and operational status
    """
    checker = PreflightChecker()
    return checker.run_all_checks()


def is_file_system_operational() -> Tuple[bool, List[str]]:
    """
    Quick check if file system operations are likely to work.

    Returns:
        Tuple of (is_operational, list_of_critical_errors)
    """
    try:
        results = run_preflight_checks()
        return results["is_operational"], results["critical_errors"]
    except Exception as e:
        return False, [f"Preflight check system failed: {str(e)}"]


def get_preflight_summary() -> str:
    """
    Get a human-readable summary of preflight check results.

    Returns:
        Formatted string with preflight check summary
    """
    try:
        results = run_preflight_checks()

        if results["is_operational"]:
            summary = (
                f"✅ PREFLIGHT CHECKS PASSED\n"
                f"All {results['checks_passed']} checks passed in {results['execution_time_seconds']:.3f}s\n"
                f"File system operations are ready for use."
            )
        else:
            summary = (
                f"❌ PREFLIGHT CHECKS FAILED\n"
                f"Failed: {results['checks_failed']}/{results['checks_total']} checks "
                f"(success rate: {results['success_rate']:.1f}%)\n"
                f"Execution time: {results['execution_time_seconds']:.3f}s\n\n"
                f"CRITICAL ERRORS:\n"
            )
            for error in results["critical_errors"]:
                summary += f"  - {error}\n"

            if results["warnings"]:
                summary += f"\nWARNINGS:\n"
                for warning in results["warnings"]:
                    summary += f"  - {warning}\n"

        return summary

    except Exception as e:
        return f"❌ PREFLIGHT CHECK SYSTEM ERROR: {str(e)}"