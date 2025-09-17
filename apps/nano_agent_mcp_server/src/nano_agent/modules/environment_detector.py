"""
Environment Detection Module for Nano Agent.

This module provides comprehensive environment detection to identify
execution contexts, restrictions, and potential file system limitations.
"""

import os
import sys
import time
import platform
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Known container/sandbox indicators
CONTAINER_FILES = [
    "/.dockerenv",
    "/.containerenv",
    "/run/.containerenv"
]

CONTAINER_ENV_VARS = [
    "CONTAINER_ID", "CONTAINER_NAME", "DOCKER_CONTAINER",
    "KUBERNETES_SERVICE_HOST", "K8S_NODE_NAME",
    "HOSTNAME", "container"
]

SANDBOX_INDICATORS = [
    "SANDBOX", "RESTRICTED_ENV", "CHROOT", "JAIL",
    "VIRTUAL_ENV", "CONDA_ENV"
]

# Directories to test for write access
TEST_DIRECTORIES = [
    "/tmp", "/var/tmp", "/dev/shm",
    ".", "~", "/home", "/Users"
]


class EnvironmentInfo:
    """Container for environment detection results."""

    def __init__(self):
        self.platform_info = {}
        self.execution_context = {}
        self.file_system_access = {}
        self.restrictions = []
        self.warnings = []
        self.capabilities = {}
        self.risk_level = "unknown"
        self.timestamp = datetime.now().isoformat()


class EnvironmentDetector:
    """Comprehensive environment detection and analysis."""

    def __init__(self):
        self.info = EnvironmentInfo()

    def analyze_environment(self) -> Dict[str, Any]:
        """
        Perform comprehensive environment analysis.

        Returns:
            Dictionary with complete environment analysis results
        """
        logger.info("Starting comprehensive environment analysis...")
        start_time = time.time()

        # Collect all environment information
        self._detect_platform_info()
        self._detect_execution_context()
        self._detect_file_system_access()
        self._detect_process_restrictions()
        self._detect_network_access()
        self._analyze_python_environment()
        self._assess_risk_level()

        execution_time = time.time() - start_time

        # Compile results
        results = {
            "timestamp": self.info.timestamp,
            "execution_time_seconds": execution_time,
            "platform_info": self.info.platform_info,
            "execution_context": self.info.execution_context,
            "file_system_access": self.info.file_system_access,
            "capabilities": self.info.capabilities,
            "restrictions": self.info.restrictions,
            "warnings": self.info.warnings,
            "risk_level": self.info.risk_level,
            "summary": self._generate_summary()
        }

        logger.info(f"Environment analysis completed in {execution_time:.3f}s - Risk Level: {self.info.risk_level}")
        return results

    def _detect_platform_info(self):
        """Detect basic platform and system information."""
        try:
            self.info.platform_info = {
                "system": platform.system(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": sys.version,
                "python_executable": sys.executable,
                "python_path": sys.path[:5],  # First 5 entries only
                "os_name": os.name,
                "current_working_directory": str(Path.cwd()),
                "home_directory": str(Path.home()) if Path.home().exists() else "unknown"
            }

            # Get environment variables (filtered for security)
            safe_env_vars = {}
            for key, value in os.environ.items():
                # Only include non-sensitive environment variables
                if not any(sensitive in key.upper() for sensitive in ["PASSWORD", "SECRET", "KEY", "TOKEN", "AUTH"]):
                    safe_env_vars[key] = value[:100] if len(value) > 100 else value  # Truncate long values

            self.info.platform_info["environment_variables_count"] = len(os.environ)
            self.info.platform_info["safe_environment_sample"] = dict(list(safe_env_vars.items())[:20])  # First 20 only

        except Exception as e:
            logger.error(f"Failed to detect platform info: {e}")
            self.info.platform_info = {"error": str(e)}

    def _detect_execution_context(self):
        """Detect the execution context (container, sandbox, etc.)."""
        context = {
            "containerized": False,
            "container_type": "none",
            "sandboxed": False,
            "sandbox_type": "none",
            "virtualized": False,
            "user_context": {},
            "process_info": {}
        }

        # Check for container indicators
        for container_file in CONTAINER_FILES:
            if Path(container_file).exists():
                context["containerized"] = True
                context["container_type"] = "docker" if "docker" in container_file else "container"
                logger.debug(f"Container detected via file: {container_file}")
                break

        # Check environment variables for container indicators
        for env_var in CONTAINER_ENV_VARS:
            if os.getenv(env_var):
                if not context["containerized"]:
                    context["containerized"] = True
                    context["container_type"] = "kubernetes" if "K8S" in env_var or "KUBERNETES" in env_var else "container"
                logger.debug(f"Container detected via env var: {env_var}")

        # Check for sandbox indicators
        for sandbox_var in SANDBOX_INDICATORS:
            if os.getenv(sandbox_var):
                context["sandboxed"] = True
                context["sandbox_type"] = sandbox_var.lower()
                self.info.restrictions.append(f"Sandbox environment detected: {sandbox_var}")
                logger.debug(f"Sandbox detected via env var: {sandbox_var}")

        # Get user context
        try:
            context["user_context"] = {
                "user": os.getenv("USER", "unknown"),
                "uid": os.getuid() if hasattr(os, "getuid") else "unknown",
                "gid": os.getgid() if hasattr(os, "getgid") else "unknown",
                "groups": os.getgroups() if hasattr(os, "getgroups") else "unknown"
            }
        except Exception as e:
            context["user_context"] = {"error": str(e)}
            logger.debug(f"Failed to get user context: {e}")

        # Get process information
        try:
            context["process_info"] = {
                "pid": os.getpid(),
                "ppid": os.getppid() if hasattr(os, "getppid") else "unknown",
                "cwd": str(Path.cwd()),
                "executable": sys.executable
            }
        except Exception as e:
            context["process_info"] = {"error": str(e)}
            logger.debug(f"Failed to get process info: {e}")

        self.info.execution_context = context

        # Add warnings based on context
        if context["containerized"]:
            self.info.warnings.append(f"Running in {context['container_type']} container - file system mounts may be restricted")

        if context["sandboxed"]:
            self.info.warnings.append(f"Running in {context['sandbox_type']} sandbox - operations may be limited")

    def _detect_file_system_access(self):
        """Test file system access across different directories."""
        access_info = {
            "writable_directories": [],
            "readonly_directories": [],
            "inaccessible_directories": [],
            "access_tests": {},
            "total_directories_tested": 0,
            "successful_writes": 0
        }

        # Expand home directory path
        expanded_dirs = []
        for test_dir in TEST_DIRECTORIES:
            try:
                if test_dir == "~":
                    expanded_dirs.append(str(Path.home()))
                elif test_dir == ".":
                    expanded_dirs.append(str(Path.cwd()))
                else:
                    expanded_dirs.append(test_dir)
            except Exception:
                pass

        # Test each directory
        for test_dir in expanded_dirs:
            access_info["total_directories_tested"] += 1
            test_result = self._test_directory_access(test_dir)
            access_info["access_tests"][test_dir] = test_result

            if test_result["writable"]:
                access_info["writable_directories"].append(test_dir)
                access_info["successful_writes"] += 1
            elif test_result["readable"]:
                access_info["readonly_directories"].append(test_dir)
            else:
                access_info["inaccessible_directories"].append(test_dir)

        # Calculate statistics
        total_tested = access_info["total_directories_tested"]
        if total_tested > 0:
            access_info["write_success_rate"] = (access_info["successful_writes"] / total_tested) * 100
        else:
            access_info["write_success_rate"] = 0

        self.info.file_system_access = access_info

        # Add restrictions based on file system access
        if access_info["successful_writes"] == 0:
            self.info.restrictions.append("CRITICAL: No writable directories found - all file operations will fail")
        elif access_info["write_success_rate"] < 50:
            self.info.restrictions.append(f"LIMITED: Only {access_info['write_success_rate']:.1f}% of directories are writable")

    def _test_directory_access(self, directory: str) -> Dict[str, Any]:
        """Test read/write access to a specific directory."""
        result = {
            "exists": False,
            "is_directory": False,
            "readable": False,
            "writable": False,
            "error": None
        }

        try:
            dir_path = Path(directory)
            result["exists"] = dir_path.exists()

            if not result["exists"]:
                result["error"] = "Directory does not exist"
                return result

            result["is_directory"] = dir_path.is_dir()
            if not result["is_directory"]:
                result["error"] = "Path is not a directory"
                return result

            # Test read access
            try:
                list(dir_path.iterdir())
                result["readable"] = True
            except PermissionError:
                result["error"] = "Permission denied for read access"
            except Exception as e:
                result["error"] = f"Read test failed: {str(e)}"

            # Test write access
            if result["readable"]:
                test_file = dir_path / f"env_test_{int(time.time())}_{os.getpid()}.tmp"
                try:
                    test_file.write_text("environment test")
                    if test_file.exists():
                        result["writable"] = True
                        test_file.unlink()  # Cleanup
                    else:
                        result["error"] = "Write appeared successful but file not created"
                except PermissionError:
                    result["error"] = "Permission denied for write access"
                except Exception as e:
                    result["error"] = f"Write test failed: {str(e)}"
                finally:
                    # Ensure cleanup
                    try:
                        if test_file.exists():
                            test_file.unlink()
                    except:
                        pass

        except Exception as e:
            result["error"] = f"Directory access test failed: {str(e)}"

        return result

    def _detect_process_restrictions(self):
        """Detect process-level restrictions."""
        restrictions = []

        # Check for process limits
        try:
            if hasattr(os, "getrlimit"):
                import resource
                # Check file descriptor limit
                fd_soft, fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                if fd_soft < 1024:
                    restrictions.append(f"Low file descriptor limit: {fd_soft}")

                # Check process limit
                try:
                    proc_soft, proc_hard = resource.getrlimit(resource.RLIMIT_NPROC)
                    if proc_soft < 100:
                        restrictions.append(f"Low process limit: {proc_soft}")
                except:
                    pass

        except Exception as e:
            logger.debug(f"Could not check process limits: {e}")

        # Check for capability restrictions
        try:
            # Test if we can get system information
            if hasattr(os, "uname"):
                os.uname()
                self.info.capabilities["system_info_access"] = True
            else:
                self.info.capabilities["system_info_access"] = False
                restrictions.append("System information access restricted")

        except Exception:
            self.info.capabilities["system_info_access"] = False
            restrictions.append("System information access blocked")

        self.info.restrictions.extend(restrictions)

    def _detect_network_access(self):
        """Detect network access capabilities."""
        network_info = {
            "hostname_resolution": False,
            "localhost_access": False,
            "external_access": False
        }

        # Test basic hostname resolution
        try:
            import socket
            socket.gethostname()
            network_info["hostname_resolution"] = True
        except Exception:
            network_info["hostname_resolution"] = False

        # Test localhost connectivity
        try:
            import socket
            with socket.create_connection(("127.0.0.1", 22), timeout=1):
                pass
            network_info["localhost_access"] = True
        except Exception:
            network_info["localhost_access"] = False

        # Note: We don't test external access to avoid network calls in production

        self.info.capabilities["network"] = network_info

        if not network_info["hostname_resolution"]:
            self.info.restrictions.append("Network access may be restricted (hostname resolution failed)")

    def _analyze_python_environment(self):
        """Analyze Python-specific environment details."""
        python_info = {
            "version": sys.version_info,
            "executable_writable": False,
            "site_packages_writable": False,
            "virtual_env": False,
            "conda_env": False
        }

        # Check if running in virtual environment
        python_info["virtual_env"] = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

        # Check for conda environment
        python_info["conda_env"] = "conda" in sys.executable.lower() or os.getenv("CONDA_DEFAULT_ENV") is not None

        # Test if Python executable directory is writable
        try:
            exec_dir = Path(sys.executable).parent
            test_file = exec_dir / f"test_{int(time.time())}.tmp"
            test_file.write_text("test")
            test_file.unlink()
            python_info["executable_writable"] = True
        except:
            python_info["executable_writable"] = False

        # Test site-packages writability
        try:
            import site
            site_packages = site.getsitepackages()
            if site_packages:
                test_dir = Path(site_packages[0])
                test_file = test_dir / f"test_{int(time.time())}.tmp"
                test_file.write_text("test")
                test_file.unlink()
                python_info["site_packages_writable"] = True
        except:
            python_info["site_packages_writable"] = False

        self.info.capabilities["python"] = python_info

        if python_info["virtual_env"]:
            self.info.warnings.append("Running in Python virtual environment")

        if python_info["conda_env"]:
            self.info.warnings.append("Running in Conda environment")

    def _assess_risk_level(self):
        """Assess the overall risk level for file operations."""
        risk_factors = 0

        # Major risk factors
        if self.info.file_system_access.get("successful_writes", 0) == 0:
            risk_factors += 10  # Critical

        if any("CRITICAL" in restriction for restriction in self.info.restrictions):
            risk_factors += 8

        # Moderate risk factors
        if self.info.execution_context.get("containerized", False):
            risk_factors += 3

        if self.info.execution_context.get("sandboxed", False):
            risk_factors += 5

        if self.info.file_system_access.get("write_success_rate", 0) < 50:
            risk_factors += 4

        # Minor risk factors
        if len(self.info.warnings) > 3:
            risk_factors += 2

        if len(self.info.restrictions) > 5:
            risk_factors += 2

        # Determine risk level
        if risk_factors >= 10:
            self.info.risk_level = "CRITICAL"
        elif risk_factors >= 6:
            self.info.risk_level = "HIGH"
        elif risk_factors >= 3:
            self.info.risk_level = "MODERATE"
        elif risk_factors >= 1:
            self.info.risk_level = "LOW"
        else:
            self.info.risk_level = "MINIMAL"

    def _generate_summary(self) -> str:
        """Generate a human-readable summary of the environment analysis."""
        summary_lines = []

        # Risk level summary
        risk_emoji = {
            "CRITICAL": "🚨",
            "HIGH": "⚠️",
            "MODERATE": "⚡",
            "LOW": "ℹ️",
            "MINIMAL": "✅"
        }

        summary_lines.append(f"{risk_emoji.get(self.info.risk_level, '❓')} RISK LEVEL: {self.info.risk_level}")

        # Platform summary
        platform = self.info.platform_info.get("system", "unknown")
        summary_lines.append(f"Platform: {platform}")

        # Execution context summary
        context_parts = []
        if self.info.execution_context.get("containerized"):
            context_parts.append(f"{self.info.execution_context.get('container_type', 'container')}")
        if self.info.execution_context.get("sandboxed"):
            context_parts.append(f"{self.info.execution_context.get('sandbox_type', 'sandbox')}")

        if context_parts:
            summary_lines.append(f"Environment: {', '.join(context_parts)}")
        else:
            summary_lines.append("Environment: native")

        # File system summary
        fs_access = self.info.file_system_access
        writable_count = len(fs_access.get("writable_directories", []))
        total_tested = fs_access.get("total_directories_tested", 0)
        success_rate = fs_access.get("write_success_rate", 0)

        summary_lines.append(f"File System: {writable_count}/{total_tested} directories writable ({success_rate:.1f}%)")

        # Critical issues
        critical_restrictions = [r for r in self.info.restrictions if "CRITICAL" in r]
        if critical_restrictions:
            summary_lines.append(f"Critical Issues: {len(critical_restrictions)}")

        # Warnings
        if self.info.warnings:
            summary_lines.append(f"Warnings: {len(self.info.warnings)}")

        return " | ".join(summary_lines)


def detect_environment() -> Dict[str, Any]:
    """
    Perform comprehensive environment detection and analysis.

    Returns:
        Dictionary with complete environment analysis
    """
    detector = EnvironmentDetector()
    return detector.analyze_environment()


def get_environment_summary() -> str:
    """
    Get a quick summary of the current environment.

    Returns:
        Human-readable environment summary
    """
    try:
        env_info = detect_environment()
        return env_info["summary"]
    except Exception as e:
        return f"Environment detection failed: {str(e)}"


def is_environment_safe_for_file_operations() -> bool:
    """
    Quick check if the environment is safe for file operations.

    Returns:
        True if file operations are likely to work reliably
    """
    try:
        env_info = detect_environment()
        risk_level = env_info["risk_level"]
        return risk_level in ["MINIMAL", "LOW", "MODERATE"]
    except Exception as e:
        logger.error(f"Environment safety check failed: {e}")
        return False