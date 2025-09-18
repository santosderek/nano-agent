"""
Self-Validating Agent System.

This module forces agents to prove their operations actually worked
by implementing mandatory external validation that cannot be faked.
"""

import os
import time
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValidationFailure(Exception):
    """Raised when external validation proves an operation didn't occur."""
    pass


class SelfValidatingAgent:
    """Agent that must prove every operation with external validation."""

    def __init__(self):
        self.validation_log = []
        self.operation_count = 0

    def external_file_exists(self, file_path: str) -> bool:
        """External validation that file exists using multiple methods."""
        path = Path(file_path).resolve()

        # Method 1: Python pathlib
        exists_pathlib = path.exists() and path.is_file()

        # Method 2: os.path
        exists_os = os.path.exists(str(path)) and os.path.isfile(str(path))

        # Method 3: subprocess ls
        try:
            result = subprocess.run(['ls', '-la', str(path)], capture_output=True, text=True, timeout=5)
            exists_subprocess = result.returncode == 0
        except:
            exists_subprocess = False

        # All methods must agree
        consensus = exists_pathlib and exists_os and exists_subprocess

        logger.info(f"File existence validation for {file_path}: pathlib={exists_pathlib}, os={exists_os}, subprocess={exists_subprocess}, consensus={consensus}")

        return consensus

    def external_file_content_matches(self, file_path: str, expected_content: str) -> bool:
        """External validation that file content matches expected."""
        if not self.external_file_exists(file_path):
            return False

        path = Path(file_path)

        try:
            # Method 1: Python read
            content_python = path.read_text(encoding='utf-8')

            # Method 2: subprocess cat
            result = subprocess.run(['cat', str(path)], capture_output=True, text=True, timeout=10)
            content_subprocess = result.stdout if result.returncode == 0 else None

            # Method 3: File size check
            expected_size = len(expected_content.encode('utf-8'))
            actual_size = path.stat().st_size
            size_matches = expected_size == actual_size

            # Method 4: Hash verification
            expected_hash = hashlib.md5(expected_content.encode('utf-8')).hexdigest()
            actual_hash = hashlib.md5(content_python.encode('utf-8')).hexdigest()
            hash_matches = expected_hash == actual_hash

            # All methods must agree
            content_matches = (
                content_python == expected_content and
                content_subprocess == expected_content and
                size_matches and
                hash_matches
            )

            logger.info(f"Content validation for {file_path}: python_match={content_python == expected_content}, subprocess_match={content_subprocess == expected_content}, size_match={size_matches}, hash_match={hash_matches}, consensus={content_matches}")

            return content_matches

        except Exception as e:
            logger.error(f"Content validation failed for {file_path}: {e}")
            return False

    def external_directory_exists(self, dir_path: str) -> bool:
        """External validation that directory exists."""
        path = Path(dir_path).resolve()

        # Method 1: Python pathlib
        exists_pathlib = path.exists() and path.is_dir()

        # Method 2: os.path
        exists_os = os.path.exists(str(path)) and os.path.isdir(str(path))

        # Method 3: subprocess ls -d
        try:
            result = subprocess.run(['ls', '-d', str(path)], capture_output=True, text=True, timeout=5)
            exists_subprocess = result.returncode == 0
        except:
            exists_subprocess = False

        consensus = exists_pathlib and exists_os and exists_subprocess

        logger.info(f"Directory existence validation for {dir_path}: pathlib={exists_pathlib}, os={exists_os}, subprocess={exists_subprocess}, consensus={consensus}")

        return consensus

    def force_external_validation(self, operation_type: str, **kwargs) -> Dict[str, Any]:
        """Force external validation of an operation - cannot be faked."""
        self.operation_count += 1
        validation_id = f"validation_{self.operation_count}_{int(time.time())}"

        logger.info(f"🔍 EXTERNAL VALIDATION #{self.operation_count}: {operation_type}")

        validation_result = {
            "validation_id": validation_id,
            "operation_type": operation_type,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "details": {},
            "errors": []
        }

        try:
            if operation_type == "file_write":
                file_path = kwargs.get("file_path")
                expected_content = kwargs.get("content")

                if not file_path or expected_content is None:
                    validation_result["errors"].append("Missing file_path or content for validation")
                    return validation_result

                # External validation
                file_exists = self.external_file_exists(file_path)
                content_matches = self.external_file_content_matches(file_path, expected_content) if file_exists else False

                validation_result["success"] = file_exists and content_matches
                validation_result["details"] = {
                    "file_path": file_path,
                    "file_exists": file_exists,
                    "content_matches": content_matches,
                    "expected_length": len(expected_content),
                    "actual_length": len(Path(file_path).read_text()) if file_exists else 0
                }

            elif operation_type == "file_read":
                file_path = kwargs.get("file_path")

                if not file_path:
                    validation_result["errors"].append("Missing file_path for validation")
                    return validation_result

                file_exists = self.external_file_exists(file_path)
                validation_result["success"] = file_exists
                validation_result["details"] = {
                    "file_path": file_path,
                    "file_exists": file_exists,
                    "file_size": Path(file_path).stat().st_size if file_exists else 0
                }

            elif operation_type == "directory_create":
                dir_path = kwargs.get("dir_path")

                if not dir_path:
                    validation_result["errors"].append("Missing dir_path for validation")
                    return validation_result

                dir_exists = self.external_directory_exists(dir_path)
                validation_result["success"] = dir_exists
                validation_result["details"] = {
                    "dir_path": dir_path,
                    "dir_exists": dir_exists
                }

            else:
                validation_result["errors"].append(f"Unknown operation type: {operation_type}")
                return validation_result

        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            logger.error(f"External validation failed for {operation_type}: {e}")

        # Log the validation result
        self.validation_log.append(validation_result)

        if validation_result["success"]:
            logger.info(f"✅ EXTERNAL VALIDATION PASSED: {operation_type}")
        else:
            logger.error(f"❌ EXTERNAL VALIDATION FAILED: {operation_type} - {validation_result['errors']}")

        return validation_result

    async def execute_with_forced_validation(self, prompt: str, expected_operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute an agent task with mandatory external validation.

        Args:
            prompt: The task prompt
            expected_operations: List of operations to validate, each with 'type' and parameters

        Returns:
            Results with validation proof
        """
        start_time = time.time()

        logger.info(f"🚀 EXECUTING WITH FORCED VALIDATION")
        logger.info(f"Expected operations: {len(expected_operations)}")

        # Import and execute the nano agent
        try:
            from ..nano_agent import prompt_nano_agent

            # Execute the agent
            logger.info("Executing nano agent...")
            agent_result = await prompt_nano_agent(prompt)

            # Now FORCE external validation of expected operations
            validation_results = []

            for i, expected_op in enumerate(expected_operations):
                logger.info(f"🔍 Validating operation {i+1}/{len(expected_operations)}: {expected_op['type']}")

                validation = self.force_external_validation(
                    operation_type=expected_op['type'],
                    **expected_op.get('params', {})
                )

                validation_results.append(validation)

                if not validation['success']:
                    logger.error(f"❌ OPERATION {i+1} FAILED VALIDATION: {expected_op['type']}")
                    # Continue with other validations to get complete picture

            # Calculate validation summary
            passed_validations = sum(1 for v in validation_results if v['success'])
            total_validations = len(validation_results)
            validation_rate = (passed_validations / total_validations * 100) if total_validations > 0 else 0

            execution_time = time.time() - start_time

            # Determine overall success
            all_validations_passed = passed_validations == total_validations

            result = {
                "success": all_validations_passed,
                "agent_reported_success": agent_result.get("success", False),
                "validation_summary": {
                    "total_operations": total_validations,
                    "passed_validations": passed_validations,
                    "failed_validations": total_validations - passed_validations,
                    "validation_rate": validation_rate
                },
                "validation_details": validation_results,
                "agent_result": agent_result,
                "execution_time_seconds": execution_time,
                "external_validation_forced": True
            }

            if all_validations_passed:
                result["message"] = f"✅ ALL EXTERNAL VALIDATIONS PASSED ({passed_validations}/{total_validations})"
                logger.info(f"🎉 ALL VALIDATIONS PASSED: {passed_validations}/{total_validations}")
            else:
                result["message"] = f"❌ EXTERNAL VALIDATIONS FAILED ({passed_validations}/{total_validations})"
                result["error"] = f"Agent reported success but {total_validations - passed_validations} operations failed external validation"
                logger.error(f"🚨 VALIDATIONS FAILED: {passed_validations}/{total_validations}")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Self-validating execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return {
                "success": False,
                "error": error_msg,
                "execution_time_seconds": execution_time,
                "external_validation_forced": True,
                "validation_system_error": True
            }


# Global self-validating agent
_self_validating_agent = SelfValidatingAgent()


async def execute_with_proof(prompt: str, expected_files: List[str] = None, expected_dirs: List[str] = None) -> Dict[str, Any]:
    """
    Execute a nano agent task and FORCE external proof that operations occurred.

    Args:
        prompt: Task description
        expected_files: List of file paths that should be created/modified
        expected_dirs: List of directory paths that should be created

    Returns:
        Results with mandatory external validation proof
    """
    expected_operations = []

    # Convert expected files to validation operations
    if expected_files:
        for file_path in expected_files:
            expected_operations.append({
                "type": "file_write",  # We'll validate the file exists
                "params": {"file_path": file_path, "content": ""}  # Content will be read externally
            })

    # Convert expected directories to validation operations
    if expected_dirs:
        for dir_path in expected_dirs:
            expected_operations.append({
                "type": "directory_create",
                "params": {"dir_path": dir_path}
            })

    return await _self_validating_agent.execute_with_forced_validation(prompt, expected_operations)