from .project_manager import ProjectManager
import json
import time
import os
from datetime import datetime
import traceback

class payloadOperations:
    def __init__(self, use_redis=True, redis_url=None, redis_host='localhost', redis_port=6379,
                 redis_db=0, redis_password=None, key_prefix='token_bucket', 
                 router_projects: list[dict[str,any]] = None, debug_mode=False, debug_file_path="debug_logs.json"):
        """
        Initialize the payload operations with router settings.
        
        :param use_redis: Whether to use Redis for token bucket management.
        :param redis_host: Redis server host.
        :param redis_port: Redis server port.
        :param redis_db: Redis database number.
        :param key_prefix: Prefix for Redis keys.
        :param router_projects: List of project dictionaries with project_id.
        :param debug_mode: Enable debugging to log operations to JSON file.
        :param debug_file_path: Path to the debug JSON file.
        """
        self.use_redis = use_redis
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.key_prefix = key_prefix
        self.debug_mode = debug_mode
        self.debug_file_path = debug_file_path
        
        # Initialize debug log structure
        self.debug_log = {
            "session_info": {
                "start_time": datetime.now().isoformat(),
                "configuration": {
                    "use_redis": use_redis,
                    "redis_host": redis_host,
                    "redis_port": redis_port,
                    "redis_db": redis_db,
                    "key_prefix": key_prefix
                }
            },
            "operations": [],
            "errors": [],
            "system_snapshots": []
        }
        
        self.router = ProjectManager(
            use_redis=self.use_redis,
            redis_url=redis_url,
            redis_host=self.redis_host,
            redis_port=self.redis_port,
            redis_db=self.redis_db,
            redis_password=redis_password,
            key_prefix=self.key_prefix
        )
        if router_projects:
            self.router_projects = [{"project_id": project["project_id"], "index": i} for i, project in enumerate(router_projects)]
            for project in self.router_projects:
                self.router.add_project(project["index"], project["project_id"]) #adding the projects to the router 
                
            # Log initial project setup
            if self.debug_mode:
                self._log_debug_info("initialization", {
                    "action": "projects_added",
                    "projects": self.router_projects,
                    "total_projects": len(self.router_projects)
                })
        else:
            self.router_projects = []
            
        # Take initial system snapshot
        if self.debug_mode:
            self._take_system_snapshot("initialization")

    def _log_debug_info(self, operation_type: str, data: dict, error: bool = False):
        """
        Log debug information to the internal debug log structure.
        
        :param operation_type: Type of operation (e.g., 'router_allocation', 'token_refund')
        :param data: Data to log
        :param error: Whether this is an error log
        """
        if not self.debug_mode:
            return
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "data": data
        }
        
        if error:
            # Add stack trace for errors
            log_entry["stack_trace"] = traceback.format_exc()
            self.debug_log["errors"].append(log_entry)
        else:
            self.debug_log["operations"].append(log_entry)

    def _take_system_snapshot(self, snapshot_reason: str):
        """
        Take a snapshot of the current system state.
        
        :param snapshot_reason: Reason for taking the snapshot
        """
        if not self.debug_mode:
            return
            
        try:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "reason": snapshot_reason,
                "project_info": self.router.get_project_info(),
                "all_balances": self.router.get_project_balances(),
                "current_project": self.router.get_current_project_index(),
                "total_projects": len(self.router_projects)
            }
            
            self.debug_log["system_snapshots"].append(snapshot)
            
        except Exception as e:
            self._log_debug_info("system_snapshot_error", {
                "reason": snapshot_reason,
                "error": str(e)
            }, error=True)

    def main_router(self, input_tokens: int):
        """
        Route the payload to a region based on the input tokens.
        :param input_tokens: Number of maximum tokens required for the payload.
        """
        operation_start = time.time()
        
        try:
            payload_tokens = input_tokens + 8_192  # Add 8192 tokens for max output
            
            # Log the routing attempt
            if self.debug_mode:
                self._log_debug_info("router_allocation_attempt", {
                    "input_tokens": input_tokens,
                    "payload_tokens": payload_tokens,
                    "buffer_tokens": 8_192
                })
            
            router_result = self.router.pick_region_with_fallback(tokens_needed=payload_tokens)
            
            operation_duration = time.time() - operation_start
            
            if router_result is None:
                # Log the failure
                if self.debug_mode:
                    self._log_debug_info("router_allocation_failed", {
                        "input_tokens": input_tokens,
                        "payload_tokens": payload_tokens,
                        "operation_duration_ms": round(operation_duration * 1000, 2),
                        "available_balances": self.router.get_project_balances()
                    })
                    self._take_system_snapshot("allocation_failure")
                
                raise Exception("No available region with sufficient tokens.")
            else:
                # Log successful allocation
                if self.debug_mode:
                    self._log_debug_info("router_allocation_success", {
                        "input_tokens": input_tokens,
                        "payload_tokens": payload_tokens,
                        "allocated_project": router_result['project_name'],
                        "allocated_region": router_result['region'],
                        "project_index": router_result['project_index'],
                        "tokens_allocated": router_result['tokens_allocated'],
                        "operation_duration_ms": round(operation_duration * 1000, 2)
                    })
                
                return router_result['project_name'], router_result['region']
                
        except Exception as e:
            operation_duration = time.time() - operation_start
            
            if self.debug_mode:
                self._log_debug_info("router_allocation_error", {
                    "input_tokens": input_tokens,
                    "error_message": str(e),
                    "operation_duration_ms": round(operation_duration * 1000, 2)
                }, error=True)
                self._take_system_snapshot("allocation_error")
            
            raise

    def input_refund(self, input_tokens: int, project_name: str, region: str):
        """
        Refund unused input tokens back to the specified project and region.
        :param input_tokens: Number of input tokens actually used.
        :param project_name: Name of the project to refund tokens to.
        :param region: Region to refund tokens to.
        """
        operation_start = time.time()

        project_index = {item["project_id"]: item["index"] for item in self.router_projects}[project_name]

    # Log refund attempt
        if self.debug_mode:
            self._log_debug_info("input_token_refund_attempt", {
                "input_tokens_used": input_tokens,
                "max_output_tokens": 8_192,
                "project_name": project_name,
                "project_index": project_index,
                "region": region
            })
        
        # Get balance before refund for comparison
        balance_before = None
        if self.debug_mode:
            project_balances = self.router.get_project_balances(project_index)
            if project_name in project_balances and region in project_balances[project_name]:
                balance_before = project_balances[project_name][region]

        refund_success = self.router.refund_tokens_to_project(project_index, region, input_tokens)

        operation_duration = time.time() - operation_start
            
        # Get balance after refund
        balance_after = None
        if self.debug_mode:
            project_balances = self.router.get_project_balances(project_index)
            if project_name in project_balances and region in project_balances[project_name]:
                balance_after = project_balances[project_name][region]

        # Log refund result
        if self.debug_mode:
            self._log_debug_info("input_token_refund_complete", {
                "refund_success": refund_success,
                "input_tokens_used": input_tokens,
                "project_name": project_name,
                "region": region,
                "balance_before": balance_before,
                "balance_after": balance_after,
                "balance_change": balance_after - balance_before if (balance_before is not None and balance_after is not None) else None,
                "operation_duration_ms": round(operation_duration * 1000, 2)
            })

        return refund_success

    def output_calc(self, output_tokens: int, project_name: str, region: str):
        """
        Calculate the total tokens required for the payload and refund unused tokens.
        :param output_tokens: Number of output tokens actually used.
        :param project_name: Name of the project to refund tokens to.
        :param region: Region to refund tokens to.
        """
        operation_start = time.time()
        
        try:
            total_tokens = 8_192 - output_tokens # Subtract output tokens from max output tokens then adding the difference to the region balance
            project_index = {item["project_id"]: item["index"] for item in self.router_projects}[project_name]
            
            # Log refund attempt
            if self.debug_mode:
                self._log_debug_info("token_refund_attempt", {
                    "output_tokens_used": output_tokens,
                    "max_output_tokens": 8_192,
                    "tokens_to_refund": total_tokens,
                    "project_name": project_name,
                    "project_index": project_index,
                    "region": region
                })
            
            # Get balance before refund for comparison
            balance_before = None
            if self.debug_mode:
                project_balances = self.router.get_project_balances(project_index)
                if project_name in project_balances and region in project_balances[project_name]:
                    balance_before = project_balances[project_name][region]
            
            # Perform the refund
            refund_success = self.router.refund_tokens_to_project(project_index, region, total_tokens)
            
            operation_duration = time.time() - operation_start
            
            # Get balance after refund
            balance_after = None
            if self.debug_mode:
                project_balances = self.router.get_project_balances(project_index)
                if project_name in project_balances and region in project_balances[project_name]:
                    balance_after = project_balances[project_name][region]
            
            # Log refund result
            if self.debug_mode:
                self._log_debug_info("token_refund_complete", {
                    "refund_success": refund_success,
                    "output_tokens_used": output_tokens,
                    "tokens_refunded": total_tokens,
                    "project_name": project_name,
                    "region": region,
                    "balance_before": balance_before,
                    "balance_after": balance_after,
                    "balance_change": balance_after - balance_before if (balance_before is not None and balance_after is not None) else None,
                    "operation_duration_ms": round(operation_duration * 1000, 2)
                })
            
            return refund_success
            
        except Exception as e:
            operation_duration = time.time() - operation_start
            
            if self.debug_mode:
                self._log_debug_info("token_refund_error", {
                    "output_tokens": output_tokens,
                    "project_name": project_name,
                    "region": region,
                    "error_message": str(e),
                    "operation_duration_ms": round(operation_duration * 1000, 2)
                }, error=True)
            
            raise

    def save_debug_log(self, file_path: str = None):
        """
        Save the debug log to a JSON file.
        
        :param file_path: Optional custom file path. If None, uses the default debug_file_path.
        """
        if not self.debug_mode:
            print("Debug mode is not enabled. No logs to save.")
            return False
            
        target_path = file_path or self.debug_file_path
        
        try:
            # Add session end info
            self.debug_log["session_info"]["end_time"] = datetime.now().isoformat()
            self.debug_log["session_info"]["total_operations"] = len(self.debug_log["operations"])
            self.debug_log["session_info"]["total_errors"] = len(self.debug_log["errors"])
            self.debug_log["session_info"]["total_snapshots"] = len(self.debug_log["system_snapshots"])
            
            # Take final snapshot
            self._take_system_snapshot("session_end")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            
            # Write to file
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(self.debug_log, f, indent=2, ensure_ascii=False)
                
            print(f"Debug log saved to: {target_path}")
            print(f"Operations logged: {len(self.debug_log['operations'])}")
            print(f"Errors logged: {len(self.debug_log['errors'])}")
            print(f"System snapshots: {len(self.debug_log['system_snapshots'])}")
            
            return True
            
        except Exception as e:
            print(f"Failed to save debug log: {str(e)}")
            return False

    def get_debug_summary(self):
        """
        Get a summary of the debug information without saving to file.
        """
        if not self.debug_mode:
            return {"error": "Debug mode is not enabled"}
            
        return {
            "session_duration": self._calculate_session_duration(),
            "total_operations": len(self.debug_log["operations"]),
            "total_errors": len(self.debug_log["errors"]),
            "total_snapshots": len(self.debug_log["system_snapshots"]),
            "operation_types": self._get_operation_type_counts(),
            "current_balances": self.router.get_project_balances(),
            "project_info": self.router.get_project_info()
        }

    def _calculate_session_duration(self):
        """Calculate session duration in seconds."""
        try:
            start = datetime.fromisoformat(self.debug_log["session_info"]["start_time"])
            end = datetime.now()
            return (end - start).total_seconds()
        except:
            return None

    def _get_operation_type_counts(self):
        """Get counts of different operation types."""
        counts = {}
        for op in self.debug_log["operations"]:
            op_type = op.get("operation_type", "unknown")
            counts[op_type] = counts.get(op_type, 0) + 1
        return counts
    
    def retry_with_next_project(self, project_name: str, region: str, input_tokens: int):
        """
        Mark the given project+region as exhausted, then retry routing with a different project.

        :param project_name: Name of the exhausted project
        :param region: Region to mark as exhausted for the project
        :param input_tokens: Number of tokens required
        :return: Tuple (new_project_name, new_region) if successful, else None
        """
        try:
            # 1. Find project index
            project_index_map = {item["project_id"]: item["index"] for item in self.router_projects}
            if project_name not in project_index_map:
                raise ValueError(f"Project '{project_name}' not found in router_projects.")
            
            project_index = project_index_map[project_name]

            # 2. Mark the region as exhausted for this project
            self.router.mark_region_exhausted(project_index, region)

            # 3. Retry routing, skipping the exhausted project index
            payload_tokens = input_tokens + 8_192  # same buffer logic
            router_result = self.router.pick_region_with_fallback(tokens_needed=payload_tokens, exclude_indices={project_index})

            if not router_result:
                if self.debug_mode:
                    self._log_debug_info("retry_allocation_failed", {
                        "exhausted_project": project_name,
                        "exhausted_region": region,
                        "input_tokens": input_tokens,
                        "payload_tokens": payload_tokens
                    })
                return None

            # 4. Log retry success
            if self.debug_mode:
                self._log_debug_info("retry_allocation_success", {
                    "previous_project": project_name,
                    "previous_region": region,
                    "new_project": router_result['project_name'],
                    "new_region": router_result['region'],
                    "project_index": router_result['project_index']
                })

            return router_result['project_name'], router_result['region']

        except Exception as e:
            if self.debug_mode:
                self._log_debug_info("retry_allocation_error", {
                    "project_name": project_name,
                    "region": region,
                    "input_tokens": input_tokens,
                    "error_message": str(e)
                }, error=True)
            raise


    def close(self):
        """
        Close the router and optionally save debug logs.
        """
        if self.debug_mode:
            self.save_debug_log()
            
        self.router.close()


