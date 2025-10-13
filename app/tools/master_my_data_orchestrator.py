import time
import base64
from datetime import datetime, timezone
from typing import Dict, Any
from app.agents.agent_registry import AGENT_REGISTRY

TOOL_VERSION = "1.1.0"  # Version bump for base64 file handling


def _assemble_master_report(results_context: dict) -> Dict[str, Any]:
    """
    Assembles the final report from the Master My Data workflow results.
    """
    source_tagger_result = results_context.get("source_tagger", {})
    entity_resolver_result = results_context.get("entity_resolver", {})
    dedup_result = results_context.get("deduplicator", {})
    compliance_result = results_context.get("compliance_tagger", {})
    
    # Extract data from each agent's results
    all_alerts = []
    final_routing = {}
    
    for agent_name, result in results_context.items():
        if "results" in result:
            for sheet_name, sheet_data in result["results"].items():
                if "alerts" in sheet_data:
                    all_alerts.extend(sheet_data["alerts"])
                if "routing" in sheet_data:
                    final_routing = sheet_data["routing"]  # Use the last routing
    
    # Build comprehensive report
    report = {
        "source_tagging": _extract_agent_summary(source_tagger_result),
        "entity_resolution": _extract_agent_summary(entity_resolver_result),
        "deduplication": _extract_agent_summary(dedup_result),
        "compliance": _extract_agent_summary(compliance_result),
        "alerts": all_alerts,
        "routing": final_routing,
    }
    
    return report


def _extract_agent_summary(agent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts a summary from an agent's result.
    """
    if not agent_result or "results" not in agent_result:
        return {}
    
    summary = {}
    for sheet_name, sheet_data in agent_result["results"].items():
        if "data" in sheet_data:
            summary[sheet_name] = sheet_data["data"]
    
    return summary


def run_master_workflow(workflow_def: dict, files: dict, params: dict = None) -> Dict[str, Any]:
    """
    Orchestrator for Master My Data workflow.
    Handles file passing between agents and parameter injection.
    
    Args:
        workflow_def: The workflow definition from workflow_definitions.py
        files: Dictionary containing input files (e.g., {"current_file": {...}})
        params: Additional parameters (e.g., {"source_system": "CRM_Data", "key_column": "email"})
    """
    start_time = time.time()
    results_context = {}
    current_file = files.get("current_file")
    params = params or {}
    
    for step in workflow_def["steps"]:
        agent_name = step["agent_name"]
        
        # Skip optional steps if conditions aren't met
        if step.get("optional", False) and not _check_optional_requirements(step, files, params):
            continue
        
        # Look up the agent in the registry
        agent_function = AGENT_REGISTRY.get(agent_name)
        if not agent_function:
            raise ValueError(f"Agent '{agent_name}' not found in registry.")
        
        # Prepare inputs based on agent requirements
        inputs = _prepare_agent_inputs(agent_name, step, current_file, files, params, results_context)
        
        # Execute the agent
        result = agent_function(*inputs)
        results_context[agent_name] = result
        
        # If this agent passes an updated file, use it for the next agent
        if step.get("passes_file", False) and "updated_file" in result:
            updated_file_data = result["updated_file"]
            # Extract base64 contents and filename from the updated_file structure
            current_file = {
                "contents": updated_file_data.get("contents_b64", updated_file_data.get("contents")),
                "filename": updated_file_data.get("filename", current_file["filename"])
            }
        
        # Check stop conditions if defined
        if "stop_condition" in step:
            stop_check = step["stop_condition"](result)
            if stop_check["is_met"]:
                final_report = _assemble_master_report(results_context)
                final_report["routing"] = stop_check
                
                return {
                    "source_file": files["current_file"]["filename"],
                    "tool": workflow_def["name"],
                    "audit": {
                        "profile_date": datetime.now(timezone.utc).isoformat(),
                        "agent_version": TOOL_VERSION,
                        "compute_time_seconds": round(time.time() - start_time, 2),
                        "workflow_status": "Halted"
                    },
                    "report": final_report
                }
    
    # Workflow completed successfully
    final_report = _assemble_master_report(results_context)
    
    return {
        "source_file": files["current_file"]["filename"],
        "tool": workflow_def["name"],
        "audit": {
            "profile_date": datetime.now(timezone.utc).isoformat(),
            "agent_version": TOOL_VERSION,
            "compute_time_seconds": round(time.time() - start_time, 2),
            "workflow_status": "Completed"
        },
        "report": final_report,
        # Include the final updated file if available
        "updated_file": current_file if current_file != files.get("current_file") else None
    }


def _prepare_agent_inputs(agent_name: str, step: dict, current_file: dict, 
                         files: dict, params: dict, results_context: dict) -> tuple:
    """
    Prepares the input arguments for an agent based on its requirements.
    Handles both raw bytes and base64-encoded file contents.
    """
    # Decode base64 contents if necessary
    file_contents = current_file["contents"]
    if isinstance(file_contents, str):
        # If it's a base64 string, decode it
        file_contents = base64.b64decode(file_contents)
    
    if agent_name == "source_tagger":
        return (
            file_contents,
            current_file["filename"],
            params.get("source_system", "Unknown_System")
        )
    
    elif agent_name == "entity_resolver":
        return (
            file_contents,
            current_file["filename"],
            params.get("key_column", "")
        )
    
    elif agent_name in ["deduplicator", "compliance_tagger"]:
        return (
            file_contents,
            current_file["filename"]
        )
    
    else:
        # Default: pass file contents and filename
        return (file_contents, current_file["filename"])


def _check_optional_requirements(step: dict, files: dict, params: dict) -> bool:
    """
    Checks if optional step requirements are met.
    """
    # Add logic here to check if optional steps should run
    # For now, return True if all required inputs are present
    return True
