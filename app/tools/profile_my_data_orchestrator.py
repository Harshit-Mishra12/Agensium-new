import time
from datetime import datetime, timezone
from app.agents.agent_registry import AGENT_REGISTRY
from app.tools.workflow_definitions import PROFILE_MY_DATA_WORKFLOW

TOOL_VERSION = "2.0.0" # Bumping version for the new smart orchestration

def _assemble_final_report(results_context: dict):
    """
    Assembles the final report from the context of a completed workflow.
    """
    # This function is similar to before, but now it works with the generic results context.
    readiness_result = results_context.get("readiness_rater", {})
    schema_result = results_context.get("schema_scanner", {})
    profiler_result = results_context.get("field_profiler", {})
    drift_result = results_context.get("drift_detector")

    main_key = list(readiness_result.get("results", {}).keys())[0]
    
    readiness_data = readiness_result.get("results", {}).get(main_key, {})
    schema_data = schema_result.get("results", {}).get(main_key, {})
    profiler_data = profiler_result.get("results", {}).get(main_key, {})
    
    drift_analysis = {}
    if drift_result:
        drift_main_key = list(drift_result.get("results", {}).keys())[0]
        drift_data = drift_result.get("results", {}).get(drift_main_key, {})
        drift_analysis = drift_data.get("data", {})

    all_alerts = (
        schema_data.get("alerts", []) +
        readiness_data.get("alerts", []) +
        profiler_data.get("alerts", []) +
        (drift_data.get("alerts", []) if drift_result else [])
    )

    final_report = {
        "readiness_score": readiness_data.get("data", {}).get("readiness_score"),
        "schema_summary": schema_data.get("data", {}).get("summary_table"),
        "field_stats": profiler_data.get("data", {}).get("columns"),
        "drift_analysis": drift_analysis,
        "alerts": all_alerts,
        "routing": readiness_data.get("routing", {}),
    }
    
    return final_report

def run_workflow(workflow_def: dict, files: dict):
    """
    This is the generic, scalable Workflow Engine.
    It can execute any workflow defined in workflow_definitions.py.
    """
    start_time = time.time()
    results_context = {} # A dictionary to store the results of each agent run
    
    for step in workflow_def["steps"]:
        agent_name = step["agent_name"]
        
        # Skip optional steps if the required files are not provided
        if step.get("optional", False) and not files.get("baseline_file"):
            continue

        # 1. Look up the agent in the registry
        agent_function = AGENT_REGISTRY.get(agent_name)
        if not agent_function:
            raise ValueError(f"Agent '{agent_name}' not found in registry.")

        # 2. Prepare inputs for the agent
        # This is a simple implementation; a real-world version might be more complex
        if agent_name == "drift_detector":
            inputs = (files["baseline_file"]["contents"], files["current_file"]["contents"], 
                      files["baseline_file"]["filename"], files["current_file"]["filename"])
        else:
            inputs = (files["current_file"]["contents"], files["current_file"]["filename"])

        # 3. Execute the agent
        result = agent_function(*inputs)
        results_context[agent_name] = result
        
        # 4. Check the stop condition (the "smart" part)
        if "stop_condition" in step:
            stop_check = step["stop_condition"](result)
            if stop_check["is_met"]:
                # If the condition is met, stop the workflow and return the partial results
                # along with the reason for stopping.
                final_report = _assemble_final_report(results_context)
                final_report["routing"] = stop_check # Overwrite routing with the stop reason
                final_report["audit"] = {
                    "profile_date": datetime.now(timezone.utc).isoformat(),
                    "agent_version": TOOL_VERSION,
                    "compute_time_seconds": round(time.time() - start_time, 2),
                    "workflow_status": "Halted"
                }
                return {
                    "source_file": files["current_file"]["filename"],
                    "tool": workflow_def["name"],
                    "report": final_report
                }

    # If the workflow completes without stopping, assemble the full report
    final_report = _assemble_final_report(results_context)
    final_report["audit"] = {
        "profile_date": datetime.now(timezone.utc).isoformat(),
        "agent_version": TOOL_VERSION,
        "compute_time_seconds": round(time.time() - start_time, 2),
        "workflow_status": "Completed"
    }

    return {
        "source_file": files["current_file"]["filename"],
        "tool": workflow_def["name"],
        "report": final_report
    }

