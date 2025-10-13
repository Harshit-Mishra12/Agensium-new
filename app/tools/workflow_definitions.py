from app.config import AGENT_ROUTES

# --- Stop Conditions: These are the "brains" of the smart orchestration ---
# The engine will execute these functions to decide whether to continue the workflow.

def check_readiness_score(result):
    """Stop the workflow if the data quality is too low for further analysis."""
    main_key = list(result.get("results", {}).keys())[0]
    score = result["results"][main_key]["data"]["readiness_score"]["overall"]
    if score < 70:
        # If the score is low, we return a reason and a suggested next step.
        return {
            "is_met": True,
            "reason": f"Workflow stopped: Overall readiness score ({score}) is below the threshold of 70.",
            "suggestion": "Data quality is low. Run the 'Clean My Data' tool before proceeding with a full profile.",
            "suggested_agent_endpoint": AGENT_ROUTES.get('clean_data_tool')
        }
    return {"is_met": False}


# --- Workflow Definitions ---
# Each tool is defined as a sequence of steps. This makes creating new tools easy.

PROFILE_MY_DATA_WORKFLOW = {
    "name": "ProfileMyData",
    "steps": [
        {
            "agent_name": "readiness_rater",
            "inputs": {"file": "current_file"}, # Use the main file uploaded by the user
            "stop_condition": check_readiness_score # Check the score before continuing
        },
        {
            "agent_name": "schema_scanner",
            "inputs": {"file": "current_file"}
        },
        {
            "agent_name": "field_profiler",
            "inputs": {"file": "current_file"}
        },
        {
            "agent_name": "drift_detector",
            "inputs": {"baseline_file": "baseline_file", "current_file": "current_file"},
            "optional": True # This step only runs if a baseline file is provided
        }
    ]
}

# --- Master My Data Workflow ---
# This workflow establishes data lineage, resolves entities, deduplicates, and checks compliance.

MASTER_MY_DATA_WORKFLOW = {
    "name": "MasterMyData",
    "steps": [
        {
            "agent_name": "source_tagger",
            "inputs": {"file": "current_file", "source_system": "source_system"},
            "passes_file": True  # This agent returns an updated file for the next agent
        },
        {
            "agent_name": "entity_resolver",
            "inputs": {"file": "updated_file", "key_column": "key_column"},
            "passes_file": True,  # Pass the file to the next agent
            "optional": False
        },
        {
            "agent_name": "deduplicator",
            "inputs": {"file": "current_file"},
            "passes_file": True  # Pass the file to the next agent
        },
        {
            "agent_name": "compliance_tagger",
            "inputs": {"file": "current_file"},
            "passes_file": False  # Last agent, no need to pass file
        }
    ]
}
