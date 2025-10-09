from app.agents.source import schema_scanner, readiness_rater, field_profiler, drift_detector, dedup_agent
from app.agents.shared import chat_agent

# This is the central registry for all agents in the system.
# The key is a unique agent identifier, and the value is the callable function.
# This makes the orchestration engine scalable and pluggable.
AGENT_REGISTRY = {
    "schema_scanner": schema_scanner.scan_schema,
    "readiness_rater": readiness_rater.rate_readiness,
    "field_profiler": field_profiler.profile_fields,
    "drift_detector": drift_detector.detect_drift,
    "deduplicator": dedup_agent.deduplicate,
    # As you build new agents (e.g., clean_data), you will register them here.
    # "clean_data_agent": clean_data.run,
        # Shared/Utility Agents
    "chat_agent": chat_agent.answer_question_on_report
}
