from app.agents.source import readiness_rater, drift_detector, governance_checker, test_coverage_agent, schema_drift_reporter
from app.agents.clean import null_handler, type_fixer
from app.agents.shared import chat_agent

# This is the central registry for all agents in the system.
# The key is a unique agent identifier, and the value is the callable function.
# This makes the orchestration engine scalable and pluggable.
AGENT_REGISTRY = {
    "readiness_rater": readiness_rater.rate_readiness,
    "drift_detector": drift_detector.detect_drift,
    "governance_checker": governance_checker.check_governance,
    "test_coverage_agent": test_coverage_agent.check_test_coverage,
    "schema_drift_reporter": schema_drift_reporter.report_schema_drift,
    "null_handler": null_handler.handle_nulls,
    "type_fixer": type_fixer.fix_types,
    # As you build new agents (e.g., clean_data), you will register them here.
    # "clean_data_agent": clean_data.run,
    # Shared/Utility Agents
    "chat_agent": chat_agent.answer_question_on_report
}
