AGENT_ROUTES = {
    # Individual Agent Endpoints
    "unified_profiler": "/unified-profiler",
    "rate_readiness": "/rate-readiness",
    "score_risk": "/score-risk",
    "detect_drift": "/detect-drift",
    "deduplicate": "/deduplicate",
    "governance_checker": "/governance-checker",
    "test_coverage_agent": "/test-coverage",
    "schema_drift_reporter": "/schema-drift",
    "null_handler": "/null-handler",
    "outlier_remover": "/outlier-remover",
    "type_fixer": "/type-fixer",
    # "check_governance": "/check-governance",
    # "readiness_rater": "/rate-readiness",

    # Tool / Orchestrator Endpoints
    'profile_my_data_tool': "/run-tool/profile-my-data",
    'master_my_data_tool': "/run-tool/master-my-data",
    "clean_data_tool": "/run-tool/cleaner",
    'define_data_tool': "/run-tool/definer",
    "govern_data_tool": "/run-tool/governance",
    # Add other tool endpoints here, e.g., "profile_my_data_tool": "/profile-my-data"

    # Shared Agent Endpoints
    "chat_with_data": "/agents/chat-with-data",
    
}
