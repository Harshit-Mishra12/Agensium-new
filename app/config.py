AGENT_ROUTES = {
    # Individual Agent Endpoints
    "scan_schema": "/scan-schema",
    "profile_fields": "/profile-fields",
    "unified_profiler": "/unified-profiler",
    "rate_readiness": "/rate-readiness",
    "score_risk": "/score-risk",
    "detect_drift": "/detect-drift",
    "source_tagger": "/source-tagger",
    "entity_resolver": "/entity-resolver",
    "deduplicate": "/deduplicate",
    "compliance_tagger": "/compliance-tagger",

    # Tool / Orchestrator Endpoints
    'profile_my_data_tool': "/run-tool/profile-my-data",
    'master_my_data_tool': "/run-tool/master-my-data",
    "clean_data_tool": "/run-tool/cleaner",
    'define_data_tool': "/run-tool/definer",
    "govern_data_tool": "/run-tool/governance",
    "readiness_rater": "/rate-readiness",
    # Add other tool endpoints here, e.g., "profile_my_data_tool": "/profile-my-data"

    # Shared Agent Endpoints
    "chat_with_data": "/agents/chat-with-data",
    
}
