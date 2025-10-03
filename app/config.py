AGENT_ROUTES = {
    # Individual Agent Endpoints
    "scan_schema": "/scan-schema",
    "profile_fields": "/profile-fields",
    "rate_readiness": "/rate-readiness",
    "detect_drift": "/detect-drift",

    # Tool / Orchestrator Endpoints
    "clean_data_tool": "/run-tool/cleaner",
    'define_data_tool': "/run-tool/definer",
    # Add other tool endpoints here, e.g., "profile_my_data_tool": "/profile-my-data"
}
