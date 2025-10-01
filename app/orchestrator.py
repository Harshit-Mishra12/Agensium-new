from app.agents.source import schema_scanner, dedup_agent

def workflow(dataset: list[dict], items: list[str]):
    """
    Simple orchestrator: call schema scan and dedup agents
    """
    scan_result = schema_scanner.scan_schema(dataset)
    dedup_result = dedup_agent.deduplicate(items)
    return {
        "schema_scan": scan_result,
        "deduplication": dedup_result
    }
