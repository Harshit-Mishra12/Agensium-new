def deduplicate(items: list[str]):
    """
    Example agent: remove duplicates
    """
    unique_items = list(dict.fromkeys(items))
    return {
        "original_count": len(items),
        "unique_count": len(unique_items),
        "items": unique_items
    }
