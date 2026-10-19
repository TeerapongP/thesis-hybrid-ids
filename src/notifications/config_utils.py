
PLACEHOLDER_MARKERS = ("changeme", "change_me", "placeholder", "your_", "xxxxx", "todo")


def looks_real(value: str | None, min_len: int = 10) -> bool:
    if not value:
        return False
    value = value.strip()
    return len(value) >= min_len and not any(marker in value.lower() for marker in PLACEHOLDER_MARKERS)
