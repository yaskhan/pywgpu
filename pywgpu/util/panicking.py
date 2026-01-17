# Panic handling utilities

def panic(message: str):
    """Raises a runtime error simulating a panic."""
    raise RuntimeError(f"Panic: {message}")
