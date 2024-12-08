"""Super Ultra Mega Project."""

class LoopLimitReachedError(RuntimeError):
    """Loop limit reached to avoid infinite unbound loops."""


def clamp(n: float, minimum: float, maximum: float) -> float:
    """Clamp value within range.

    Args:
        n (float): value
        minimum (float): minimum
        maximum (float): maximum

    Returns:
        float: clamped value
    """
    return max(minimum, min(n, maximum))
