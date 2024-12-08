"""Super Ultra Mega Project."""

class LoopLimitReachedError(RuntimeError):
    """Loop limit reached to avoid infinite unbound loops."""


def clampf(n: float, minimum: float, maximum: float) -> float:
    """Clamp float value within range.

    Args:
        n (float): value
        minimum (float): minimum
        maximum (float): maximum

    Returns:
        float: clamped value
    """
    return max(minimum, min(n, maximum))

def clampi(n: int, minimum: int, maximum: int) -> int:
    """Clamp int value within range.

    Args:
        n (int): value
        minimum (int): minimum
        maximum (int): maximum

    Returns:
        int: clamped value
    """
    return max(minimum, min(n, maximum))
