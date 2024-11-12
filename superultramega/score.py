"""Map traversal scoring."""

from .map import Room
from .traversal_graph import Graph


def score_room(_room: Room, _traversal_graph: Graph) -> float:
    """Scores a room based on how well it is laid out of a traversal graph.

    Args:
        room (Room): room to score
        traversal_graph (Graph): graph to evaluate using

    Returns:
        float: score
    """
    return 0.0
