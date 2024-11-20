"""Map traversal scoring."""

from .map import Room, is_room_constrained
from .traversal_graph import Graph
import math
def score_room(_room: Room, _traversal_graph: Graph) -> float:
    """Scores a room based on how well it is laid out of a traversal graph.

    Args:
        room (Room): room to score
        traversal_graph (Graph): graph to evaluate using

    Returns:
        float: score between 0.0 and 1.0, where 1.0 is optimal
    """
    # First check if room satisfies all constraints
    if not is_room_constrained(_room):
        return 0.0

    # If no paths to evaluate, return 1.0 if room is constrained
    if not _traversal_graph.paths:
        return 1.0

    # Calculate path efficiency scores
    total_path_score = 0.0
    for path in _traversal_graph.paths:
        # Find source and target items
        source_item = next((item for item in _room.items if item.name == path.source), None)
        target_item = next((item for item in _room.items if item.name == path.target), None)
        
        if not source_item or not target_item:
            continue

        # Calculate distance between items
        distance = math.dist(
            (source_item.origin.x, source_item.origin.y),
            (target_item.origin.x, target_item.origin.y)
        )

        # Calculate max possible distance in room
        max_distance = math.dist((0, 0), (_room.bounds.x, _room.bounds.y))
        
        # Score this path (1.0 for minimal distance, approaching 0.0 for maximum distance)
        path_score = 1.0 - (distance / max_distance)
        total_path_score += path_score

    # Average path scores
    avg_path_score = total_path_score / len(_traversal_graph.paths)
 
    return avg_path_score