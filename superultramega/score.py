"""Map traversal scoring."""

import heapq
import math
from typing import Final

from .map import Item, Orientation, Room
from .traversal_graph import Graph

VOXEL_RESOLUTION: Final[float] = 2**-4


class Cell:
    """Cell type in the grid."""

    position: tuple[int, int]
    g_cost: float
    h_cost: float
    f_cost: float
    parent: "Cell | None"

    def __init__(
        self,
        position: tuple[int, int],
        g_cost: float = 0.0,
        h_cost: float = 0.0,
        parent: "Cell | None" = None,
    ) -> None:
        """Create new node.

        Args:
            position (tuple[int, int]): position
            g_cost (float, optional): g cost. Defaults to 0.0.
            h_cost (float, optional): h cost. Defaults to 0.0.
            parent (Node | None, optional): parent node. Defaults to None.
        """
        self.position = position  # (x, y)
        self.g_cost = g_cost  # The cost from the start to this node
        self.h_cost = h_cost  # The heuristic cost to the goal
        self.f_cost = g_cost + h_cost  # f = g + h
        self.parent = parent  # The parent node used to trace the path

    def __lt__(self, other: "Cell") -> bool:
        """Less than implementation.

        Args:
            other (Cell): other cell to compare

        Returns:
            bool: if current cell has lower cost than other cell
        """
        return self.f_cost < other.f_cost


def heuristic(source: tuple[int, int], target: tuple[int, int]) -> float:
    """Manhattan distance heuristic function.

    Args:
        source (tuple[int, int]): start
        target (tuple[int, int]): end

    Returns:
        float: Manhattan distance
    """
    return abs(source[0] - target[0]) + abs(source[1] - target[1])


def a_star(start: tuple[int, int], goal: tuple[int, int], grid: list[list[bool]]) -> list[tuple[int, int]]:
    """Run a_star search.

    Args:
        start (tuple[int, int]): start position
        goal (tuple[int, int]): end position
        grid (list[list[bool]]): grid map

    Returns:
        list[tuple[int, int]]: list of voxels for traversal
    """
    open_list: list[Cell] = []
    closed_list: set[tuple[int, int]] = set()

    # Start node and goal node
    start_node = Cell(start, 0.0, heuristic(start, goal))
    goal_node = Cell(goal)

    heapq.heappush(open_list, start_node)

    while open_list:
        # Get the node with the lowest f_cost
        current_node = heapq.heappop(open_list)

        # If the current node is the goal, reconstruct the path
        if current_node.position == goal_node.position:
            path: list[tuple[int, int]] = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()
            return path

        closed_list.add(current_node.position)

        # Generate neighbors (4 possible directions)
        neighbors: list[tuple[int, int]] = [
            (current_node.position[0] + 1, current_node.position[1]),
            (current_node.position[0] - 1, current_node.position[1]),
            (current_node.position[0], current_node.position[1] + 1),
            (current_node.position[0], current_node.position[1] - 1),
        ]

        for neighbor in neighbors:
            if not (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])):
                continue  # Skip out-of-bound neighbors

            if grid[neighbor[0]][neighbor[1]]:  # Check for obstacles
                continue

            if neighbor in closed_list:
                continue  # Skip already evaluated nodes

            g_cost = current_node.g_cost + 1  # Assume each step has a cost of 1
            h_cost = heuristic(neighbor, goal)
            neighbor_node = Cell(neighbor, g_cost, h_cost, current_node)

            # Check if the neighbor is already in the open list with a lower f_cost
            if any(
                open_node.position == neighbor and open_node.f_cost <= neighbor_node.f_cost for open_node in open_list
            ):
                continue

            heapq.heappush(open_list, neighbor_node)

    return []  # Return an empty list if no path is found


def score_room(room: Room, traversal_graph: Graph) -> float:
    """Scores a room based on how well it is laid out of a traversal graph.

    Args:
        room (Room): room to score
        traversal_graph (Graph): graph to evaluate using

    Returns:
        float: score
    """
    voxel_grid_x: int = math.ceil(room.bounds.x / VOXEL_RESOLUTION)
    voxel_grid_y: int = math.ceil(room.bounds.y / VOXEL_RESOLUTION)

    grid: list[list[bool]] = [[False] * voxel_grid_x] * voxel_grid_y

    item_rasters: dict[str, list[tuple[int, int]]] = {}

    for item in room.items:
        lower_bound_x: int = round((item.origin.x - (item.bounds.x / 2)) / VOXEL_RESOLUTION)
        lower_bound_y: int = round((item.origin.y - (item.bounds.y / 2)) / VOXEL_RESOLUTION)
        upper_bound_x: int = round((item.origin.x + (item.bounds.x / 2)) / VOXEL_RESOLUTION)
        upper_bound_y: int = round((item.origin.y + (item.bounds.y / 2)) / VOXEL_RESOLUTION)

        if item.orientation == Orientation.VERTICAL:
            lower_bound_x_temp: int = lower_bound_x
            lower_bound_y_temp: int = lower_bound_y
            upper_bound_x_temp: int = upper_bound_x
            upper_bound_y_temp: int = upper_bound_y

            lower_bound_x = lower_bound_y_temp
            lower_bound_y = lower_bound_x_temp
            upper_bound_x = upper_bound_y_temp
            upper_bound_y = upper_bound_x_temp

        item_rasters[item.name] = [
            (x, y) for x in range(lower_bound_x, upper_bound_x) for y in range(lower_bound_y, upper_bound_y)
        ]

    for raster in item_rasters.values():
        for x, y in raster:
            grid[x][y] = True

    score: int = 0

    for path in traversal_graph.paths:
        source: Item | None = next((item for item in room.items if item.name == path.source), None)
        target: Item | None = next((item for item in room.items if item.name == path.target), None)

        if source is None or target is None:
            continue

        # De-raster source and target to prevent it from blocking traversal
        for name, raster in item_rasters.items():
            if name in (path.source, path.target):
                for x, y in raster:
                    grid[x][y] = False

        source_x: int = round(source.origin.x / VOXEL_RESOLUTION)
        source_y: int = round(source.origin.y / VOXEL_RESOLUTION)
        target_x: int = round(target.origin.x / VOXEL_RESOLUTION)
        target_y: int = round(target.origin.y / VOXEL_RESOLUTION)

        traversal: list[tuple[int, int]] = a_star((source_x, source_y), (target_x, target_y), grid)

        score += len(traversal)

        # Re-restarter to allow for the next paths to work
        for name, raster in item_rasters.items():
            if name in (path.source, path.target):
                for x, y in raster:
                    grid[x][y] = True

    return score
