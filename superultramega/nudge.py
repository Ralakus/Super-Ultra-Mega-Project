"""Map nudge to fit into constraints."""

import math
from copy import deepcopy
from typing import Final

from . import LoopLimitReachedError, clamp
from .map import (
    CoordinatePair,
    Item,
    PositionConstraint,
    RadiusConstraint,
    Room,
    is_item_constrained,
    is_item_position_constrained,
    is_item_radius_constrained,
)

MAX_NUDGE_ITERATIONS: Final[int] = 2 ^ 16
"""Max number of iterations to run to avoid infinite loops."""

NUDGE_DELTA: Final[float] = 2.0**-5
"""Amount to change position by each nudge iteration."""


def nudge(room: Room, item_name: str) -> CoordinatePair:
    """Nudge an item that isn't within constraints to fit into constraints.

    Args:
        room (Room): room
        item_name (str): item name in map

    Raise:
        LoopLimitReachedError: when nudge cannot be completed within `MAX_NUDGE_ITERATIONS` iterations

    Returns:
        CoordinatePair: nudged coordinate pair
    """
    item: Item | None = next((item for item in room.items if item.name == item_name), None)
    if item is None:
        return CoordinatePair(x=0, y=0)

    if item.fixed:
        return item.origin

    # Ensure room is within room bounds
    item.origin.x = clamp(item.origin.x, 0, room.bounds.x)
    item.origin.y = clamp(item.origin.y, 0, room.bounds.y)

    coordinates: CoordinatePair = deepcopy(item.origin)
    counter: int = 0

    while not is_item_constrained(item):

        delta: CoordinatePair = CoordinatePair(x=0, y=0)

        radius_constrained: bool = any(
            is_item_radius_constrained(item, constraint)
            for constraint in item.constraints
            if isinstance(constraint, RadiusConstraint)
        )

        # Find closest radius constriant to work towards
        target_radius: RadiusConstraint | None = min(
            (
                constraint
                for constraint in item.constraints
                if isinstance(constraint, RadiusConstraint) and not radius_constrained
            ),
            key=lambda constraint: math.dist(
                (constraint.origin.x, constraint.origin.y),
                (item.origin.x, item.origin.y),
            ),
            default=None,
        )

        if target_radius is not None:
            delta.x += target_radius.origin.x - item.origin.x
            delta.y += target_radius.origin.y - item.origin.y

        overlapping_keep_out_zones: list[PositionConstraint] = [
            constraint for constraint in room.keep_out_zones if is_item_position_constrained(item, constraint)
        ]

        for zone in overlapping_keep_out_zones:
            position: CoordinatePair = CoordinatePair(
                x=(zone.lower_bound.x + zone.upper_bound.x) / 2,
                y=(zone.lower_bound.y + zone.upper_bound.y) / 2,
            )

            delta.x -= position.x - item.origin.x
            delta.y -= position.y - item.origin.y

        coordinates.x += delta.x * NUDGE_DELTA
        coordinates.y += delta.y * NUDGE_DELTA

        # Failsafe to avoid infinite loops
        counter += 1
        if counter >= MAX_NUDGE_ITERATIONS:
            raise LoopLimitReachedError

    return coordinates
