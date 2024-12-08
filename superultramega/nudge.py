"""Map nudge to fit into constraints."""

import math
from typing import Final

from . import LoopLimitReachedError, clampf
from .map import (
    CoordinatePair,
    Item,
    Orientation,
    PositionConstraint,
    RadiusConstraint,
    Room,
    is_item_constrained,
    is_item_position_constrained,
    is_item_radius_constrained,
)

MAX_NUDGE_ITERATIONS: Final[int] = 2**32
"""Max number of iterations to run to avoid infinite loops."""

NUDGE_DELTA: Final[float] = 2.0**-3
"""Amount to change position by each nudge iteration."""


def nudge(room: Room, item_name: str) -> None:
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
        return

    if item.fixed:
        return

    if item.orientation == Orientation.VERTICAL:
        item.bounds.invert()

    # Ensure room is within room bounds
    item.origin.x = clampf(item.origin.x, item.bounds.x, room.bounds.x - item.bounds.x)
    item.origin.y = clampf(item.origin.y, item.bounds.y, room.bounds.y - item.bounds.y)

    if len(item.constraints) == 0:
        if item.orientation == Orientation.VERTICAL:
            item.bounds.invert()

        return

    counter: int = 0

    while not is_item_constrained(item):

        if math.isnan(item.origin.x):
            item.origin.x = 0.0

        if math.isnan(item.origin.y):
            item.origin.y = 0.0

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

        item.origin.x += delta.x * NUDGE_DELTA
        item.origin.y += delta.y * NUDGE_DELTA

        # Failsafe to avoid infinite loops
        counter += 1
        if counter >= MAX_NUDGE_ITERATIONS:
            raise LoopLimitReachedError

    # Ensure room is within room bounds
    item.origin.x = clampf(item.origin.x, item.bounds.x, room.bounds.x - item.bounds.x)
    item.origin.y = clampf(item.origin.y, item.bounds.y, room.bounds.y - item.bounds.y)

    if item.orientation == Orientation.VERTICAL:
        item.bounds.invert()

    return
