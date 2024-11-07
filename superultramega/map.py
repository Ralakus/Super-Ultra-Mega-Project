"""Map representation."""

import math
from enum import IntEnum

from pydantic import BaseModel, Field


class CoordinatePair(BaseModel):
    """2D coordinate wrapper."""

    x: float = Field(default=0.0)
    y: float = Field(default=0.0)


class PositionConstraint(BaseModel):
    """Contrains object within certain bounds."""

    lower_bound: CoordinatePair
    upper_bound: CoordinatePair


class RadiusConstraint(BaseModel):
    """Constrains object to a certain distance from point."""

    origin: CoordinatePair
    radius: float


Constraint = RadiusConstraint | PositionConstraint


class Orientation(IntEnum):
    """Ordinal orientation."""

    HORIZONTAL = 1
    """Default positioning, axis are alinged."""
    VERTICAL = -1
    """Axis are swapped."""


class Item(BaseModel):
    """An object in the room."""

    name: str
    bounds: CoordinatePair
    """Lower bound is assumed to be 0,0 relative to origin."""
    origin: CoordinatePair
    """Origin sits right in middle of object: bounds.x/2, bounds.y/2."""
    constraints: list[Constraint]
    fixed: bool
    """If object can be moved by model."""


class Room(BaseModel):
    """Room map."""

    bounds: CoordinatePair
    """Lower bound is assumed to be 0,0."""

    items: list[Item]


def is_item_radius_constrained(item: Item, constraint: RadiusConstraint) -> bool:
    """Check to see if item complies with radius constraint.

    Args:
        item (Item): item
        constraint (RadiusConstraint): radius constriant

    Returns:
        bool: if item complies with constraint
    """
    distance: float = math.dist((item.origin.x, item.origin.y), (constraint.origin.x, constraint.origin.y))
    return distance <= constraint.radius


def is_item_position_constrained(item: Item, constraint: PositionConstraint) -> bool:
    """Check to see if item complies with position constraint.

    Args:
        item (Item): item
        constraint (PositionConstraint): position constriant

    Returns:
        bool: if item complies with constraint
    """
    item_lower_bound: CoordinatePair = CoordinatePair(
        x=item.origin.x - item.bounds.x / 2,
        y=item.origin.y - item.bounds.y / 2,
    )
    item_upper_bound: CoordinatePair = CoordinatePair(
        x=item.origin.x + item.bounds.x / 2,
        y=item.origin.y + item.bounds.y / 2,
    )

    within_lower_bounds: bool = (
        item_lower_bound.x >= constraint.lower_bound.x and item_lower_bound.y >= constraint.lower_bound.y
    )

    within_upper_bounds: bool = (
        item_upper_bound.x >= constraint.upper_bound.x and item_upper_bound.y >= constraint.upper_bound.y
    )

    return within_lower_bounds and within_upper_bounds


def is_item_constrained(item: Item) -> bool:
    """Check to see if item complies with constraints.

    Args:
        item (Item): item

    Returns:
        bool: if item complies with constraints
    """

    def is_constrained(item: Item, constraint: Constraint) -> bool:
        """Match constraint and checks if item complies.

        Args:
            item (Item): item
            constraint (Constraint): constraint

        Returns:
            bool: if item complies with constraint
        """
        if isinstance(constraint, PositionConstraint):
            return is_item_position_constrained(item, constraint)

        if isinstance(constraint, RadiusConstraint):
            return is_item_radius_constrained(item, constraint)

        # Unreachable code intentional to avoid lint warnings
        return False

    return all(is_constrained(item, constraint) for constraint in item.constraints)


def is_room_constrained(room: Room) -> bool:
    """Check if all items in map comply with constraints.

    Args:
        room (Room): Map

    Returns:
        bool: if all items complies with constraints
    """
    return all(is_item_constrained(item) for item in room.items)
