"""Map representation."""

import math
from enum import IntEnum

from pydantic import BaseModel, Field
from torch import Tensor


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
    orientation: Orientation

    def vector_span(self) -> int:
        """Get the vector span of item.

        Fixed items have zero span.

        Returns:
            int: number of elements in vector
        """
        if self.fixed:
            return 0

        # Span is determined by:
        # * bounds
        # * origin
        # * orientation
        span: int = 5
        for constraint in self.constraints:
            if isinstance(constraint, PositionConstraint):
                span += 4

            if isinstance(constraint, RadiusConstraint):
                span += 3

        return span

    def vectorize(self, index: int, tensor: Tensor) -> None:
        """Vectorize item at index in tensor.

        If item is fixed in place, it will not be added to tensor.

        Args:
            index (int): index to insert at
            tensor (Tensor): tensor to insert into
        """
        if self.fixed:
            return

        tensor[index + 0] = self.bounds.x
        tensor[index + 1] = self.bounds.y
        tensor[index + 2] = self.origin.x
        tensor[index + 3] = self.origin.y
        tensor[index + 4] = float(self.orientation)

        # Offset index to end of header
        index += 5

        for constraint in self.constraints:
            if isinstance(constraint, PositionConstraint):
                tensor[index + 0] = constraint.lower_bound.x
                tensor[index + 1] = constraint.lower_bound.y
                tensor[index + 2] = constraint.upper_bound.x
                tensor[index + 3] = constraint.upper_bound.y

                index += 4

            if isinstance(constraint, RadiusConstraint):
                tensor[index + 0] = constraint.origin.x
                tensor[index + 1] = constraint.origin.y
                tensor[index + 2] = constraint.radius

                index += 3

    def devectorize(self, index: int, tensor: Tensor) -> None:
        """Update item with values in vector.

        Fixed items are unchanged.

        Args:
            index (int): index to read from
            tensor (Tensor): tensor
        """
        if self.fixed:
            return

        self.bounds.x = tensor[index + 0].item()
        self.bounds.y = tensor[index + 1].item()
        self.origin.x = tensor[index + 2].item()
        self.origin.y = tensor[index + 3].item()
        self.orientation = Orientation.HORIZONTAL if tensor[index + 4].item() >= 0 else Orientation.VERTICAL

        # Offset index to end of header
        index += 5

        for constraint in self.constraints:
            if isinstance(constraint, PositionConstraint):
                constraint.lower_bound.x = tensor[index + 0].item()
                constraint.lower_bound.y = tensor[index + 1].item()
                constraint.upper_bound.x = tensor[index + 2].item()
                constraint.upper_bound.y = tensor[index + 3].item()

                index += 4

            if isinstance(constraint, RadiusConstraint):
                constraint.origin.x = tensor[index + 0].item()
                constraint.origin.y = tensor[index + 1].item()
                constraint.radius = tensor[index + 2].item()

                index += 3


class Room(BaseModel):
    """Room map."""

    bounds: CoordinatePair
    """Lower bound is assumed to be 0,0."""
    items: list[Item]
    keep_out_zones: list[PositionConstraint]
    """Zones to avoid placing anything not fixed."""

    def vector_span(self) -> int:
        """Get vector span of entire room.

        Returns:
            int: vector span
        """
        return sum(item.vector_span() for item in self.items)

    def vectorize(self, tensor: Tensor) -> None:
        """Vectorize data room into tensor.

        Args:
            tensor (Tensor): tensor to vectorize into
        """
        index: int = 0
        for item in self.items:
            item.vectorize(index, tensor)
            index += item.vector_span()

    def devectorize(self, tensor: Tensor) -> None:
        """Update room with update vector tensor.

        Args:
            tensor (Tensor): tensor
        """
        index: int = 0
        for item in self.items:
            item.devectorize(index, tensor)
            index += item.vector_span()


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
    if item.orientation != Orientation.HORIZONTAL:
        x = item.bounds.x
        y = item.bounds.y

        item.bounds.x = y
        item.bounds.y = x

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

    if item.orientation != Orientation.HORIZONTAL:
        x = item.bounds.x
        y = item.bounds.y

        item.bounds.x = y
        item.bounds.y = x

    return within_lower_bounds and within_upper_bounds


def is_item_constrained(item: Item) -> bool:
    """Check to see if item complies with constraints.

    Args:
        item (Item): item

    Returns:
        bool: if item complies with constraints
    """
    if item.fixed:
        return True

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

    return any(is_constrained(item, constraint) for constraint in item.constraints)


def is_room_constrained(room: Room) -> bool:
    """Check if all items in map comply with constraints.

    Args:
        room (Room): Map

    Returns:
        bool: if all items complies with constraints
    """
    return all(is_item_constrained(item) for item in room.items) and all(
        all(not is_item_position_constrained(item, keep_out) for item in room.items)
        for keep_out in room.keep_out_zones
    )
