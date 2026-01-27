"""
Rigid body coordinate transformations for suspension components.

This module provides a hierarchical rigid body system where:
- Hardpoints: Define the kinematic reference frame of the body.
- Attachments: Move rigidly with the body but don't define its orientation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from kinematics.core.types import Vec3, make_vec3
from kinematics.core.vector_utils.generic import normalize_vector


@dataclass
class LocalCoordinateSystem:
    """
    Local coordinate system defined by origin and three orthonormal axes.

    Attributes:
        origin: Origin point in world coordinates.
        x_axis: Local X axis unit vector in world coordinates.
        y_axis: Local Y axis unit vector in world coordinates.
        z_axis: Local Z axis unit vector in world coordinates.
    """

    origin: Vec3
    x_axis: Vec3
    y_axis: Vec3
    z_axis: Vec3

    def __post_init__(self):
        """
        Ensure axes are unit vectors.
        """
        self.x_axis = normalize_vector(self.x_axis)
        self.y_axis = normalize_vector(self.y_axis)
        self.z_axis = normalize_vector(self.z_axis)

    def world_to_local(self, world_point: Vec3) -> Vec3:
        """
        Transform a point from world coordinates to local coordinates.

        Args:
            world_point: Point in world coordinates.

        Returns:
            Point in local coordinates.
        """
        # Translate to origin.
        relative = world_point - self.origin

        # Project onto local axes.
        local_x = np.dot(relative, self.x_axis)
        local_y = np.dot(relative, self.y_axis)
        local_z = np.dot(relative, self.z_axis)

        return make_vec3(np.array([local_x, local_y, local_z]))

    def local_to_world(self, local_point: Vec3) -> Vec3:
        """
        Transform a point from local coordinates to world coordinates.

        Args:
            local_point: Point in local coordinates.

        Returns:
            Point in world coordinates.
        """
        # Construct world position from local coordinates.
        world_point = (
            self.origin
            + local_point[0] * self.x_axis
            + local_point[1] * self.y_axis
            + local_point[2] * self.z_axis
        )

        return make_vec3(world_point)

    @classmethod
    def from_three_points(
        cls,
        origin: Vec3,
        z_point: Vec3,
        y_reference: Vec3,
    ) -> "LocalCoordinateSystem":
        """
        Construct a local coordinate system from three points.

        The construction follows this logic:
        1. Origin is at the specified origin point.
        2. Z axis points from origin to z_point.
        3. Y axis is in the plane defined by origin, z_point, and y_reference,
           orthogonal to Z.
        4. X axis completes the right-handed system (X = Y × Z).

        Args:
            origin: Origin point.
            z_point: Point defining the Z axis direction.
            y_reference: Point used to define the Y axis (with orthogonalisation).

        Returns:
            LocalCoordinateSystem constructed from the three points.

        Raises:
            ValueError: If points are collinear or invalid.
        """
        # Primary axis (Z): from origin to z_point.
        z_vec = z_point - origin
        z_axis = normalize_vector(z_vec)

        # Secondary axis (Y): project y_reference into plane perpendicular to Z.
        y_vec = y_reference - origin

        # Remove component along Z to ensure orthogonality.
        y_along_z = np.dot(y_vec, z_axis)
        y_perpendicular = y_vec - y_along_z * z_axis

        y_magnitude = np.linalg.norm(y_perpendicular)
        if y_magnitude < 1e-10:
            raise ValueError(
                "Y reference is collinear with Z axis - cannot define unique plane."
            )

        y_axis = y_perpendicular / y_magnitude

        # Tertiary axis (X): complete right-handed system.
        x_axis = normalize_vector(np.cross(y_axis, z_axis))

        return cls(
            origin=make_vec3(origin),
            x_axis=make_vec3(x_axis),
            y_axis=make_vec3(y_axis),
            z_axis=make_vec3(z_axis),
        )


class RigidBody(ABC):
    """
    Abstract base class for rigid bodies in suspension kinematics.

    A rigid body is defined by hardpoints (which define the kinematic reference
    frame) and attachments (which move with the body but don't define it).

    The body maintains local offsets for attachments that are transformed
    to world coordinates based on the current hardpoint positions.

    Subclasses must implement:
    - construct_lcs(): Define how to build the LCS from hardpoints.
    - init_local_frame(): Define how to compute attachment local offsets.

    Attributes:
        attachment_local_offsets: Local coordinates of attachments (computed at init).
        lcs: Current local coordinate system (updated from hardpoints).
    """

    def __init__(self):
        """
        Initialize the rigid body with empty attachment dictionary.
        """
        self.attachment_local_offsets: dict[str, Vec3] = {}
        self.lcs: LocalCoordinateSystem | None = None

    def get_lcs(self) -> LocalCoordinateSystem:
        """
        Get the current local coordinate system.

        Returns:
            The current LCS.

        Raises:
            RuntimeError: If LCS has not been initialized.
        """
        if self.lcs is None:
            raise RuntimeError(
                f"{self.__class__.__name__} LCS not initialized. "
                f"Call init_local_frame() first."
            )
        return self.lcs

    def get_world_position(self, attachment_name: str) -> Vec3:
        """
        Get the world position of an attachment.

        Args:
            attachment_name: Name of the attachment.

        Returns:
            World position of the attachment in current body configuration.

        Raises:
            KeyError: If attachment_name is not found.
        """
        if attachment_name not in self.attachment_local_offsets:
            raise KeyError(
                f"Attachment '{attachment_name}' not found in {self.__class__.__name__}"
            )

        local_offset = self.attachment_local_offsets[attachment_name]
        return self.get_lcs().local_to_world(local_offset)

    def add_attachment(self, name: str, world_position: Vec3) -> None:
        """
        Add an attachment at a specific world position.

        The world position is immediately converted to local coordinates
        using the current LCS. This allows arbitrary attachment points to be
        added to any rigid body.

        Args:
            name: Name of the attachment.
            world_position: World coordinates of the attachment.

        Raises:
            RuntimeError: If LCS has not been initialized.
        """
        local_offset = self.get_lcs().world_to_local(world_position)
        self.attachment_local_offsets[name] = local_offset

    def set_attachment_local_offset(
        self, attachment_name: str, local_offset: Vec3
    ) -> None:
        """
        Directly set the local offset for an attachment.

        This is used during initialisation or when applying transformations
        like camber shims that modify the attachment positions relative to
        the hardpoints.

        Args:
            attachment_name: Name of the attachment.
            local_offset: Local coordinates of the attachment.
        """
        self.attachment_local_offsets[attachment_name] = make_vec3(local_offset)

    @abstractmethod
    def construct_lcs(self) -> LocalCoordinateSystem:
        """
        Construct the local coordinate system from hardpoints.

        This method must be implemented by subclasses to define how their
        specific hardpoint configuration determines the body's orientation.

        Returns:
            LocalCoordinateSystem constructed from the body's hardpoints.

        Example:
            For an Upright:
            - Origin at lower_ball_joint.
            - Z axis toward upper_ball_joint.
            - Y axis in plane toward tie_rod_pickup.
        """
        raise NotImplementedError

    @abstractmethod
    def init_local_frame(self) -> None:
        """
        Initialize the local coordinate frame and compute local offsets for all
        attachments.

        This method must be implemented by subclasses to:
        1. Call construct_lcs() to build the LCS.
        2. Convert all attachment world positions to local coordinates.
        3. Store them in attachment_local_offsets.

        This is called once at initialization and again after transformations
        (like camber shims) to update the design state.
        """
        raise NotImplementedError
