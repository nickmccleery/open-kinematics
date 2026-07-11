"""
Suspension construction from validated geometry specifications.

Concrete builder functions are registered once in
``kinematics.suspensions.registry``. The public ``build_suspension`` function
dispatches through that same catalogue.
"""

from __future__ import annotations

from typing import cast

from kinematics.core.enums import PointID, ShimType
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.point_ref import PointKey, Side
from kinematics.schema.config import SuspensionConfig
from kinematics.schema.geometry import (
    DoubleWishboneAxleGeometrySpec,
    DoubleWishboneCoiloverGeometrySpec,
    DoubleWishboneGeometrySpec,
    DoubleWishbonePushrodRockerArbGeometrySpec,
    DoubleWishbonePushrodRockerAxleGeometrySpec,
    DoubleWishbonePushrodRockerGeometrySpec,
    GeometrySpecBase,
)
from kinematics.suspensions.axle import (
    DoubleWishboneAxleSuspension,
    DoubleWishbonePushrodRockerAxleSuspension,
)
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.corner import (
    DoubleWishboneCoiloverSuspension,
    DoubleWishbonePushrodRockerArbSuspension,
    DoubleWishbonePushrodRockerSuspension,
    DoubleWishboneSuspension,
)


def build_suspension(spec: GeometrySpecBase) -> Suspension:
    """Construct a suspension using the single registered type definition."""
    from kinematics.suspensions.registry import get_suspension_definition

    definition = get_suspension_definition(spec.type)
    if definition is None:
        raise TypeError(f"Unsupported geometry spec type: {spec.type}")
    if not isinstance(spec, definition.spec_type):
        raise TypeError(
            f"Type '{spec.type}' requires {definition.spec_type.__name__}, "
            f"got {type(spec).__name__}."
        )
    return definition.build(spec)


def build_double_wishbone(spec: GeometrySpecBase) -> Suspension:
    """Build the basic double-wishbone corner."""
    typed = cast(DoubleWishboneGeometrySpec, spec)
    return _build_corner(typed, DoubleWishboneSuspension)


def build_double_wishbone_coilover(spec: GeometrySpecBase) -> Suspension:
    """Build the lower-wishbone-mounted coilover corner."""
    typed = cast(DoubleWishboneCoiloverGeometrySpec, spec)
    return _build_corner(typed, DoubleWishboneCoiloverSuspension)


def build_double_wishbone_pushrod_rocker(spec: GeometrySpecBase) -> Suspension:
    """Build the explicit pushrod-rocker corner."""
    typed = cast(DoubleWishbonePushrodRockerGeometrySpec, spec)
    return _build_corner(
        typed,
        DoubleWishbonePushrodRockerSuspension,
        spring_type=typed.spring.type,
    )


def build_double_wishbone_pushrod_rocker_arb(spec: GeometrySpecBase) -> Suspension:
    """Build a pushrod-rocker corner with an anti-roll-bar pickup."""
    typed = cast(DoubleWishbonePushrodRockerArbGeometrySpec, spec)
    return _build_corner(
        typed,
        DoubleWishbonePushrodRockerArbSuspension,
        spring_type=typed.spring.type,
    )


def build_double_wishbone_axle(spec: GeometrySpecBase) -> Suspension:
    """Build the basic two-corner double-wishbone axle."""
    typed = cast(DoubleWishboneAxleGeometrySpec, spec)
    return _build_axle(
        typed,
        corner_cls=DoubleWishboneSuspension,
        axle_cls=DoubleWishboneAxleSuspension,
        include_arb=False,
    )


def build_double_wishbone_pushrod_rocker_axle(
    spec: GeometrySpecBase,
) -> Suspension:
    """Build the full pushrod-rocker axle with anti-roll bar."""
    typed = cast(DoubleWishbonePushrodRockerAxleGeometrySpec, spec)
    return _build_axle(
        typed,
        corner_cls=DoubleWishbonePushrodRockerArbSuspension,
        axle_cls=DoubleWishbonePushrodRockerAxleSuspension,
        include_arb=True,
        spring_type=typed.spring.type,
    )


def _build_corner(
    spec: DoubleWishboneGeometrySpec
    | DoubleWishboneCoiloverGeometrySpec
    | DoubleWishbonePushrodRockerArbGeometrySpec
    | DoubleWishbonePushrodRockerGeometrySpec,
    cls: type[DoubleWishboneSuspension],
    **kwargs: object,
) -> DoubleWishboneSuspension:
    """Build one concrete corner after exact point validation."""
    _check_valid_points(spec.hardpoints, cls)
    _check_shim_support(spec.config, cls)
    return cls(
        name=spec.name,
        version=spec.version,
        units=spec.units,
        hardpoints=_copy_points(spec.hardpoints),
        config=spec.config,
        **kwargs,
    )


def _build_axle(
    spec: DoubleWishboneAxleGeometrySpec | DoubleWishbonePushrodRockerAxleGeometrySpec,
    *,
    corner_cls: type[DoubleWishboneSuspension],
    axle_cls: type[DoubleWishboneAxleSuspension],
    include_arb: bool,
    spring_type: str | None = None,
) -> DoubleWishboneAxleSuspension:
    """Build a concrete axle from explicit or mirrored side geometry."""
    _check_shim_support(spec.config, corner_cls)
    blocks = spec.hardpoints
    droplink_arb_points: dict[Side, Point3] = {}

    if blocks.is_explicit:
        assert blocks.left is not None and blocks.right is not None
        side_points: dict[Side, dict[PointKey, Point3]] = {}
        for side, block in ((Side.LEFT, blocks.left), (Side.RIGHT, blocks.right)):
            points = _copy_points(block)
            if include_arb:
                droplink = points.pop(PointID.DROPLINK_ARB, None)
                if droplink is not None:
                    droplink_arb_points[side] = droplink
            _validate_side_signs(points, side)
            side_points[side] = points
    else:
        assert blocks.points is not None
        source_side = blocks.side
        source = _copy_points(blocks.points)
        source_droplink = (
            source.pop(PointID.DROPLINK_ARB, None) if include_arb else None
        )
        _validate_side_signs(source, source_side)
        other_side = Side.RIGHT if source_side == Side.LEFT else Side.LEFT
        side_points = {
            source_side: source,
            other_side: _mirror_hardpoints(source),
        }
        if source_droplink is not None:
            droplink_arb_points[source_side] = source_droplink
            droplink_arb_points[other_side] = _mirror_point(source_droplink)

    for side in (Side.LEFT, Side.RIGHT):
        _check_valid_points(side_points[side], corner_cls)

    center_points = _check_center_points(blocks.center)
    if not include_arb and center_points:
        raise ValueError(
            "Suspension type 'double_wishbone_axle' does not accept a center "
            "ARB block; use 'double_wishbone_pushrod_rocker_axle'."
        )

    side_configs = {
        Side.LEFT: spec.config,
        Side.RIGHT: _mirror_config(spec.config),
    }
    corner_kwargs = {} if spring_type is None else {"spring_type": spring_type}
    corners = {
        side: corner_cls(
            name=f"{spec.name}_{side.name.lower()}",
            version=spec.version,
            units=spec.units,
            hardpoints=side_points[side],
            config=side_configs[side],
            **corner_kwargs,
        )
        for side in (Side.LEFT, Side.RIGHT)
    }
    axle_kwargs: dict[str, object] = {}
    if include_arb:
        axle_kwargs = {
            "center_points": center_points,
            "droplink_arb_points": droplink_arb_points,
        }
    return axle_cls(
        name=spec.name,
        version=spec.version,
        units=spec.units,
        hardpoints={},
        config=spec.config,
        corners=corners,
        **axle_kwargs,
    )


_VALID_CENTER_POINTS = frozenset({PointID.ARB_AXIS_A, PointID.ARB_AXIS_B})


def _copy_points(points: dict[PointID, Point3]) -> dict[PointKey, Point3]:
    """Copy a point map so built suspensions never alias spec-owned data."""
    return {point: position.copy() for point, position in points.items()}


def _check_valid_points(points: dict[PointKey, Point3], cls: type[Suspension]) -> None:
    """Reject points the concrete suspension class does not define."""
    unknown = set(points) - set(cls.all_valid_points())
    if unknown:
        names = ", ".join(sorted(point.name for point in unknown))
        raise ValueError(f"Invalid hardpoints for {cls.TYPE_KEY}: {names}")


def _check_shim_support(config: SuspensionConfig, cls: type[Suspension]) -> None:
    """Reject a camber shim config on a type that does not support shims."""
    if (
        config.camber_shim is not None
        and ShimType.OUTBOARD_CAMBER not in cls.SUPPORTED_SHIMS
    ):
        raise ValueError(
            f"Suspension type '{cls.TYPE_KEY}' does not support outboard camber shims"
        )


def _check_center_points(center: dict[PointID, Point3]) -> dict[PointID, Point3]:
    """Validate the shared center block."""
    unknown = set(center) - _VALID_CENTER_POINTS
    if unknown:
        names = ", ".join(sorted(point.name for point in unknown))
        raise ValueError(
            f"Unknown center point(s): {names} (expected arb_axis_a/arb_axis_b)."
        )
    return {point: position.copy() for point, position in center.items()}


def _validate_side_signs(points: dict[PointKey, Point3], side: Side) -> None:
    """Check that a corner's outboard Y sign matches its declared side."""
    axle_outboard = points.get(PointID.AXLE_OUTBOARD)
    if axle_outboard is None:
        return
    axle_out_y = float(axle_outboard.data[1])
    if side == Side.LEFT and axle_out_y <= 0:
        raise ValueError(
            "Side 'left' requires AXLE_OUTBOARD Y > 0 "
            f"(got {axle_out_y}); check the hardpoint handedness."
        )
    if side == Side.RIGHT and axle_out_y >= 0:
        raise ValueError(
            "Side 'right' requires AXLE_OUTBOARD Y < 0 "
            f"(got {axle_out_y}); check the hardpoint handedness."
        )


def _mirror_point(point: Point3) -> Point3:
    """Reflect a point through the XZ plane."""
    x, y, z = float(point.data[0]), float(point.data[1]), float(point.data[2])
    return Point3([x, -y, z])


def _mirror_hardpoints(
    hardpoints: dict[PointKey, Point3],
) -> dict[PointKey, Point3]:
    """Reflect every hardpoint through the XZ plane."""
    return {point: _mirror_point(position) for point, position in hardpoints.items()}


def _mirror_config(config: SuspensionConfig) -> SuspensionConfig:
    """Mirror side-dependent camber-shim geometry for the opposite corner."""
    if config.camber_shim is None:
        return config
    shim = config.camber_shim
    normal = shim.shim_face_normal
    mirrored_normal = Direction3(
        [float(normal.data[0]), -float(normal.data[1]), float(normal.data[2])]
    )
    mirrored_shim = shim.model_copy(
        update={
            "shim_face_point_a": _mirror_point(shim.shim_face_point_a),
            "shim_face_point_b": _mirror_point(shim.shim_face_point_b),
            "shim_face_normal": mirrored_normal,
        }
    )
    return config.model_copy(update={"camber_shim": mirrored_shim})
