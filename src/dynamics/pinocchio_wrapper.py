"""Pinocchio-based kinematics/dynamics accessors.

This module is the sole source of truth for CoM, centroidal momentum, and foot
frame computations. All other modules must import from here.
"""

# G1 URDF diagnostics (FreeFlyer model)
#
# nq = 36
# nv = 35
#
# Foot frames found:
#   left_ankle_roll_link
#   right_ankle_roll_link
#
# Joint names:
#   root_joint
#   left_hip_pitch_joint
#   left_hip_roll_joint
#   left_hip_yaw_joint
#   left_knee_joint
#   left_ankle_pitch_joint
#   left_ankle_roll_joint
#   right_hip_pitch_joint
#   right_hip_roll_joint
#   right_hip_yaw_joint
#   right_knee_joint
#   right_ankle_pitch_joint
#   right_ankle_roll_joint
#   waist_yaw_joint
#   waist_roll_joint
#   waist_pitch_joint
#   left_shoulder_pitch_joint
#   left_shoulder_roll_joint
#   left_shoulder_yaw_joint
#   left_elbow_joint
#   left_wrist_roll_joint
#   left_wrist_pitch_joint
#   left_wrist_yaw_joint
#   right_shoulder_pitch_joint
#   right_shoulder_roll_joint
#   right_shoulder_yaw_joint
#   right_elbow_joint
#   right_wrist_roll_joint
#   right_wrist_pitch_joint
#   right_wrist_yaw_joint

from __future__ import annotations

from typing import Iterable
import math

import numpy as np
import pinocchio as pin

try:  # Optional, for convex hulls
    from scipy.spatial import ConvexHull

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:  # Optional, for convex hulls and distance
    from shapely.geometry import MultiPoint, Point, Polygon

    _HAVE_SHAPELY = True
except Exception:
    _HAVE_SHAPELY = False

_G = 9.81

# Degenerate-case fallback only (a "contacting" foot whose contact geometry is
# unknown). The real support polygon comes from FOOT_CONTACT_OFFSETS below.
_SUPPORT_SQUARE_SIDE = 0.05

# Foot contact geometry of the G1, in the ankle_roll_link body frame.
#
# These are the positions of the four collision spheres that the official
# Unitree MJCF (g1_29dof_rev_1_0.xml) attaches to each ankle_roll_link -- two at
# the heel (x = -0.05) and two at the toe (x = +0.12), at y = +/-0.025 (heel) and
# +/-0.03 (toe), all 0.03 m below the ankle roll axis. They describe a real
# ~17 cm x 6 cm foot. Using them instead of a fixed-size square is what makes the
# support polygon and the capture-point margin physically meaningful.
#
# tests/test_foot_geometry.py asserts these against the MJCF so they cannot
# silently drift from the model.
FOOT_CONTACT_OFFSETS: np.ndarray = np.array(
    [
        [-0.05, 0.025, -0.03],
        [-0.05, -0.025, -0.03],
        [0.12, 0.03, -0.03],
        [0.12, -0.03, -0.03],
    ],
    dtype=np.float64,
)


def _square_around_point(pt_xy: np.ndarray, side: float) -> np.ndarray:
    half = side * 0.5
    return np.array(
        [
            [pt_xy[0] - half, pt_xy[1] - half],
            [pt_xy[0] + half, pt_xy[1] - half],
            [pt_xy[0] + half, pt_xy[1] + half],
            [pt_xy[0] - half, pt_xy[1] + half],
        ],
        dtype=np.float64,
    )


def _rectangle_around_segment(p0: np.ndarray, p1: np.ndarray, width: float) -> np.ndarray:
    seg = p1 - p0
    norm = np.linalg.norm(seg)
    if norm < 1e-8:
        return _square_around_point(p0, width)
    perp = np.array([-seg[1], seg[0]], dtype=np.float64) / norm
    half = width * 0.5
    offset = perp * half
    return np.array([p0 + offset, p1 + offset, p1 - offset, p0 - offset], dtype=np.float64)


def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    area = _polygon_area(poly)
    if area < 1e-8:
        return np.mean(poly, axis=0)
    x = poly[:, 0]
    y = poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = float(np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area))
    cy = float(np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area))
    return np.array([cx, cy], dtype=np.float64)


def _point_to_segment_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(pt - a))
    t = float(np.dot(pt - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def _point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    if poly.shape[0] < 3:
        return False
    signs = []
    for i in range(poly.shape[0]):
        p0 = poly[i]
        p1 = poly[(i + 1) % poly.shape[0]]
        edge = p1 - p0
        rel = pt - p0
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if abs(cross) < 1e-12:
            continue
        signs.append(math.copysign(1.0, cross))
    if not signs:
        return True
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def _signed_distance_to_polygon(pt: np.ndarray, poly: np.ndarray) -> float:
    if poly.shape[0] < 2:
        return -999.0
    min_dist = float("inf")
    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        min_dist = min(min_dist, _point_to_segment_distance(pt, a, b))
    inside = _point_in_polygon(pt, poly)
    return min_dist if inside else -min_dist


def _convex_hull(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] == 1:
        return _square_around_point(points_xy[0], _SUPPORT_SQUARE_SIDE)
    if points_xy.shape[0] == 2:
        return _rectangle_around_segment(points_xy[0], points_xy[1], _SUPPORT_SQUARE_SIDE)
    if _HAVE_SHAPELY:
        hull = MultiPoint(points_xy).convex_hull
        if hull.geom_type == "Polygon":
            return np.array(hull.exterior.coords[:-1], dtype=np.float64)
        if hull.geom_type == "LineString":
            coords = np.array(hull.coords, dtype=np.float64)
            if coords.shape[0] == 2:
                return _rectangle_around_segment(coords[0], coords[1], _SUPPORT_SQUARE_SIDE)
        if hull.geom_type == "Point":
            coords = np.array(hull.coords, dtype=np.float64)
            return _square_around_point(coords[0], _SUPPORT_SQUARE_SIDE)
    if _HAVE_SCIPY:
        hull = ConvexHull(points_xy)
        return points_xy[hull.vertices]
    raise ImportError("Support polygon requires scipy or shapely.")


class PinocchioWrapper:
    """Lightweight Pinocchio wrapper for G1 kinematics and dynamics."""

    # Optional overrides if auto-detection fails.
    DEFAULT_LEFT_FOOT_FRAMES: list[str] = ["left_ankle_roll_link"]
    DEFAULT_RIGHT_FOOT_FRAMES: list[str] = ["right_ankle_roll_link"]

    def __init__(self, urdf_path: str, foot_contact_offsets: np.ndarray | None = None):
        # Note: we use JointModelFreeFlyer() here, which adds a floating base to the model.
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.foot_contact_offsets = (
            FOOT_CONTACT_OFFSETS.copy()
            if foot_contact_offsets is None
            else np.asarray(foot_contact_offsets, dtype=np.float64).reshape(-1, 3)
        )
        self.data = self.model.createData()
        self._frame_names = [f.name for f in self.model.frames]

        self.left_foot_frame = self._resolve_foot_frame("left", self.DEFAULT_LEFT_FOOT_FRAMES)
        self.right_foot_frame = self._resolve_foot_frame("right", self.DEFAULT_RIGHT_FOOT_FRAMES)
        if self.left_foot_frame is None or self.right_foot_frame is None:
            raise ValueError(
                "Foot frame names not found. Run the URDF diagnostics (Step 2) "
                "and update DEFAULT_LEFT_FOOT_FRAMES/DEFAULT_RIGHT_FOOT_FRAMES."
            )
        if self.left_foot_frame not in self._frame_names:
            raise ValueError(f"Left foot frame not found: {self.left_foot_frame}")
        if self.right_foot_frame not in self._frame_names:
            raise ValueError(f"Right foot frame not found: {self.right_foot_frame}")
        self.left_foot_id = self.model.getFrameId(self.left_foot_frame)
        self.right_foot_id = self.model.getFrameId(self.right_foot_frame)

        # effortLimit[0:6] is the (unbounded, inf) floating-base freeflyer joint;
        # only the actuated joints from index 6 on have a real URDF effort limit.
        self.effort_limit = np.array(self.model.effortLimit[6:], dtype=np.float64)

    def _resolve_foot_frame(self, side: str, preferred: Iterable[str]) -> str | None:
        for name in preferred:
            if name in self._frame_names:
                return name
        tokens = ("ankle", "foot", "toe", "heel", "link")
        candidates = [
            name
            for name in self._frame_names
            if side in name.lower() and any(tok in name.lower() for tok in tokens)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            for tok in ("ankle", "foot", "toe", "heel"):
                filtered = [name for name in candidates if tok in name.lower()]
                if len(filtered) == 1:
                    return filtered[0]
        return None

    def _update_all(self, q: np.ndarray, dq: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        dq = np.asarray(dq, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.nq:
            raise ValueError(f"q has shape {q.shape[0]} but model.nq={self.model.nq}")
        if dq.shape[0] != self.model.nv:
            raise ValueError(f"dq has shape {dq.shape[0]} but model.nv={self.model.nv}")
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.centerOfMass(self.model, self.data, q, dq)
        pin.computeCentroidalMomentum(self.model, self.data, q, dq)

    def compute_com(self, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (com [3], com_vel [3])."""
        self._update_all(q, dq)
        com = np.array(self.data.com[0]).reshape(3)
        com_vel = np.array(self.data.vcom[0]).reshape(3)
        return com, com_vel

    def compute_centroidal_momentum(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Returns hg [6] — [linear_momentum (3), angular_momentum (3)]."""
        self._update_all(q, dq)
        return np.array(self.data.hg.vector).reshape(6)

    def get_frame_pose(self, frame_name: str, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Returns 4x4 SE3 homogeneous transform (np.ndarray)."""
        self._update_all(q, dq)
        frame_id = self.model.getFrameId(frame_name)
        if frame_id >= len(self.model.frames):
            raise ValueError(f"Frame not found: {frame_name}")
        pose = self.data.oMf[frame_id]
        try:
            mat = pose.homogeneous
        except AttributeError:
            mat = pose.toHomogeneousMatrix()
        return np.array(mat, dtype=np.float64)

    def get_foot_positions(self, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (left_foot_pos [3], right_foot_pos [3]) in world frame."""
        self._update_all(q, dq)
        lf = np.array(self.data.oMf[self.left_foot_id].translation).reshape(3)
        rf = np.array(self.data.oMf[self.right_foot_id].translation).reshape(3)
        return lf, rf

    def foot_contact_points(self, q: np.ndarray, dq: np.ndarray, side: str) -> np.ndarray:
        """World positions of one foot's four contact spheres, shape [4, 3]."""
        self._update_all(q, dq)
        frame_id = self.left_foot_id if side == "left" else self.right_foot_id
        placement = self.data.oMf[frame_id]
        rot = np.array(placement.rotation, dtype=np.float64)
        trans = np.array(placement.translation, dtype=np.float64).reshape(3)
        return (self.foot_contact_offsets @ rot.T) + trans

    def compute_com_jacobian(self, q: np.ndarray) -> np.ndarray:
        """d(CoM_xyz)/dq, shape [3, nv]."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        return np.array(pin.jacobianCenterOfMass(self.model, self.data, q), dtype=np.float64)

    def inverse_dynamics(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """Generalized forces via RNEA, shape [nv]."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        dq = np.asarray(dq, dtype=np.float64).reshape(-1)
        ddq = np.asarray(ddq, dtype=np.float64).reshape(-1)
        return np.array(pin.rnea(self.model, self.data, q, dq, ddq), dtype=np.float64)

    def foot_jacobian(self, q: np.ndarray, side: str) -> np.ndarray:
        """World-frame linear velocity Jacobian of one foot frame, shape [3, nv]."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        frame_id = self.left_foot_id if side == "left" else self.right_foot_id
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(self.model, self.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
        return np.array(J[:3, :], dtype=np.float64)

    def get_support_features(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        contacts: np.ndarray,
        support_points_world: np.ndarray | None = None,
    ) -> dict:
        """Compute support polygon and stability features.

        The support polygon is the convex hull of the contact-sphere positions of
        every foot in contact (four spheres per foot, from the real MJCF foot
        geometry). Pass `support_points_world` [N, 2 or 3] to override with points
        measured directly from the simulator instead of from forward kinematics.
        """
        contacts = np.asarray(contacts, dtype=bool).reshape(-1)
        if contacts.shape[0] != 2:
            raise ValueError("contacts must be shape [2] -> (left, right)")

        self._update_all(q, dq)
        com = np.array(self.data.com[0]).reshape(3)
        com_vel = np.array(self.data.vcom[0]).reshape(3)

        lf = np.array(self.data.oMf[self.left_foot_id].translation).reshape(3)
        rf = np.array(self.data.oMf[self.right_foot_id].translation).reshape(3)

        points_xy = []
        if support_points_world is not None and len(support_points_world) > 0:
            pts = np.asarray(support_points_world, dtype=np.float64).reshape(-1, 3 if
                np.asarray(support_points_world).shape[-1] == 3 else 2)
            points_xy = [p[:2] for p in pts]
        else:
            if contacts[0]:
                points_xy.extend(self.foot_contact_points(q, dq, "left")[:, :2])
            if contacts[1]:
                points_xy.extend(self.foot_contact_points(q, dq, "right")[:, :2])

        if not len(points_xy):
            return {
                "support_polygon": np.zeros((0, 2), dtype=np.float64),
                "support_area": 0.0,
                "support_center": np.array([np.nan, np.nan], dtype=np.float64),
                "com_margin": -999.0,
                "capture_point": np.array([np.nan, np.nan], dtype=np.float64),
                "cp_margin": -999.0,
                "support_mode": "no_contact",
            }

        points_xy = np.array(points_xy, dtype=np.float64)
        if contacts[0] and contacts[1]:
            support_mode = "double"
        elif contacts[0]:
            support_mode = "single_left"
        else:
            support_mode = "single_right"

        # With real foot geometry a single contacting foot already yields four
        # hull points, so the degenerate 1-/2-point branches below are reached
        # only if contact geometry was unavailable.
        if points_xy.shape[0] == 1:
            support_poly = _square_around_point(points_xy[0], _SUPPORT_SQUARE_SIDE)
        elif points_xy.shape[0] == 2:
            support_poly = _rectangle_around_segment(
                points_xy[0], points_xy[1], _SUPPORT_SQUARE_SIDE
            )
        else:
            support_poly = _convex_hull(points_xy)

        support_area = _polygon_area(support_poly)
        support_center = _polygon_centroid(support_poly)

        omega = math.sqrt(_G / max(float(com[2]), 0.1))
        capture_point = com[:2] + com_vel[:2] / omega

        com_margin = _signed_distance_to_polygon(com[:2], support_poly)
        cp_margin = _signed_distance_to_polygon(capture_point, support_poly)

        return {
            "support_polygon": support_poly,
            "support_area": float(support_area),
            "support_center": support_center,
            "com_margin": float(com_margin),
            "capture_point": capture_point,
            "cp_margin": float(cp_margin),
            "support_mode": support_mode,
        }


if __name__ == "__main__":
    import pinocchio as pin

    wrapper = PinocchioWrapper("assets/unitree_g1/g1.urdf")

    # pin.neutral() puts the floating base at the world origin, which buries the
    # feet ~0.76 m below the floor and makes every height check meaningless.
    # Lift the base so the lowest foot contact sphere rests exactly on z = 0.
    q0 = pin.neutral(wrapper.model)
    dq0 = np.zeros(wrapper.model.nv)
    foot_pts = np.vstack(
        [wrapper.foot_contact_points(q0, dq0, "left"),
         wrapper.foot_contact_points(q0, dq0, "right")]
    )
    q0[2] -= float(foot_pts[:, 2].min())
    print(f"Base lifted to z={q0[2]:.4f} so lowest foot sphere sits on the floor")

    com, com_vel = wrapper.compute_com(q0, dq0)
    hg = wrapper.compute_centroidal_momentum(q0, dq0)
    lf, rf = wrapper.get_foot_positions(q0, dq0)
    contacts = np.array([True, True])
    feats = wrapper.get_support_features(q0, dq0, contacts)

    print("CoM:", com)
    print("CoM vel:", com_vel)
    print("hg:", hg)
    print("Left foot:", lf)
    print("Right foot:", rf)
    print("CP margin:", feats["cp_margin"])
    print("Support mode:", feats["support_mode"])

    assert com[2] > 0.3, f"CoM height suspiciously low: {com[2]}"
    assert com[2] < 1.5, f"CoM height suspiciously high: {com[2]}"
    assert feats["support_mode"] == "double"
    assert feats["cp_margin"] > 0
    print("All smoke tests passed.")
