import contextlib
import copy
import io
from dataclasses import dataclass
from typing import List

import numpy as np
from bs4 import BeautifulSoup

from RBDReference import RBDReference
from URDFParser.Joint import Joint
from URDFParser.Robot import Robot
from URDFParser.URDFParser import URDFParser

from test.pinocchio_equivalents.adapters.normalization import (
    ConventionMismatch,
    movable_joint_names_excluding_floating_root,
    normalize_matrix,
    normalize_vector,
)


class ProjectParseError(RuntimeError):
    """Raised when the strict GRiD-side parser flow fails."""


@dataclass
class ProjectModelAdapter:
    spec: object
    base_mode: str
    robot: Robot
    reference: RBDReference
    parse_output: str
    mismatches: List[ConventionMismatch]

    @property
    def nq(self) -> int:
        return self.robot.get_num_pos()

    @property
    def nv(self) -> int:
        return self.robot.get_num_vel()

    @property
    def joint_names(self) -> List[str]:
        return [joint.get_name() for joint in self.robot.get_joints_ordered_by_id()]

    @property
    def actuated_joint_names(self) -> List[str]:
        return movable_joint_names_excluding_floating_root(self.base_mode, self.joint_names)

    def rnea(self, q, qd, qdd):
        c, _v, _a, _f = self.reference.rnea(q, qd, qdd)
        return normalize_vector(c)

    def minv(self, q):
        return normalize_matrix(self.reference.minv(q))

    def crba(self, q):
        return normalize_matrix(self.reference.crba(q))

    def rnea_grad(self, q, qd, qdd):
        dc_du = normalize_matrix(self.reference.rnea_grad(q, qd, qdd))
        return dc_du[:, : self.nv], dc_du[:, self.nv :]

    def end_effector_pose(self, q, target_name: str, offset=None):
        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        ee_pose = self.reference.end_effector_pose(
            q,
            ee_joint_names=target_name,
            ee_offsets=[np.matrix([offset])],
        )[0]
        return normalize_vector(np.asarray(ee_pose).reshape(-1))


def build_project_adapter(spec, resolved_model, base_mode: str) -> ProjectModelAdapter:
    floating_base = base_mode == "floating"
    robot, parse_output = strict_parse_robot(
        resolved_model.urdf_path,
        floating_base=floating_base,
    )
    mismatches = [
        ConventionMismatch(
            category="parse_behavior",
            detail="URDFParser.parse() suppresses exceptions and returns None instead of surfacing structured errors.",
        ),
        ConventionMismatch(
            category="joint_order",
            detail="Joint order is parser-defined DFS order with optional sibling tie-breaking, so tests must align by joint name explicitly.",
        ),
    ]
    if floating_base:
        mismatches.append(
            ConventionMismatch(
                category="floating_base_quaternion",
                detail="GRiD floating-base configurations use quaternion order wxyz, unlike Pinocchio's free-flyer xyzw convention.",
            )
        )
    return ProjectModelAdapter(
        spec=spec,
        base_mode=base_mode,
        robot=robot,
        reference=RBDReference(robot),
        parse_output=parse_output,
        mismatches=mismatches,
    )


def strict_parse_robot(urdf_path: str, floating_base: bool):
    parser = URDFParser()
    output_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_capture):
            Joint.floating_base = floating_base
            with open(urdf_path, "r", encoding="utf-8") as urdf_file:
                parser.soup = BeautifulSoup(urdf_file.read(), "xml").find("robot")
            if parser.soup is None:
                raise ValueError("URDF file did not contain a <robot> root element.")
            parser.robot = Robot(parser.soup["name"], floating_base, True)
            parser.parse_links()
            parser.parse_joints()
            parser.renumber_linksJoints(using_quaternion=True, alpha_tie_breaker=False)
            parser.print_joint_order()
            robot = copy.deepcopy(parser.robot)
    except Exception as exc:
        raise ProjectParseError(str(exc)) from exc

    return robot, output_capture.getvalue()
