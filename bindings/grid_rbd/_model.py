"""CPU-only model inspection shared with the generation path."""
import logging
from pathlib import Path

_log = logging.getLogger('grid_rbd')


def parse_model(urdf_path, *, floating_base=False):
    from URDFParser import URDFParser

    class LibraryParser(URDFParser):
        def print_joint_order(self):
            # Do not redirect process-global stdout around concurrent parses.
            _log.info('grid_rbd: joint order: %s (nq=%d, nv=%d)',
                      ', '.join(j.name for j in self.robot.get_joints_ordered_by_id()),
                      self.robot.get_num_pos(), self.robot.get_num_vel())

    if type(floating_base) is not bool:
        raise TypeError('floating_base must be bool')
    return LibraryParser().parse(str(Path(urdf_path)), floating_base=floating_base)


def describe_model(urdf_path, *, floating_base=False, runtime_transform=False, schema_version=1):
    """Return schema-1 coordinate/frame metadata without CUDA or compilation.

    The descriptor ID covers coordinate geometry, ownership and names, not
    inertia, actuation, code-generation options or artifact identity. Placements
    are nominal when runtime_transform=True. Unknown schema versions and options
    are errors, never silently ignored. Zero-coordinate metadata is valid but
    does not promise every GPU operation supports a zero-coordinate model.
    """
    from grid_codegen.model_descriptor import SCHEMA_VERSION, descriptor_from_robot
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError('unsupported model descriptor schema version')
    if type(runtime_transform) is not bool:
        raise TypeError('runtime_transform must be bool')
    return descriptor_from_robot(parse_model(urdf_path, floating_base=floating_base),
                                 runtime_transform=runtime_transform)
