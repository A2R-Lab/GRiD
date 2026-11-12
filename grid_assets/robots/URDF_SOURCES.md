# Vendored URDFs: provenance and sources

These files were copied from the `robot_descriptions` package cache
(`~/.cache/robot_descriptions/...`) on 2026-05-29. Vendoring them here
lets the test/equivalence/bench pipelines run without the multi-GB
`robot_descriptions` dependency.

| robot_id | local path | upstream source | sha256[:12] |
|---|---|---|---|
| iiwa14 | `grid_assets/robots/iiwa14.urdf` | `drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf` | `94a986da7077` |
| go2 | `grid_assets/robots/go2.urdf` | `unitree_ros/robots/go2_description/urdf/go2_description.urdf` | `7d19fe48e2e6` |
| g1 | `grid_assets/robots/g1.urdf` | `unitree_ros/robots/g1_description/g1_29dof.urdf` | `e1dc89366bf9` |
| h1_2 | `grid_assets/robots/h1_2.urdf` | `unitree_ros/robots/h1_2_description/h1_2.urdf` | `dcf7f22984a7` |
| fr3 | `grid_assets/robots/fr3.urdf` | `xacrodoc/fr3_description/fr3_description-0b7cd3d638a08e80.urdf` | `d0d99c0ca801` |
| rizon4 | `grid_assets/robots/rizon4.urdf` | `xacrodoc/rizon4_description/rizon4_description-1f7ca11ba02bfe34.urdf` (kinematics, limits, meshes); inertials from `flexivrobotics/flexiv_description` branch `jazzy` at `7d5dd28c` (2026-10-19), `config/Rizon4/inertia_parameters.yaml`, Apache-2.0 | `07dd55880066` |
| gen3 | `grid_assets/robots/gen3.urdf` | `xacrodoc/gen3_description/gen3_description-e9602d5440f026ea.urdf` | `0e7c89432f98` |
| fetch | `grid_assets/robots/fetch.urdf` | `roboschool/roboschool/models_robot/fetch_description/robots/fetch.urdf` | `8a6ce15ab481` |
| baxter | `grid_assets/robots/baxter.urdf` | `baxter_common/baxter_description/urdf/baxter.urdf` | `ab936bfb412f` |
| h2_plus | `grid_assets/robots/h2_plus.urdf` | vendored H2+ extended-humanoid model (nv=81 scaling testbed; no public MJCF/URDF upstream — in-repo vendored asset) | `212533d728ca` |

## Local modifications

- rizon4: the xacrodoc expansion of the old flexiv xacro emitted each
  link's `<origin>/<mass>/<inertia>` triple bare (not wrapped in
  `<inertial>`), so every link parsed massless and the local copy used to
  wrap eight rounded triples by hand. On 2026-11-03 the eight `<inertial>`
  elements were rewritten from the vendor's current table,
  `config/Rizon4/inertia_parameters.yaml` of `flexiv_description` (branch
  `jazzy`, commit `7d5dd28c`, 2026-10-19), which the vendor's
  `get_inertias` xacro macro wraps into exactly this `<inertial>` form:
  centre of mass, mass and the six inertia components per link, verbatim.
  Joint origins, axes and limits were checked against the same commit's
  `default_kinematics.yaml` and `joint_limits.yaml` and are unchanged.
  Flexiv's manual states the description's dynamics parameters are
  approximate by design; they are the vendor's published values. The sha
  column above still names the unmodified xacrodoc file.
