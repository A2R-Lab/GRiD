# Vendored URDFs — provenance & sources

Sourced by copy from the `robot_descriptions` package cache
(`~/.cache/robot_descriptions/...`) on 2026-05-29. Vendoring these here
lets the test/equivalence/bench pipelines run without the multi-GB
`robot_descriptions` dependency.

| robot_id | local path | upstream source | sha256[:12] |
|---|---|---|---|
| iiwa14 | `robot_assets/iiwa14.urdf` | `drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf` | `94a986da7077` |
| go2 | `robot_assets/go2.urdf` | `unitree_ros/robots/go2_description/urdf/go2_description.urdf` | `7d19fe48e2e6` |
| g1 | `robot_assets/g1.urdf` | `unitree_ros/robots/g1_description/g1_29dof.urdf` | `e1dc89366bf9` |
| h1_2 | `robot_assets/h1_2.urdf` | `unitree_ros/robots/h1_2_description/h1_2.urdf` | `dcf7f22984a7` |
| fr3 | `robot_assets/fr3.urdf` | `xacrodoc/fr3_description/fr3_description-0b7cd3d638a08e80.urdf` | `d0d99c0ca801` |
| rizon4 | `robot_assets/rizon4.urdf` | `xacrodoc/rizon4_description/rizon4_description-1f7ca11ba02bfe34.urdf` | `0a8db2d7fb5e` |
| gen3 | `robot_assets/gen3.urdf` | `xacrodoc/gen3_description/gen3_description-e9602d5440f026ea.urdf` | `0e7c89432f98` |
| fetch | `robot_assets/fetch.urdf` | `roboschool/roboschool/models_robot/fetch_description/robots/fetch.urdf` | `8a6ce15ab481` |
| baxter | `robot_assets/baxter.urdf` | `baxter_common/baxter_description/urdf/baxter.urdf` | `ab936bfb412f` |
