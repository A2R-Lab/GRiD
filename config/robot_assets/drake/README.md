# iiwa collision mesh subset

These two meshes are copied unchanged from RobotLocomotion/drake commit
`7abea0556ede980a5077fe1a8cfbae59b57c7c27`, under
`manipulation/models/iiwa_description/meshes/collision/`:

| File | SHA-256 |
|---|---|
| link_6.obj | bfdba14c8462325caec427fe73ace2bbb24e4c604305840371c7b13ed1011ab1 |
| link_7.obj | cafe5f56d19435c9858398856ebb7fca04028bdd801b10bc1193fe0355826b4b |

Source repository: https://github.com/RobotLocomotion/drake/tree/7abea0556ede980a5077fe1a8cfbae59b57c7c27/manipulation/models/iiwa_description

The bundled iiwa14 URDF uses primitive collisions for the other links. This is
the complete external mesh subset needed for its collision model, not a copy of
Drake or its visual meshes. The package-relative directory layout preserves the
original URDF's `package://drake/...` paths. No network or ROS_PACKAGE_PATH setting
is needed to spherize `config/robot_assets/iiwa14.urdf` in this checkout.

The original iiwa_stack notices and Drake modifications/license notice are
retained under `manipulation/models/iiwa_description/`; Drake's root license is
`LICENSE.TXT`. These third-party files retain their own licenses.

Reproduce a complete model from the repository root:

```bash
.venv/bin/python -m grid_codegen.spherize config/robot_assets/iiwa14.urdf \
  --out /tmp/iiwa14_spheres.urdf --mesh-mode bounded-bulge --report
```

The bounded-bulge policy is voxel-based; its sampled report is not a continuous
coverage certificate. The older `grid_codegen/collision/assets/iiwa14_spherized.urdf`
omitted these meshes and is a historical partial model, not a current default.
