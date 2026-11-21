"""The stale-extension check the receipt launcher runs before it starts (CPU only)."""
import types

from test.installed_extension import bound_names, problems

SOURCE = '''
    py::class_<Runner>(m, "Runner")
        .def("set_armature_params", &Runner::set)
        .def("get_armature_params", &Runner::get)
        .def_property_readonly("num_joints", &Runner::nj);
    m.def("build_info", &info);
'''
PYPROJECT = 'name = "grid-rbd"\nversion = "0.7.2"\n'


def _module(runner_attrs, module_attrs=("build_info",)):
    module = types.ModuleType("fake_core")
    module.Runner = type("Runner", (), {name: None for name in runner_attrs})
    for name in module_attrs:
        setattr(module, name, None)
    return module


def test_bound_names_reads_every_kind_of_binding():
    assert bound_names(SOURCE) == {"set_armature_params", "get_armature_params", "num_joints", "build_info"}


def test_current_extension_has_no_problems():
    module = _module(("set_armature_params", "get_armature_params", "num_joints"))
    assert problems(SOURCE, PYPROJECT, module, "0.7.2") == []


def test_extension_built_before_a_binding_was_added_is_named():
    module = _module(("set_armature_params", "num_joints"))
    (found,) = problems(SOURCE, PYPROJECT, module, "0.7.2")
    assert "get_armature_params" in found and "lacks 1 binding" in found


def test_stale_installed_version_is_named():
    module = _module(("set_armature_params", "get_armature_params", "num_joints"))
    (found,) = problems(SOURCE, PYPROJECT, module, "0.7.0")
    assert "0.7.0" in found and "0.7.2" in found
