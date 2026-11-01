"""Minimal real descriptor emission for CPU-only mocked consumer builds."""
import json


def generate_metadata_header(urdf, options, out):
    from grid_rbd import describe_model
    from grid_codegen.model_metadata_cpp import emit_model_metadata
    descriptor = describe_model(urdf, floating_base=options.get('floating_base', False),
                                runtime_transform=options.get('runtime_transform', False))
    out.write_text(emit_model_metadata(descriptor))
    (out.parent/'model_descriptor.json').write_text(json.dumps(descriptor))


class FakeToolchain:
    """A resolved consumer toolchain for CPU tests: no host compiler is probed."""
    identity = {"host": "fake-host", "family": "fake"}
    prepend = []

    def compile_flags(self):
        return []

    def environment(self):
        return None

    def link_command(self, nvcc, out, objects):
        return [nvcc, "--shared", "-o", str(out), *objects]
