# Security Policy

GRiD is a research code-generation library for GPU rigid body dynamics. It
generates and compiles CUDA C++ from URDF inputs; it is not a network service
and does not handle credentials or untrusted remote data.

## Reporting a vulnerability

If you discover a security issue (for example, a code-generation path that could
be abused to emit unsafe host/device code from a crafted URDF), please report it
privately rather than opening a public issue:

- Use GitHub's **"Report a vulnerability"** flow on the
  [Security tab](https://github.com/A2R-Lab/GRiD/security/advisories), or
- Contact the A2R Lab maintainers directly.

Please include a description, reproduction steps, and the affected commit. We
will acknowledge the report and work with you on a fix and disclosure timeline.

## Supported versions

GRiD is developed on the `main` line and has not been formally released; fixes
land on the current development branch.
