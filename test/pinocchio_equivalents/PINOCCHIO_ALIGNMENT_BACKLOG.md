# Pinocchio Alignment Backlog

This backlog records small, safe, additive improvements that would make
`URDFParser` and `RBDReference` easier to validate against Pinocchio and easier to
use from a Pinocchio-oriented workflow.

## Parsing And Testability Improvements For URDFParser

- Add an additive strict parse API or a `raise_on_error` option so parse failures
  can surface structured exceptions instead of returning `None`.
- Return parse diagnostics that include the URDF path, root link, and failing stage
  rather than only printing to stdout.
- Expose parse-order metadata directly:
  resolved root link
  joint traversal policy
  original URDF joint ordering
  final internal joint ordering
- Add stable helpers for:
  joint names
  movable joint names
  joint limits
  fixed-joint retention/merge results
- Make parser stdout reporting optional so tests and downstream tools can opt into
  quieter machine-readable behavior.

Observed smoke-corpus pain points in this checkout:

- `iiwa14` fixed-base strict parse hit a SymPy parse failure on an empty expression.
- `go2` fixed-base strict parse hit an index error during GRiD-side parse flow.
- `g1` fixed-base strict parse hit a missing joint-origin assumption (`NoneType`
  subscript) in `parse_joints`.

## User-Facing Convention Improvements For URDFParser

- Add additive Pinocchio-friendly metadata helpers such as `nq`, `nv`, and a clear
  base-joint description without removing the current API.
- Make floating-base state conventions explicit in the Python interface and docs,
  including quaternion ordering.
- Clarify how fixed joints are merged and how that affects the model structure that
  downstream algorithms see.
- Expose joint and parent/child indexing in a way that is easier to compare with
  Pinocchio's model semantics.

## User-Facing Convention Improvements For RBDReference

- Add Pinocchio-shaped convenience wrappers for the primary outputs of:
  inverse dynamics
  mass matrix workflows
  gravity or bias terms when semantics are clarified
- Standardize public returns on `numpy.ndarray` instead of mixing arrays, matrices,
  and bundled tuples where a narrow primary result would be clearer.
- Make floating-base input expectations explicit and queryable, especially for
  quaternion order and free-flyer state layout.
- Mark methods that are fixed-base only or floating-base incomplete in the public
  Python docs.
- Prefer additive wrappers and metadata accessors over breaking changes so existing
  GRiD users are not disrupted.
