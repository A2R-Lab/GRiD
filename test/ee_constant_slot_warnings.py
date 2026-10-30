"""Fail-closed classification of the diagnosed Package 3 EE constant-slot warning.

Not a general racecheck suppression. Applies only to the three reviewed FP32
fixtures with --racecheck-report all, source audit and separate numerical and
repeatability gates. All raw logs remain evidence; disposition is KNOWN_WARNING,
never zero-hazard PASS. Brian approved this disposition on 2026-10-30.
"""
import re

CASES = {
    'test_exact_target_scratch_floa0': ('end_effector_pose_inner_iiwa_joint_ee', 24, 3),
    'test_exact_target_scratch_floa1': ('end_effector_pose_inner_FR_foot_joint', 16, 2),
    'test_exact_target_scratch_floa2': ('end_effector_pose_inner', 8, 1),
}


def known_constant_slot_warning(text, case):
    if case not in CASES or re.search(r'^========= (Error:|ERROR|Fatal|Invalid)', text, re.M):
        return False
    target, accesses, groups = CASES[case]
    summary = re.findall(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)', text)
    total = accesses + groups
    if summary != [(str(total), '0', str(total))]:
        return False
    warnings = re.findall(r'^========= Warning: (.*?)(?=^========= Warning:|^========= RACECHECK SUMMARY:|\Z)', text, re.M | re.S)
    if len(warnings) != total:
        return False
    details = analyses = 0
    for warning in warnings:
        if 'grid::glass::dot_strided<float,' not in warning or f'grid::{target}<float,' not in warning:
            return False
        if warning.startswith('(Warp Level Programming) Potential WAR hazard detected at __shared__'):
            pairs = re.findall(r'Current Value : (\d+), Incoming Value : (\d+)', warning)
            if len(pairs) != 1 or pairs[0][0] != pairs[0][1] or int(pairs[0][0]) not in (0, 63, 128):
                return False
            if 'Write Thread' not in warning or 'Read Thread' not in warning:
                return False
            details += 1
        elif warning.startswith('Race reported between Read access at ') and '[8 hazards]' in warning:
            analyses += 1
        else:
            return False
    return (details, analyses) == (accesses, groups)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.exit(0 if known_constant_slot_warning(Path(sys.argv[1]).read_text(), sys.argv[2]) else 1)
