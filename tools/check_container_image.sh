#!/usr/bin/env bash
# Post-publication smoke, not a source receipt. Run with sudo if Docker needs it.
set -euo pipefail
if [[ $# != 1 || ! "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Usage: bash $0 VERSION (for example 0.5.3)" >&2
    exit 2
fi
task_version=$1
task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for task_tool in docker flock nvidia-smi pgrep; do
    command -v "$task_tool" >/dev/null || { echo "Missing prerequisite: $task_tool" >&2; exit 1; }
done
task_out=$(mktemp -d "${TMPDIR:-/tmp}/grid-container-smoke-${task_version}-XXXXXX")
chmod 0755 "$task_out"
exec > >(tee "$task_out/smoke.log") 2>&1
trap 'task_rc=$?; echo "Container smoke exit=$task_rc; logs: $task_out"' EXIT
echo "OUTPUT=$task_out"
# An empty CLI config prevents a maintainer's saved credentials from masking a
# private package. Never log in or modify the user's Docker configuration.
mkdir "$task_out/anonymous-docker-config"
task_docker=(docker --config "$task_out/anonymous-docker-config")
"${task_docker[@]}" pull "ghcr.io/a2r-lab/grid:$task_version"
task_image=$("${task_docker[@]}" image inspect "ghcr.io/a2r-lab/grid:$task_version" --format '{{index .RepoDigests 0}}')
[[ "$task_image" =~ ^ghcr.io/a2r-lab/grid@sha256:[a-f0-9]{64}$ ]] || {
    echo "Unexpected image identity: $task_image"; exit 1;
}
echo "IMAGE=$task_image"
"${task_docker[@]}" image inspect "$task_image"
# Use the existing lab-wide lock. Do not create a root-owned lock or change its
# permissions. A read-only open avoids fs.protected_regular's O_CREAT rejection
# in sticky /tmp; Linux flock can still hold an exclusive lock on this fd.
task_lock=${GRID_CONTAINER_TIMING_LOCK:-/tmp/a2rlab-timing.lock}
[[ -f "$task_lock" && ! -L "$task_lock" ]] || {
    echo "Expected existing regular shared lock missing: $task_lock"; exit 1;
}
exec 9<"$task_lock"
flock -n -x 9 || { echo 'Shared timing lock busy; no GPU test launched.'; exit 1; }
task_compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$task_compute" ]] || { echo "GPU compute jobs active: $task_compute"; exit 1; }
if pgrep -x 'nvcc|ptxas|pytest' >/dev/null; then
    echo 'Compile/test process present; stop before GPU checks.'; exit 1
fi
"${task_docker[@]}" run --rm --network none --gpus all "$task_image" nvidia-smi
task_mounts=(--mount "type=bind,source=$task_out,target=/work"
    --mount "type=bind,source=$task_repo/tools/check_installed_package.py,target=/checks/check_installed_package.py,readonly"
    --mount "type=bind,source=$task_repo/tools/check_container_baked_robot.py,target=/checks/baked.py,readonly")
"${task_docker[@]}" run --rm --network none --gpus all "${task_mounts[@]}" "$task_image" \
    python /checks/baked.py --expected-version "$task_version"
"${task_docker[@]}" run --rm --network none --gpus all -e GRID_RBD_BUILD_JOBS=2 "${task_mounts[@]}" "$task_image" \
    python /checks/check_installed_package.py --work /work/probe --gpu
"${task_docker[@]}" run --rm --network none --gpus all "${task_mounts[@]}" "$task_image" \
    python /checks/check_installed_package.py --work /work/probe --gpu --reload
touch "$task_out/PASS"
echo 'PASS container GPU smoke: anonymous pull, baked iiwa14, installed provenance, three-backend values/autograd, cache-only reload.'
