"""Read-only build resource discovery and session-local PostgreSQL tuning.

Run this module to inspect the machine without starting an index build.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cpu_budget(*, affinity=None, cgroup_root=Path('/sys/fs/cgroup'),
               proc_cgroup=Path('/proc/self/cgroup')) -> dict:
    if affinity is None:
        affinity = (os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity')
                    else range(os.cpu_count() or 1))
    affinity = sorted(affinity)
    quotas = []
    membership = _read(proc_cgroup) or ''
    for line in membership.splitlines():
        if not line.startswith('0::'):
            continue
        relative = line.split('::', 1)[1].lstrip('/')
        if '..' in Path(relative).parts:
            continue
        current = cgroup_root / relative
        while True:
            raw = _read(current / 'cpu.max')
            if raw:
                quota, period = raw.split()
                if quota != 'max':
                    quotas.append(int(quota) / int(period))
            if current == cgroup_root:
                break
            current = current.parent
    quota = min(quotas) if quotas else None
    available = max(1, min(len(affinity), math.floor(quota) if quota is not None else len(affinity)))
    physical = set()
    for cpu in affinity:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        package, core = _read(topology / 'physical_package_id'), _read(topology / 'core_id')
        if package is not None and core is not None:
            physical.add((package, core))
    # A bounded starting point, not a measured optimum or a claim of exclusivity.
    automatic = min(32, available, len(physical) or available)
    return dict(logical_cpus=os.cpu_count(), affinity_cpu_count=len(affinity),
                physical_cores_in_affinity=len(physical) or None,
                cgroup_cpu_quota=quota, effective_cpu_budget=available,
                auto_build_threads=automatic)


def build_threads(value: str | int) -> int:
    budget = cpu_budget()
    if str(value).lower() == 'auto':
        return budget['auto_build_threads']
    count = int(value)
    if count < 1 or count > budget['effective_cpu_budget']:
        raise ValueError(f"build threads must be 1..{budget['effective_cpu_budget']} or auto")
    return count


def configure_pg_build(connection, *, workers: str | int,
                       maintenance_memory_mb: int) -> dict:
    """Tune only this build connection; never ALTER SYSTEM or restart a server.

    workers counts parallel workers in addition to the leader. Actual launched
    workers can be fewer, and depend on the index access method and server load.
    """
    if maintenance_memory_mb < 1:
        raise ValueError('maintenance memory must be a positive number of MiB')
    requested = (min(8, max(0, cpu_budget()['auto_build_threads'] - 1))
                 if str(workers).lower() == 'auto' else int(workers))
    if requested < 0:
        raise ValueError('PostgreSQL build workers must be nonnegative or auto')
    limits = {name: int(value) for name, value in connection.execute(
        "SELECT name,setting FROM pg_settings WHERE name IN "
        "('max_parallel_workers','max_worker_processes')").fetchall()}
    actual = min(requested, limits['max_parallel_workers'], limits['max_worker_processes'])
    connection.execute("SELECT set_config('max_parallel_maintenance_workers', %s, false)", (str(actual),))
    connection.execute("SELECT set_config('maintenance_work_mem', %s, false)", (f'{maintenance_memory_mb}MB',))
    return dict(requested_parallel_workers=requested, configured_parallel_workers=actual,
                maintenance_memory_mb=maintenance_memory_mb, server_limits=limits)


def main():
    memory = {}
    for line in (_read(Path('/proc/meminfo')) or '').splitlines():
        name, value = line.split(':', 1)
        if name in {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'}:
            memory[name + '_KiB'] = int(value.split()[0])
    print(json.dumps(dict(cpu=cpu_budget(), host_memory=memory,
                          note='auto is a conservative starting point; no build is started'), indent=2))


if __name__ == '__main__':
    main()
