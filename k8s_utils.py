"""Pure helpers for the Cleric Query Agent (no I/O — safe to unit-test)."""
from __future__ import annotations

import re


def strip_k8s_suffix(pod_name: str) -> str:
    """Remove a Kubernetes-generated suffix from a pod name.

    Pods created by a Deployment are named ``<name>-<replicaset-hash>-<pod-hash>``
    (e.g. ``frontend-6b5f4cf68c-6g5lt``). Strip the two trailing hash segments so
    callers see ``frontend``. Names that merely contain dashes (e.g.
    ``redis-docker``) are returned unchanged.
    """
    pattern = r"(.*)-[a-z0-9]+-[a-z0-9]+$"
    match = re.match(pattern, pod_name)
    if match:
        return match.group(1)
    return pod_name
