"""Pure helpers for the Cleric Query Agent (no I/O — safe to unit-test)."""
from __future__ import annotations

import re

_SUFFIX_RE = re.compile(r"^(.*)-([a-z0-9]+)-([a-z0-9]+)$")


def _looks_like_hash(segment: str) -> bool:
    """Kubernetes-generated hash segments always contain a digit; real words
    (``app``, ``gateway``, ...) usually don't. Used to avoid over-stripping."""
    return any(ch.isdigit() for ch in segment)


def strip_k8s_suffix(pod_name: str) -> str:
    """Remove a Kubernetes-generated suffix from a pod name.

    Pods created by a Deployment are named ``<name>-<replicaset-hash>-<pod-hash>``
    (e.g. ``frontend-6b5f4cf68c-6g5lt``). Strip the two trailing hash segments so
    callers see ``frontend``.

    The two trailing segments are only stripped when they *look like* generated
    hashes (contain a digit), so ordinary hyphenated names such as ``redis-docker``
    or ``my-app-v2`` are returned unchanged.
    """
    match = _SUFFIX_RE.match(pod_name)
    if match and _looks_like_hash(match.group(2)) and _looks_like_hash(match.group(3)):
        return match.group(1)
    return pod_name
