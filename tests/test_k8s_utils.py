from k8s_utils import strip_k8s_suffix


def test_strips_deployment_pod_suffix():
    assert strip_k8s_suffix("frontend-6b5f4cf68c-6g5lt") == "frontend"


def test_strips_multiword_name_suffix():
    assert strip_k8s_suffix("redis-leader-7d9fbcf5b-abc12") == "redis-leader"


def test_preserves_plain_name():
    assert strip_k8s_suffix("nginx") == "nginx"


def test_preserves_single_dash_name():
    # Only one dash: not a generated suffix, leave it alone.
    assert strip_k8s_suffix("redis-docker") == "redis-docker"
