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


def test_preserves_hyphenated_version_name():
    # Regression: "my-app-v2" must not be stripped to "my" — the trailing
    # segments aren't hashes (no digit in "app").
    assert strip_k8s_suffix("my-app-v2") == "my-app-v2"


def test_preserves_multiword_wordy_name():
    assert strip_k8s_suffix("api-gateway-service") == "api-gateway-service"
