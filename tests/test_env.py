"""Unit tests for PyTorch environment detection module."""

from ml.common.env_check import get_pytorch_environment_info, format_env_summary


def test_pytorch_env_detection_structure():
    """Verify PyTorch environment check returns complete structure."""
    info = get_pytorch_environment_info()

    assert "python_version" in info
    assert "platform" in info
    assert "pytorch_version" in info
    assert "cuda_available" in info
    assert "compute_device" in info
    assert "cpu_fallback" in info
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["cpu_fallback"], bool)


def test_format_env_summary():
    """Verify formatting function returns non-empty formatted text."""
    info = get_pytorch_environment_info()
    summary = format_env_summary(info)

    assert isinstance(summary, str)
    assert "PyTorch Environment Diagnostic Report" in summary
    assert info["python_version"] in summary
