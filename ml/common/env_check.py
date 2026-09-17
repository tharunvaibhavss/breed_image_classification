"""PyTorch environment detection module.

Provides environment diagnostic reporting for Python version, PyTorch version,
CUDA availability, GPU device identification, and CPU fallback verification.
"""

import sys
import platform
from typing import Dict, Any, Optional


def get_pytorch_environment_info() -> Dict[str, Any]:
    """Inspect and return current PyTorch execution environment parameters.

    Returns:
        Dict[str, Any] containing:
            - python_version: System Python version
            - pytorch_version: Installed PyTorch version (or string indicating missing)
            - cuda_available: Boolean flag indicating if CUDA is usable
            - cuda_version: CUDA version string if available, else None
            - device_count: Number of accessible CUDA devices
            - device_name: Name of primary GPU device if available, else None
            - compute_device: Active device type ('cuda:0' or 'cpu')
            - cpu_fallback: Boolean flag indicating fallback to CPU execution
    """
    python_ver = sys.version.split()[0]
    platform_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    try:
        import torch

        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if cuda_avail else None
        dev_count = torch.cuda.device_count() if cuda_avail else 0

        if cuda_avail and dev_count > 0:
            dev_name: Optional[str] = torch.cuda.get_device_name(0)
            compute_device = "cuda:0"
            cpu_fallback = False
        else:
            dev_name = None
            compute_device = "cpu"
            cpu_fallback = True

        return {
            "status": "ok",
            "python_version": python_ver,
            "platform": platform_info,
            "pytorch_version": torch_ver,
            "cuda_available": cuda_avail,
            "cuda_version": cuda_ver,
            "device_count": dev_count,
            "gpu_name": dev_name,
            "compute_device": compute_device,
            "cpu_fallback": cpu_fallback,
        }
    except ImportError:
        return {
            "status": "warning",
            "python_version": python_ver,
            "platform": platform_info,
            "pytorch_version": "Not Installed",
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "gpu_name": None,
            "compute_device": "cpu",
            "cpu_fallback": True,
            "error": "PyTorch package is not installed in current environment.",
        }


def format_env_summary(info: Dict[str, Any]) -> str:
    """Format PyTorch environment info dictionary into human-readable string summary.

    Args:
        info: Dictionary returned by get_pytorch_environment_info()

    Returns:
        Formatted summary string for logging or CLI output.
    """
    summary_lines = [
        "=" * 60,
        " PyTorch Environment Diagnostic Report",
        "=" * 60,
        f" Python Version   : {info.get('python_version')}",
        f" Platform         : {info.get('platform')}",
        f" PyTorch Version  : {info.get('pytorch_version')}",
        f" CUDA Available   : {info.get('cuda_available')}",
    ]

    if info.get("cuda_available"):
        summary_lines.extend(
            [
                f" CUDA Version     : {info.get('cuda_version')}",
                f" GPU Count        : {info.get('device_count')}",
                f" GPU Device Name  : {info.get('gpu_name')}",
                f" Active Device    : {info.get('compute_device')}",
                f" CPU Fallback     : No (Using GPU acceleration)",
            ]
        )
    else:
        summary_lines.extend(
            [
                f" GPU Device Name  : N/A (GPU not available)",
                f" Active Device    : {info.get('compute_device')}",
                f" CPU Fallback     : Yes (Operating on CPU fallback mode)",
            ]
        )

    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)
