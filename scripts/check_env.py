#!/usr/bin/env python3
"""CLI script to run PyTorch environment diagnostics and print summary."""

import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.common.env_check import get_pytorch_environment_info, format_env_summary


def main():
    """Run PyTorch environment check and display formatted report."""
    env_info = get_pytorch_environment_info()
    report = format_env_summary(env_info)
    print(report)


if __name__ == "__main__":
    main()
