"""Tinker client subpackage for Metacog (real SDK wrapper).

Note: this package is named `tink` (not `tinker`) to avoid shadowing the
real `tinker` SDK that we wrap. The SDK is installed in .venv and lives
at `tinker/`. We import it as `import tinker` here.
"""

from .client import MetacogConfig, SampleResult, TinkerClient, health_check

__all__ = ["MetacogConfig", "SampleResult", "TinkerClient", "health_check"]
