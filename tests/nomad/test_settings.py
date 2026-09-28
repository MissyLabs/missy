from __future__ import annotations

import pytest

from missy.config.settings import _parse_nomad
from missy.core.exceptions import ConfigurationError


def test_nomad_config_is_disabled_and_scope_denied_by_default() -> None:
    config = _parse_nomad(None)
    assert config.enabled is False
    assert config.allowed_namespaces == []
    assert config.workload_templates == {}


def test_nomad_config_parses_strict_workload_template() -> None:
    config = _parse_nomad(
        {
            "enabled": True,
            "allowed_namespaces": ["testing"],
            "default_namespace": "testing",
            "max_parallel_jobs": 4,
            "max_benchmark_parallelism": 2,
            "approved_artifact_prefixes": ["artifact://missy/"],
            "workload_templates": {
                "repo": {
                    "description": "repository test",
                    "allowed_parameters": ["repository"],
                    "required_parameters": ["repository"],
                    "request": {
                        "purpose": "repository test",
                        "image": "registry.test/repo@sha256:" + "a" * 64,
                    },
                }
            },
        }
    )
    assert config.default_namespace == "testing"
    assert config.workload_templates["repo"]["required_parameters"] == ["repository"]


def test_nomad_config_rejects_unknown_template_security_key() -> None:
    with pytest.raises(ConfigurationError, match="unrecognized key"):
        _parse_nomad(
            {
                "workload_templates": {
                    "repo": {
                        "request": {},
                        "allowed_parameters": [],
                        "required_parameters": [],
                        "arbitrary_shell": True,
                    }
                }
            }
        )


def test_nomad_config_rejects_benchmark_parallelism_above_global_limit() -> None:
    with pytest.raises(ConfigurationError, match="must not exceed"):
        _parse_nomad({"max_parallel_jobs": 2, "max_benchmark_parallelism": 3})
