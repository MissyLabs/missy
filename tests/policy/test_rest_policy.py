"""Tests for missy.policy.rest_policy.RestPolicy."""

from __future__ import annotations

from missy.policy.rest_policy import RestPolicy, RestRule


class TestRestPolicy:
    """Tests for L7 REST policy enforcement."""

    def test_allow_get_deny_delete(self):
        """GET is allowed but DELETE is denied on the same host."""
        policy = RestPolicy.from_config(
            [
                {"host": "api.github.com", "method": "GET", "path": "/repos/**", "action": "allow"},
                {"host": "api.github.com", "method": "DELETE", "path": "/**", "action": "deny"},
            ]
        )
        assert policy.check("api.github.com", "GET", "/repos/foo") == "allow"
        assert policy.check("api.github.com", "DELETE", "/repos/foo") == "deny"

    def test_glob_path_matching(self):
        """/repos/** matches /repos/foo/bar."""
        policy = RestPolicy.from_config(
            [
                {"host": "api.github.com", "method": "GET", "path": "/repos/**", "action": "allow"},
            ]
        )
        assert policy.check("api.github.com", "GET", "/repos/foo/bar") == "allow"
        # Non-matching path returns None (pass-through)
        assert policy.check("api.github.com", "GET", "/users/foo") is None

    def test_no_rules_passes(self):
        """Empty rules means everything passes through (returns None)."""
        policy = RestPolicy.from_config([])
        assert policy.check("api.github.com", "GET", "/repos") is None

        policy_none = RestPolicy()
        assert policy_none.check("example.com", "POST", "/data") is None

    def test_first_match_wins(self):
        """Rules are evaluated top-to-bottom; first match wins."""
        policy = RestPolicy.from_config(
            [
                {"host": "api.example.com", "method": "GET", "path": "/admin/**", "action": "deny"},
                {
                    "host": "api.example.com",
                    "method": "GET",
                    "path": "/admin/**",
                    "action": "allow",
                },
            ]
        )
        # First rule (deny) should win
        assert policy.check("api.example.com", "GET", "/admin/users") == "deny"

    def test_method_wildcard(self):
        """method: '*' matches all HTTP methods."""
        policy = RestPolicy.from_config(
            [
                {"host": "api.example.com", "method": "*", "path": "/health", "action": "allow"},
            ]
        )
        assert policy.check("api.example.com", "GET", "/health") == "allow"
        assert policy.check("api.example.com", "POST", "/health") == "allow"
        assert policy.check("api.example.com", "DELETE", "/health") == "allow"
        assert policy.check("api.example.com", "PATCH", "/health") == "allow"

    def test_host_case_insensitive(self):
        """Host matching is case-insensitive."""
        policy = RestPolicy.from_config(
            [
                {"host": "API.GitHub.COM", "method": "GET", "path": "/**", "action": "allow"},
            ]
        )
        assert policy.check("api.github.com", "GET", "/repos") == "allow"

    def test_method_case_insensitive(self):
        """Method matching is case-insensitive."""
        policy = RestPolicy.from_config(
            [
                {"host": "api.example.com", "method": "get", "path": "/**", "action": "allow"},
            ]
        )
        assert policy.check("api.example.com", "GET", "/foo") == "allow"

    def test_no_match_different_host(self):
        """Rules for one host do not affect another."""
        policy = RestPolicy.from_config(
            [
                {"host": "api.github.com", "method": "GET", "path": "/**", "action": "deny"},
            ]
        )
        assert policy.check("api.example.com", "GET", "/foo") is None

    def test_from_rest_rule_objects(self):
        """RestPolicy can be created directly from RestRule objects."""
        rules = [
            RestRule(host="api.example.com", method="POST", path="/data", action="allow"),
        ]
        policy = RestPolicy(rules)
        assert policy.check("api.example.com", "POST", "/data") == "allow"
