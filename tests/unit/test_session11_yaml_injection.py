"""Session 11: Tests for YAML injection prevention in wizard config builder.

Validates that _yaml_safe_value() properly escapes special characters in
user-supplied values (API keys, tokens, URLs) to prevent YAML injection.
"""

from __future__ import annotations

import yaml

from missy.cli.wizard import _yaml_safe_value


class TestYamlSafeValue:
    """Test _yaml_safe_value escaping."""

    def test_simple_string_double_quoted(self):
        """Simple strings should be double-quoted."""
        assert _yaml_safe_value("sk-abc123") == '"sk-abc123"'

    def test_string_with_double_quotes_uses_single_quotes(self):
        """Strings with double quotes should be single-quoted."""
        result = _yaml_safe_value('key"with"quotes')
        assert result.startswith("'")
        # Must be valid YAML
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == 'key"with"quotes'

    def test_string_with_newline_escaped(self):
        """Strings with newlines must not create new YAML keys."""
        result = _yaml_safe_value("key\ninjected: true")
        parsed = yaml.safe_load(f"val: {result}")
        # The critical security property: no additional keys created
        assert "injected" not in str(parsed.keys())
        assert len(parsed) == 1
        assert "key" in parsed["val"]

    def test_string_with_colon_space_escaped(self):
        """Strings with ': ' must not create new YAML keys."""
        result = _yaml_safe_value("value: injected")
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "value: injected"

    def test_string_with_hash_escaped(self):
        """Strings with # must not create YAML comments."""
        result = _yaml_safe_value("key#comment")
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "key#comment"

    def test_string_with_braces_escaped(self):
        """Strings with {} must not create YAML flow mappings."""
        result = _yaml_safe_value("{injected: true}")
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "{injected: true}"

    def test_string_with_brackets_escaped(self):
        """Strings with [] must not create YAML flow sequences."""
        result = _yaml_safe_value("[1, 2, 3]")
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "[1, 2, 3]"

    def test_string_with_single_quotes(self):
        """Strings with single quotes should be properly escaped."""
        result = _yaml_safe_value("it's a key")
        # When single-quoted, internal single quotes are doubled
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "it's a key"

    def test_string_with_backslash_escaped(self):
        """Strings with backslashes must be preserved."""
        result = _yaml_safe_value("path\\to\\file")
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "path\\to\\file"

    def test_string_with_ampersand_escaped(self):
        """Strings with & must not create YAML anchors."""
        result = _yaml_safe_value("&anchor_name")
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "&anchor_name"

    def test_string_with_asterisk_escaped(self):
        """Strings with * must not create YAML aliases."""
        result = _yaml_safe_value("*alias")
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "*alias"

    def test_string_with_exclamation_escaped(self):
        """Strings with ! must not create YAML tags."""
        result = _yaml_safe_value("!python/object:evil")
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "!python/object:evil"

    def test_typical_api_key_unchanged(self):
        """Normal API keys should be safely handled."""
        key = "sk-ant-api03-abcdef1234567890"
        result = _yaml_safe_value(key)
        assert result == f'"{key}"'

    def test_empty_string(self):
        """Empty strings should be quoted."""
        result = _yaml_safe_value("")
        assert result == '""'

    def test_percent_escaped(self):
        """Strings with % must be properly escaped."""
        result = _yaml_safe_value("100%")
        parsed = yaml.safe_load(f"val: {result}")
        assert parsed["val"] == "100%"
