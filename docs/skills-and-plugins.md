# Skills and Plugins

Missy has two extension mechanisms: **skills** (lightweight, in-process) and
**plugins** (external, security-gated).  This document covers both, including
how to write and configure them.

---

## Terminology

| Term | Definition |
|---|---|
| **Tool** | A callable function registered with the tool registry (`missy/tools/`) that the agent can invoke during a conversation.  Tools are the lowest-level callable unit. |
| **Skill** | A higher-level, self-describing callable that declares permissions, a version, and a description. Skills are library objects until an application explicitly constructs, registers, and authorizes them. |
| **Plugin** | An externally-loaded component that extends Missy's capabilities.  Plugins are disabled by default and must be explicitly enabled and allowlisted in the configuration file. |

The key distinction: **skills are in-process library code** registered
programmatically behind an explicit permission authorizer, while **plugins are
external code** that must pass through security gates before loading. Merely
being shipped or discovered does not make a skill executable.

---

## Skills

### BaseSkill Interface

All skills extend `BaseSkill` (`missy/skills/base.py`):

```python
from missy.skills.base import BaseSkill, SkillPermissions, SkillResult

class MySkill(BaseSkill):
    name = "my_skill"                    # Unique registry key
    description = "Does something."      # One-line help text
    version = "0.1.0"                    # Semantic version (default)
    permissions = SkillPermissions()     # Required resources (all False by default)

    def execute(self, **kwargs) -> SkillResult:
        # Perform the skill's action
        return SkillResult(success=True, output="result data")
```

### Class Attributes

| Attribute | Type | Required | Description |
|---|---|---|---|
| `name` | str | Yes | Unique identifier used as the registry key |
| `description` | str | Yes | One-line description shown in help text |
| `version` | str | No | Semantic version string (default: `"0.1.0"`) |
| `permissions` | `SkillPermissions` | Yes | Declares which resources the skill needs |

### SkillPermissions

Declares what resources a skill requires at execution time.  All permissions
default to `False`.

| Permission | Type | Default | Description |
|---|---|---|---|
| `network` | bool | `False` | Skill may make outbound network requests |
| `filesystem_read` | bool | `False` | Skill may read from the filesystem |
| `filesystem_write` | bool | `False` | Skill may write to the filesystem |
| `shell` | bool | `False` | Skill may execute shell commands |

### SkillResult

The return type from `execute()`:

| Field | Type | Description |
|---|---|---|
| `success` | bool | `True` if the skill executed without error |
| `output` | Any | The skill's return value (JSON-serialisable) |
| `error` | str | Error description; empty string on success |

### Skill Registry

Skills are managed by `SkillRegistry` (`missy/skills/registry.py`):

```python
from missy.skills.registry import init_skill_registry

def authorize_skill(name, permissions):
    # Apply the application's real policy to every declared permission.
    return name == "my_skill" and not permissions.shell

registry = init_skill_registry(permission_authorizer=authorize_skill)
registry.register(MySkill())

# Execute a skill by name
result = registry.execute("my_skill", session_id="s1", task_id="t1", text="hello")
```

Key methods:

- `register(skill)` -- add a skill instance to the registry
- `get(name)` -- look up a skill by name (returns `None` if not found)
- `list_skills()` -- return sorted list of all registered skill names
- `execute(name, **kwargs)` -- execute a skill and emit audit events

Every execution attempt emits an audit event with `event_type: "skill.execute"`
and `category: "skill"`. A registry created without a permission authorizer
fails closed: the skill implementation is not called and the result is a
policy denial. Failed executions return a `SkillResult` with `success=False`
rather than raising.

### Example Skill Implementation

```python
from missy.skills.base import BaseSkill, SkillPermissions, SkillResult

class SystemInfoSkill(BaseSkill):
    name = "system_info"
    description = "Return basic system information."
    version = "1.0.0"
    permissions = SkillPermissions(
        filesystem_read=True,   # Reads /etc/os-release, etc.
    )

    def execute(self, **kwargs) -> SkillResult:
        import platform

        info = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        }
        return SkillResult(success=True, output=info)
```

### Runtime ownership and the CLI inventory

The gateway, API server, and shipped service bootstraps currently do **not**
initialize a production `SkillRegistry`. The six shipped built-ins are
therefore explicitly **library-only**:

- `SystemInfoSkill`
- `DateTimeSkill`
- `ConfigShowSkill`
- `HealthCheckSkill`
- `SummarizeSessionSkill`
- `WorkspaceListSkill`

They may be constructed directly by tests or by an embedding application, but
they are not exposed to the model, Discord, or the gateway tool loop. There is
no supported live invocation surface until a bootstrap owner registers them
with a real permission authorizer.

The CLI command below starts a new, isolated process and reports only the empty
registry it constructs. It cannot inspect or serialize a different running
process's registry and labels that authority in its output.

```bash
missy skills list
```

This is not a remote gateway inventory. `missy skills scan` is likewise only
metadata discovery and never activates a manifest.

---

## SKILL.md Dynamic Skill Discovery

### Overview

Alongside the in-process `BaseSkill` mechanism above, Missy supports the
open **SKILL.md** format for cross-agent portable skills — the same
convention used by other agent frameworks (including this repo's own
`.claude/skills/`-style directories).  `SkillDiscovery`
(`missy/skills/discovery.py`) scans a directory tree for `SKILL.md` files
and parses each into a `SkillManifest`.

**Relationship to `BaseSkill`:** these are two independent mechanisms, not
layers of the same system.  `BaseSkill` skills are trusted Python code
registered with `SkillRegistry` and directly executable via
`registry.execute(...)`.  `SkillDiscovery` only *discovers and parses*
`SKILL.md` files — it currently has no wiring into `SkillRegistry` or the
agent's tool-execution loop.  In practice, `SkillDiscovery` is used today
by the `missy skills scan` CLI command to enumerate what SKILL.md-format
skills are available on disk (e.g. as documentation/instructions an
operator or agent can read), not as a code-execution path.

### SKILL.md Format

A SKILL.md file is Markdown with a YAML frontmatter block delimited by
`---` lines, followed by free-form instructions in the body:

```markdown
---
name: web-search
description: Search the web using DuckDuckGo
version: 1.0.0
author: MissyLabs
tools: [web_fetch]
---

# Instructions
When the user asks to search the web...
```

| Frontmatter field | Required | Description |
|---|---|---|
| `name` | Yes | Short identifier for the skill. Parsing raises `ValueError` if missing. |
| `description` | No | Human-readable one-line summary. |
| `version` | No | Semantic version string. |
| `author` | No | Author or organisation name. |
| `tools` | No | YAML inline list of validated tool names the skill expects to use (`[a, b]`). Non-list values and ambiguous/control-character names are rejected. |

The body (everything after the closing `---`) becomes
`SkillManifest.instructions` — free-form markdown intended to be read by
an agent or operator, not machine-executed.

`SkillDiscovery` implements its own minimal YAML-subset parser for the
frontmatter (`_parse_yaml`) so that PyYAML is not a hard dependency for
this feature; it handles simple `key: value` pairs and inline lists only.

### Directory Scanning

```python
from missy.skills.discovery import SkillDiscovery

discovery = SkillDiscovery()
manifests = discovery.scan_directory("~/.missy/skills")   # recursive rglob("SKILL.md")
```

- The default directory is `~/.missy/skills`
  (`missy.skills.discovery.DEFAULT_SKILLS_DIR`).
- Scanning is recursive (`Path.rglob("SKILL.md")`); any `SKILL.md` file
  found at any depth under the given root is parsed.
- Files that fail to parse (missing frontmatter, missing `name`) are
  skipped with a logged warning rather than aborting the whole scan.
- A non-existent directory returns an empty list rather than raising.

### Fuzzy Search

```python
results = discovery.search("web", manifests)
```

`search(query, skills)` performs an NFKC-normalized, case-folded substring
match against `name` first, then `description`; skills with a name match are
ranked ahead of description-only matches. Results are capped at 100, including
for an empty query, and distinct confusable scripts are not aliased.

### CLI: `missy skills scan`

```bash
missy skills scan                        # scans ~/.missy/skills
missy skills scan --path ./my-skills     # scan an explicit directory
```

Lists every discovered SKILL.md file in a table (name, version, author,
description, tools). This is purely a discovery/listing command — it does
not register, enable, or execute anything.

---

## Plugins

### Overview

Plugins are externally-loaded components that are **disabled by default**.
They must pass two security gates before loading:

1. `config.plugins.enabled` must be `true`.
2. The plugin's `name` must appear in `config.plugins.allowed_plugins`.

### BasePlugin Interface

All plugins extend `BasePlugin` (`missy/plugins/base.py`):

```python
from missy.plugins.base import BasePlugin, PluginPermissions

class WeatherPlugin(BasePlugin):
    name = "weather"
    description = "Fetches weather data from an external API."
    version = "1.0.0"
    permissions = PluginPermissions(
        network=True,
        allowed_hosts=["api.openweathermap.org"],
    )

    def initialize(self) -> bool:
        # Validate API key, set up HTTP client, etc.
        # Return True on success, False on failure.
        return True

    def execute(self, **kwargs):
        # Perform the plugin's action
        location = kwargs.get("location", "")
        return {"temperature": 22, "location": location}
```

### Class Attributes

| Attribute | Type | Required | Description |
|---|---|---|---|
| `name` | str | Yes | Unique identifier for the plugin |
| `description` | str | Yes | One-line description |
| `version` | str | No | Semantic version (default: `"0.1.0"`) |
| `permissions` | `PluginPermissions` | Yes | Full permission manifest |
| `enabled` | bool | No | Whether the plugin is active (default: `False`; set to `True` by the loader after successful init) |

### PluginPermissions

Plugins must declare every resource they require.  This manifest is inspected
during security review.

| Permission | Type | Default | Description |
|---|---|---|---|
| `network` | bool | `False` | Plugin may make outbound network requests |
| `filesystem_read` | bool | `False` | Plugin may read from the filesystem |
| `filesystem_write` | bool | `False` | Plugin may write to the filesystem |
| `shell` | bool | `False` | Plugin may execute shell commands |
| `allowed_hosts` | list[str] | `[]` | Explicit hostnames the plugin may contact (meaningful when `network=True`) |
| `allowed_paths` | list[str] | `[]` | Filesystem paths the plugin may access (meaningful when `filesystem_read` or `filesystem_write` is `True`) |

### Plugin Manifest

Every plugin exposes a `get_manifest()` method that returns a serialisable
dictionary:

```json
{
  "name": "weather",
  "version": "1.0.0",
  "description": "Fetches weather data from an external API.",
  "permissions": {
    "network": true,
    "filesystem_read": false,
    "filesystem_write": false,
    "shell": false,
    "allowed_hosts": ["api.openweathermap.org"],
    "allowed_paths": []
  },
  "enabled": true
}
```

This manifest is used during security review and is recorded in audit events
when the plugin is loaded.

### Plugin Loader

The `PluginLoader` (`missy/plugins/loader.py`) is the gatekeeper for all
plugin activity.

**Loading a plugin** (`load_plugin(plugin)`):

1. Check `config.plugins.enabled`.  If `False`, raise `PolicyViolationError`.
2. Check `plugin.name in config.plugins.allowed_plugins`.  If not present,
   raise `PolicyViolationError`.
3. Call `plugin.initialize()`.  If it returns `False` or raises, the plugin
   is not loaded (returns `False`).
4. Set `plugin.enabled = True` and register the plugin.
5. Emit a `plugin.load` audit event with the full manifest.

**Executing a plugin** (`execute(name, **kwargs)`):

1. Look up the plugin by name.  If not loaded, raise `PolicyViolationError`.
2. Check `plugin.enabled`.  If `False`, raise `PolicyViolationError`.
3. Emit a `plugin.execute.start` audit event.
4. Call `plugin.execute(**kwargs)`.
5. Emit a `plugin.execute` audit event with result `"allow"` or `"error"`.

### How to Install a Plugin

1. Implement a class extending `BasePlugin` with a complete `permissions`
   manifest.
2. Add the plugin's `name` to `plugins.allowed_plugins` in your config:

```yaml
plugins:
  enabled: true
  allowed_plugins:
    - "weather"
```

3. Load the plugin programmatically:

```python
from missy.plugins.loader import init_plugin_loader, get_plugin_loader
from my_plugins import WeatherPlugin

loader = get_plugin_loader()
loader.load_plugin(WeatherPlugin())
```

### How to Disable All Third-Party Plugins

Set `plugins.enabled: false` in your configuration (this is the default):

```yaml
plugins:
  enabled: false
  allowed_plugins: []
```

When `enabled` is `false`, any attempt to load a plugin raises
`PolicyViolationError` regardless of the `allowed_plugins` list.

### Listing Plugins via CLI

```bash
missy plugins
```

Shows the plugin system status (enabled/disabled), the allowlist, and a table
of loaded plugins with their name, version, description, and enabled status.

---

## Permission Model

Both skills and plugins declare permissions through their respective
`SkillPermissions` and `PluginPermissions` dataclasses.  These permissions
map to policy checks in the policy engine:

| Permission | Policy Check | Policy Engine Method |
|---|---|---|
| `network` | Network allowlist | `PolicyEngine.check_network(host)` |
| `filesystem_read` | Read path allowlist | `PolicyEngine.check_read(path)` |
| `filesystem_write` | Write path allowlist | `PolicyEngine.check_write(path)` |
| `shell` | Shell command allowlist | `PolicyEngine.check_shell(command)` |

Permission declarations on skills and plugins serve as documentation and
security review metadata.  The actual enforcement happens at the policy engine
level when the skill or plugin attempts the operation at runtime.

---

## Audit Events

| Event Type | Category | When |
|---|---|---|
| `skill.execute` | `plugin` | Skill execution (success, error, or not-found) |
| `plugin.load` | `plugin` | Plugin load attempt (allow, deny, or error) |
| `plugin.execute.start` | `plugin` | Plugin execution begins |
| `plugin.execute` | `plugin` | Plugin execution completes (allow, deny, or error) |
