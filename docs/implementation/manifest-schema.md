# Plugin and Skill Manifest Schemas

Plugins and skills in Missy declare their identity and resource
requirements through structured manifests. These manifests are used at
load time for security validation and at runtime for audit logging.

**Source files:**

- `missy/plugins/base.py` -- `BasePlugin`, `PluginPermissions`
- `missy/plugins/loader.py` -- `PluginLoader` (manifest validation)
- `missy/skills/base.py` -- `BaseSkill`, `SkillPermissions`
- `missy/skills/registry.py` -- `SkillRegistry`

---

## Plugin Manifest

### Manifest Fields

The manifest is produced by `BasePlugin.get_manifest()` and contains:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | Yes | Unique registry key for the plugin. |
| `version` | `string` | Yes | Semantic version string. Defaults to `"0.1.0"`. |
| `description` | `string` | Yes | One-line description of what the plugin does. |
| `permissions` | `object` | Yes | Resource permissions declared by the plugin (see below). |
| `enabled` | `boolean` | Yes | Whether the plugin is currently active. Always `false` before `initialize()` succeeds. |

### Permission Fields (PluginPermissions)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `network` | `boolean` | `false` | Plugin may make outbound network requests. |
| `filesystem_read` | `boolean` | `false` | Plugin may read from the filesystem. |
| `filesystem_write` | `boolean` | `false` | Plugin may write to the filesystem. |
| `shell` | `boolean` | `false` | Plugin may execute shell commands. |
| `allowed_hosts` | `list[string]` | `[]` | Explicit hostnames the plugin is permitted to contact. Meaningful only when `network` is `true`. |
| `allowed_paths` | `list[string]` | `[]` | Filesystem paths the plugin is permitted to access. Meaningful only when `filesystem_read` or `filesystem_write` is `true`. |

### Extended Manifest Fields (for distribution)

The following fields are not part of the current `get_manifest()` output
but are recommended for published plugin packages:

| Field | Type | Description |
|-------|------|-------------|
| `author` | `string` | Plugin author or maintainer. |
| `network_requirements` | `list[string]` | All hostnames or domains the plugin needs to reach. |
| `filesystem_requirements` | `list[string]` | All paths the plugin needs to access. |
| `risk_level` | `string` | One of `low`, `medium`, `high` (see Risk Levels below). |

### Full Example Plugin Manifest (JSON)

```json
{
  "name": "weather",
  "version": "1.2.0",
  "description": "Fetches weather data from the OpenWeatherMap API.",
  "author": "Missy Contributors",
  "permissions": {
    "network": true,
    "filesystem_read": false,
    "filesystem_write": false,
    "shell": false,
    "allowed_hosts": ["api.openweathermap.org"],
    "allowed_paths": []
  },
  "network_requirements": ["api.openweathermap.org"],
  "filesystem_requirements": [],
  "risk_level": "low",
  "enabled": true
}
```

```json
{
  "name": "git-assistant",
  "version": "0.5.0",
  "description": "Runs git commands and summarises repository status.",
  "author": "Missy Contributors",
  "permissions": {
    "network": false,
    "filesystem_read": true,
    "filesystem_write": false,
    "shell": true,
    "allowed_hosts": [],
    "allowed_paths": ["/home/user/workspace"]
  },
  "network_requirements": [],
  "filesystem_requirements": ["/home/user/workspace"],
  "risk_level": "high",
  "enabled": false
}
```

---

## Skill Manifest

Skills are lighter-weight than plugins. They do not have a `get_manifest()`
method, but their identity and permissions are declared through class
attributes.

### Manifest Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | Yes | Unique registry key for the skill. |
| `version` | `string` | Yes | Semantic version string. Defaults to `"0.1.0"`. |
| `description` | `string` | Yes | One-line description. |
| `permissions` | `SkillPermissions` | Yes | Resource permissions (see below). |

### Permission Fields (SkillPermissions)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `network` | `boolean` | `false` | Skill may make outbound network requests. |
| `filesystem_read` | `boolean` | `false` | Skill may read from the filesystem. |
| `filesystem_write` | `boolean` | `false` | Skill may write to the filesystem. |
| `shell` | `boolean` | `false` | Skill may execute shell commands. |

Note: Unlike plugins, skills do not declare `allowed_hosts` or
`allowed_paths` in their permissions. These are currently validated at the
tool level (via `ToolPermissions`) rather than the skill level.

### Full Example Skill Manifest (JSON)

```json
{
  "name": "system_info",
  "version": "0.1.0",
  "description": "Reports system platform, CPU, and memory information.",
  "permissions": {
    "network": false,
    "filesystem_read": true,
    "filesystem_write": false,
    "shell": false
  }
}
```

```json
{
  "name": "workspace_list",
  "version": "0.1.0",
  "description": "Lists files in the configured workspace directory.",
  "permissions": {
    "network": false,
    "filesystem_read": true,
    "filesystem_write": false,
    "shell": false
  }
}
```

---

## Permission Vocabulary

The permission system uses these atomic capabilities:

| Permission | Scope | Description |
|------------|-------|-------------|
| `filesystem:read` | Plugin / Skill / Tool | May read files from declared paths. |
| `filesystem:write` | Plugin / Skill / Tool | May write files to declared paths. |
| `network:outbound` | Plugin / Skill / Tool | May make outbound HTTP requests to declared hosts. |
| `shell:exec` | Plugin / Skill / Tool | May execute shell commands. |
| `plugin:load` | Plugin | May be loaded into the runtime (requires `PluginPolicy.enabled` and name in `allowed_plugins`). |

These map to the boolean flags in `PluginPermissions`, `SkillPermissions`,
and `ToolPermissions`:

| Boolean field | Permission |
|--------------|------------|
| `network` | `network:outbound` |
| `filesystem_read` | `filesystem:read` |
| `filesystem_write` | `filesystem:write` |
| `shell` | `shell:exec` |

---

## Risk Levels

Risk levels are advisory labels for published plugin packages. They are
not enforced by the runtime but guide operators in security review:

| Level | Criteria | Examples |
|-------|----------|---------|
| `low` | No network, no filesystem, no shell. Pure computation or formatting. | Calculator, text formatting |
| `medium` | Network access (read-only) OR filesystem read access. No shell. | Weather API, file reader |
| `high` | Any of: filesystem write, shell execution, or network + filesystem combined. | Git assistant, deployment tools |

### Determination Rules

1. If `shell == true` -> `high`
2. If `filesystem_write == true` -> `high`
3. If `network == true` AND (`filesystem_read == true` OR `filesystem_write == true`) -> `high`
4. If `network == true` OR `filesystem_read == true` -> `medium`
5. Otherwise -> `low`

---

## Version Compatibility Rules

Missy does not currently enforce version constraints on plugins or skills.
The following conventions are recommended:

1. **Semantic versioning** (MAJOR.MINOR.PATCH) for all plugins and skills.
2. **MAJOR version bump** when the `execute()` signature changes or
   permissions are added.
3. **MINOR version bump** for new features that do not change the API.
4. **PATCH version bump** for bug fixes.
5. When Missy introduces breaking changes to `BasePlugin` or `BaseSkill`,
   the framework version should be bumped and plugins should declare a
   minimum compatible framework version in their package metadata.

---

## How the Loader Validates Manifests Before Execution

`PluginLoader.load_plugin(plugin)` performs the following checks in order:

1. **Global enable check** -- `config.plugins.enabled` must be `True`.
   If `False`, raises `PolicyViolationError` with reason
   `"plugins_disabled"`.

2. **Allow-list check** -- `plugin.name` must appear in
   `config.plugins.allowed_plugins`. If absent, raises
   `PolicyViolationError` with reason `"not_in_allowed_list"`.

3. **Initialisation** -- Calls `plugin.initialize()`. If it returns
   `False` or raises an exception, the plugin is not loaded (but no
   `PolicyViolationError` is raised -- it is logged as an error).

4. **Enable and register** -- On success, `plugin.enabled` is set to
   `True` and the plugin is stored in the loader's internal registry.
   A `plugin.load` audit event is emitted with the full manifest.

At execution time, `PluginLoader.execute(name)` checks:

1. The plugin must be loaded (present in the registry).
2. The plugin must be enabled (`plugin.enabled == True`).

If either check fails, `PolicyViolationError` is raised with a `deny`
audit event.

Skills do not go through this validation pipeline. They are registered
directly into `SkillRegistry` and executed without policy checks (skills
are trusted, in-process code). Audit events are still emitted for every
skill execution.

Tools go through `ToolRegistry.execute()`, which validates the tool's
`ToolPermissions` against the active `PolicyEngine` before calling
`tool.execute()`. This checks network, filesystem read, filesystem write,
and shell permissions using the policy engine's check methods.
