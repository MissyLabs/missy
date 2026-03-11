# Policy Engine

The policy engine is Missy's enforcement layer. Every action that touches
the network, filesystem, or shell is checked against a declarative policy
before it is allowed to proceed. The engine follows a secure-by-default
posture: everything is denied unless explicitly allowed.

**Source files:**

- `missy/policy/engine.py` -- facade and singleton
- `missy/policy/network.py` -- network sub-engine
- `missy/policy/filesystem.py` -- filesystem sub-engine
- `missy/policy/shell.py` -- shell sub-engine
- `missy/config/settings.py` -- policy dataclasses
- `missy/core/exceptions.py` -- `PolicyViolationError`

---

## Architecture

`PolicyEngine` is a facade that composes three domain-specific sub-engines.
It delegates each check to the appropriate sub-engine without adding logic
of its own.

```
                      PolicyEngine (facade)
                     /        |        \
          NetworkPolicy  FilesystemPolicy  ShellPolicy
               |              |               |
        NetworkPolicyEngine  FilesystemPolicyEngine  ShellPolicyEngine
```

### Initialisation

A module-level singleton is managed by two functions:

```python
from missy.policy.engine import init_policy_engine, get_policy_engine

engine = init_policy_engine(config)   # Create and install
engine = get_policy_engine()          # Retrieve (thread-safe)
```

`init_policy_engine(config)` constructs a new `PolicyEngine` from a
`MissyConfig` and atomically installs it under a `threading.Lock`. Calling
it a second time replaces the existing engine. `get_policy_engine()` raises
`RuntimeError` if the engine has not been initialised.

### Facade Methods

| Method | Delegates to | Signature |
|--------|-------------|-----------|
| `check_network(host, session_id, task_id)` | `NetworkPolicyEngine.check_host()` | `-> bool` |
| `check_read(path, session_id, task_id)` | `FilesystemPolicyEngine.check_read()` | `-> bool` |
| `check_write(path, session_id, task_id)` | `FilesystemPolicyEngine.check_write()` | `-> bool` |
| `check_shell(command, session_id, task_id)` | `ShellPolicyEngine.check_command()` | `-> bool` |

All methods return `True` on allow and raise `PolicyViolationError` on
deny.

---

## NetworkPolicyEngine

**Config type:** `NetworkPolicy`

```python
@dataclass
class NetworkPolicy:
    default_deny: bool = True
    allowed_cidrs: list[str] = field(default_factory=list)
    allowed_domains: list[str] = field(default_factory=list)
    allowed_hosts: list[str] = field(default_factory=list)
    provider_allowed_hosts: list[str] = field(default_factory=list)
    tool_allowed_hosts: list[str] = field(default_factory=list)
    discord_allowed_hosts: list[str] = field(default_factory=list)
```

### Check Algorithm: `check_host(host)`

The host is normalised to lowercase with IPv6 brackets stripped.

1. **Default allow mode** -- If `default_deny` is `False`, allow
   everything immediately. Event emitted with rule `"default_allow"`.

2. **Bare IP address** -- If the host parses as an IPv4 or IPv6 address,
   check it against `allowed_cidrs` only. No DNS is performed for bare IPs.
   If no CIDR matches, deny immediately.

3. **Exact host match** -- Compare the host against each entry in
   `allowed_hosts`. The configured entry may include a port suffix
   (`"api.github.com:443"`); the port is stripped before comparison.
   Case-insensitive.

4. **Domain suffix match** -- Compare the host against `allowed_domains`.
   - `"github.com"` matches only `"github.com"` exactly.
   - `"*.github.com"` matches `"api.github.com"`, `"github.com"`, and any
     subdomain ending in `.github.com`.

5. **DNS resolution + CIDR re-check** -- Resolve the hostname using
   `socket.getaddrinfo()` and check each resulting IP address against the
   CIDR allow-lists. DNS failures fall through to deny.

6. **Deny** -- If none of the above matched, raise `PolicyViolationError`
   with `category="network"`.

### CIDR Check Details

CIDR strings are parsed once at construction time into
`ipaddress.IPv4Network` / `ipaddress.IPv6Network` objects. Invalid CIDRs
are logged and skipped. At check time, the IP address is tested with the
`in` operator against each pre-parsed network. Mixed IPv4/IPv6 comparisons
are silently skipped (no error).

### Audit Events

Every `check_host()` call emits a `network_check` event:

```json
{
  "event_type": "network_check",
  "category": "network",
  "result": "allow" | "deny",
  "policy_rule": "cidr:10.0.0.0/8" | "host:api.github.com" | "domain:*.github.com" | "default_allow" | null,
  "detail": {"host": "api.github.com"}
}
```

---

## FilesystemPolicyEngine

**Config type:** `FilesystemPolicy`

```python
@dataclass
class FilesystemPolicy:
    allowed_write_paths: list[str] = field(default_factory=list)
    allowed_read_paths: list[str] = field(default_factory=list)
```

### Check Algorithm: `check_read(path)` / `check_write(path)`

1. **Path resolution** -- The input path (string or `Path`) is resolved to
   an absolute path using `Path.resolve(strict=False)`. This resolves
   existing symlinks, preventing symlink-traversal attacks where an attacker
   plants a symlink inside an allowed directory that points outside it.
   `strict=False` means paths to not-yet-existing files are still resolved.

2. **Containment check** -- The resolved path is compared against each entry
   in the relevant allow-list (`allowed_read_paths` for reads,
   `allowed_write_paths` for writes). The configured entries are also
   resolved. A path matches if it equals the allowed directory or is nested
   inside it (checked with `Path.is_relative_to()`). Trailing slashes are
   handled correctly.

3. **Allow or deny** -- If a match is found, return `True`. Otherwise raise
   `PolicyViolationError` with `category="filesystem"`.

### Audit Events

Every check emits a `filesystem_read` or `filesystem_write` event:

```json
{
  "event_type": "filesystem_read",
  "category": "filesystem",
  "result": "allow" | "deny",
  "policy_rule": "/home/user/workspace" | null,
  "detail": {"path": "/home/user/workspace/notes.txt", "operation": "read"}
}
```

---

## ShellPolicyEngine

**Config type:** `ShellPolicy`

```python
@dataclass
class ShellPolicy:
    enabled: bool = False
    allowed_commands: list[str] = field(default_factory=list)
```

### Check Algorithm: `check_command(command)`

1. **Global disable** -- If `policy.enabled` is `False`, deny
   unconditionally. The rule is recorded as `"shell_disabled"`.

2. **Program extraction** -- The first token is extracted from the command
   string using `shlex.split()` (POSIX shell tokenisation). Empty, whitespace-
   only, or unparseable commands (malformed quoting) are denied.

3. **Allow-list check** -- The extracted program name is compared against
   `allowed_commands` by basename. Specifically:
   - The basename of the program token (e.g. `git` from `/usr/bin/git`) is
     compared against the basename of each `allowed_commands` entry.
   - Exact basename equality is required. `"git"` does **not** match
     `"gitk"` -- this is intentional to avoid over-permissioning.

4. **Allow or deny** -- If a match is found, return `True`. Otherwise raise
   `PolicyViolationError` with `category="shell"`.

### Audit Events

Every check emits a `shell_check` event:

```json
{
  "event_type": "shell_check",
  "category": "shell",
  "result": "allow" | "deny",
  "policy_rule": "cmd:git" | "shell_disabled" | null,
  "detail": {"command": "git status"}
}
```

---

## PolicyViolationError

Defined in `missy/core/exceptions.py`:

```python
class PolicyViolationError(MissyError):
    def __init__(self, message: str, *, category: str, detail: str) -> None:
        ...
```

| Attribute | Description |
|-----------|-------------|
| `message` | Human-readable error message (inherited from `Exception`). |
| `category` | The policy domain: `"network"`, `"filesystem"`, `"shell"`, or `"plugin"`. |
| `detail` | Extended explanation of why the action was denied. |

Every sub-engine raises `PolicyViolationError` on deny. The exception is
the canonical signal for "this action is not permitted" -- callers can
catch it to take corrective action (e.g. return an error to the user)
rather than letting it propagate as a crash.

---

## How Other Modules Use the Policy Engine

| Module | Check method used |
|--------|-------------------|
| `missy.gateway.client.PolicyHTTPClient._check_url()` | `check_network(host)` |
| `missy.tools.registry.ToolRegistry._check_permissions()` | `check_network()`, `check_read()`, `check_write()`, `check_shell()` |
| `missy.plugins.loader.PluginLoader` | Uses `PluginPolicy` directly (not via `PolicyEngine`) |

---

## How to Add a New Policy Type

To add a fourth policy domain (e.g. "database"):

1. **Define the policy dataclass** in `missy/config/settings.py`:

   ```python
   @dataclass
   class DatabasePolicy:
       enabled: bool = False
       allowed_connections: list[str] = field(default_factory=list)
   ```

2. **Add the field to `MissyConfig`:**

   ```python
   @dataclass
   class MissyConfig:
       ...
       database: DatabasePolicy
   ```

3. **Create the sub-engine** at `missy/policy/database.py` following the
   pattern of `NetworkPolicyEngine`:
   - Accept the policy dataclass in `__init__`
   - Implement a public check method (e.g. `check_connection()`)
   - Emit audit events via `AuditEvent.now()` and `event_bus.publish()`
   - Raise `PolicyViolationError` on deny

4. **Wire it into `PolicyEngine`:**

   ```python
   class PolicyEngine:
       def __init__(self, config: MissyConfig) -> None:
           ...
           self.database = DatabasePolicyEngine(config.database)

       def check_database(self, connection: str, ...) -> bool:
           return self.database.check_connection(connection, ...)
   ```

5. **Add the parser** in `missy/config/settings.py` and call it from
   `load_config()`.

6. **Update the `EventCategory` type** in `missy/core/events.py` to include
   `"database"`.
