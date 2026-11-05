# Offline native operator administration

`missy-rs --config /absolute/private/config.yaml admin` offers `security-scan`, `sessions`, `mcp-plan` and `mcp-apply`. Specify tenant, user, channel and session using the admin scope arguments. These commands do not contact providers, discover MCP servers, launch commands or reload processes.

Security scan reports file metadata, capability flags and whether configured MCP executables exist on PATH. It does not read credentials or prove executable integrity, check external network reachability, inspect parent path ancestry, or establish that a service is safe. Run it under the intended service identity.

Sessions lists distinct sessions present in turns or session metadata within the exact tenant/user/channel, optionally filtered by `--prefix`, bounded by `--limit` (1–1000). It does not consult memory, and a session represented only by memory is not listed. State database opening may run native schema maintenance.

MCP instructions are private (mode 0600), owned, single-link JSON files. Example, with absolute paths:

```json
{"operation":"add","entry":{"name":"example","command":["/usr/bin/example"],"env":{},"manifest_sha256":"","timeout_seconds":30},"manifest":"/private/tools.json"}
```

Pin uses `{"operation":"pin","name":"example","manifest":"/private/tools.json"}`; remove uses `{"operation":"remove","name":"example"}`. The manifest is a private JSON tools/list result containing a `tools` array. The pin is the runtime's canonical digest of the whole result, not the raw file digest. No authority is added: add/pin requires existing MCP enablement and an exact argv entry in `policy.allowed_commands`. No server is contacted to verify the manifest. Runtime MCP explicitly clears the inherited environment and passes only configured entries.

`mcp-plan /private/instruction.json` prints the exact config SHA256 and an instruction binding SHA256 incorporating both instruction and raw manifest bytes. Apply using `mcp-apply /private/instruction.json --config-sha256 HASH --instruction-sha256 HASH --snapshot /private/new-snapshot.json --confirm`. The snapshot is private and contains the original complete configuration, including any secrets; protect it. It is compatible with `config rollback`. Applying requires an owned, non-group/world-writable parent, refuses symlink and hardlink files, takes the configuration operation lock, validates again, snapshots first and atomically replaces the file. It does not notify or restart running processes. An independent editor that ignores the lock can still race the final rename; coordinate all writers.
