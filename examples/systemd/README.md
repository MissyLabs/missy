# Systemd Service for Missy Gateway

This directory contains a systemd template unit file for running the Missy
gateway as a background service.

---

## Prerequisites

- Missy must be installed in the target user's Python environment (e.g.
  via `pip install --user missy` or a virtual environment).
- The `missy` executable must be available at `/home/<USER>/.local/bin/missy`.
  If it is installed elsewhere, edit the `ExecStart` path in the service file.
- A valid `config.yaml` must exist at `~/.missy/config.yaml` (or wherever
  the gateway expects it).
- Any required API keys must be set as environment variables. You can use a
  systemd `EnvironmentFile` or an override to inject them:

  ```bash
  systemctl edit missy-gateway@youruser
  ```

  Then add:

  ```ini
  [Service]
  Environment=ANTHROPIC_API_KEY=sk-ant-...
  Environment=DISCORD_BOT_TOKEN=...
  ```

---

## Installation

Copy the service file to the systemd system directory:

```bash
sudo cp missy-gateway.service /etc/systemd/system/missy-gateway@.service
sudo systemctl daemon-reload
```

The `@` in the filename makes this a template unit. The part after `@` in
the instance name becomes the `%i` substitution variable, which is used
for `User`, `Group`, `WorkingDirectory`, and `ReadWritePaths`.

---

## Enable and Start

```bash
sudo systemctl enable --now missy-gateway@youruser
```

This enables the service to start on boot and starts it immediately.
Replace `youruser` with the actual Linux username.

---

## View Logs

```bash
journalctl -u missy-gateway@youruser -f
```

The `-f` flag follows the log output in real time. To see the last 100
lines:

```bash
journalctl -u missy-gateway@youruser -n 100
```

---

## Reload Configuration

If you change `config.yaml` or environment variables, restart the service:

```bash
sudo systemctl restart missy-gateway@youruser
```

Note: `systemctl reload` is not implemented for this service (there is
no `ExecReload` directive). Use `restart` instead.

---

## Stop and Disable

```bash
sudo systemctl stop missy-gateway@youruser
sudo systemctl disable missy-gateway@youruser
```

---

## Security Hardening Notes

The service file includes several systemd security directives:

| Directive | Effect |
|-----------|--------|
| `NoNewPrivileges=true` | Prevents the process from gaining new privileges (e.g. via setuid). |
| `PrivateTmp=true` | Gives the service its own `/tmp` namespace. |
| `ProtectSystem=strict` | Makes the entire filesystem read-only except for paths listed in `ReadWritePaths`. |
| `ProtectHome=read-only` | Makes `/home` read-only except for paths listed in `ReadWritePaths`. |
| `ReadWritePaths=...` | Grants write access to `~/.missy` (for audit logs, jobs, memory) and `~/workspace` (for agent output). |

If the Missy workspace is at a different path, update `ReadWritePaths`
accordingly.

---

## Troubleshooting

**Service fails to start with "permission denied":**
- Verify that the `missy` binary is executable: `ls -l /home/youruser/.local/bin/missy`
- Verify that `ReadWritePaths` includes all directories Missy needs to write to.

**API calls fail with "Network access denied":**
- Check that `config.yaml` includes the required domains in `allowed_domains` or `allowed_hosts`.
- Verify that API key environment variables are set in the service environment.

**Service starts but no Discord messages are processed:**
- Check that `discord.enabled: true` is set in `config.yaml`.
- Verify that `DISCORD_BOT_TOKEN` is set in the service environment.
- Check `journalctl` output for Gateway connection errors.
