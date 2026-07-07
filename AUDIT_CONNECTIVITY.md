# Connectivity Audit Notes

Last updated: 2026-07-07

## Discord Connectivity

- Discord REST still routes through `DiscordRestClient` and `PolicyHTTPClient`.
- Discord Gateway still uses the raw Gateway WebSocket client.
- Discord voice is optional and uses `discord.py` for voice transport after the channel lazy-starts `DiscordVoiceManager`.
- Agent-callable voice actions now require a live voice binding and declare Discord network permissions at the tool layer.

## Network/Install Activity This Session

- Ran `sudo -n apt-get update`.
- Installed `python3-opencv` through apt because full-suite vision tests require `cv2`.
- Installed user-site `opencv-python-headless 5.0.0.93` with `python3 -m pip install --user --break-system-packages` because user-site NumPy `2.4.3` is incompatible with the apt OpenCV extension.

## Remaining Connectivity Gaps

- Library-level Discord voice sockets are not individually mediated by `PolicyHTTPClient`; policy enforcement is currently at tool execution/startup boundaries.
- Discord diagnostics should report whether REST, Gateway, and voice transports are separately reachable and policy-allowed.
- Multi-account Discord connectivity needs account-scoped voice binding before concurrent voice sessions are safe.
