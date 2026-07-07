# TEST_RESULTS

- Timestamp: 2026-07-07 19:08:32 EDT

## Focused Discord Image/Security

Command:

```bash
pytest tests/channels/test_discord_image_analyze.py tests/channels/test_discord_image_gaps.py::TestSaveDiscordAttachmentPathTraversal tests/security/test_discord_attachment_codeevolution_filewrite_security.py::TestDiscordSaveAttachmentSanitization -q
```

Result:

```text
54 passed in 0.30s
```

## Broader Discord Media/Channel/Protocol

Command:

```bash
pytest tests/channels/test_discord_image_analyze.py tests/channels/test_discord_image_gaps.py tests/channels/discord/test_image_commands.py tests/channels/test_discord_channel_coverage.py tests/channels/test_discord_channel_gap_coverage.py tests/channels/test_discord_protocol_deep.py tests/security/test_discord_attachment_codeevolution_filewrite_security.py::TestDiscordSaveAttachmentSanitization -q
```

Result:

```text
353 passed in 8.16s
```

## Full Suite

Command:

```bash
pytest -q
```

Result:

```text
20266 passed, 13 skipped in 367.66s (0:06:07)
```

## Lint/Format

```text
ruff check .: passed
ruff format --check .: 708 files already formatted
```
