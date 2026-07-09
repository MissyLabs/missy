# AUDIT_SECURITY

- Timestamp: 2026-07-09 11:19:42

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and tool-intelligence scan
```
/home/missy/missy/TEST_EDGE_CASES.md:7:- generated tool candidates remain disabled until approved
/home/missy/missy/TEST_EDGE_CASES.md:8:- generated tool permissions are least-privilege
/home/missy/missy/TEST_EDGE_CASES.md:9:- tool metadata includes provenance, schema, tests, versions, and lifecycle state
/home/missy/missy/TEST_EDGE_CASES.md:10:- benchmark runs compare OpenAI, Ollama, Anthropic, and mock/local providers where available
/home/missy/missy/TEST_EDGE_CASES.md:11:- provider-specific enablement respects benchmark thresholds
/home/missy/missy/TEST_EDGE_CASES.md:12:- unsafe or flaky tools are flagged or disabled per provider
/home/missy/missy/TEST_EDGE_CASES.md:13:- tool schema incompatibilities are detected and reported
/home/missy/missy/TEST_EDGE_CASES.md:14:- benchmark failures do not enable tools accidentally
/home/missy/missy/TEST_EDGE_CASES.md:15:- future overhaul compatibility for Discord, scheduling, and provider routing
/home/missy/missy/LAST_SESSION_SUMMARY.md:8:  (`_SUMMARY_TOPIC`) that captures `resolved_provider`/`tools_used`/`cost`
/home/missy/missy/LAST_SESSION_SUMMARY.md:13:- `missy/api/operator_controls.py`: added `scheduler.remove_job`, a third
/home/missy/missy/LAST_SESSION_SUMMARY.md:14:  confirmation-gated scheduler control (destructive-flagged) alongside
/home/missy/missy/LAST_SESSION_SUMMARY.md:16:- `missy/api/server.py`: new routes `GET/POST /api/v1/scheduler/jobs`,
/home/missy/missy/LAST_SESSION_SUMMARY.md:17:  `DELETE /api/v1/scheduler/jobs/{id}` (delegates to `scheduler.remove_job`),
/home/missy/missy/LAST_SESSION_SUMMARY.md:22:  schema migration); `cleanup()` now exempts pinned turns via
/home/missy/missy/LAST_SESSION_SUMMARY.md:29:  console's completion handler now shows a provider/tools/cost summary
/home/missy/missy/LAST_SESSION_SUMMARY.md:37:  changes (the security/connectivity audits are now hand-written summaries
/home/missy/missy/LAST_SESSION_SUMMARY.md:74:login → CSRF → scheduler job create/list/remove (with/without confirmation)
/home/missy/missy/LAST_SESSION_SUMMARY.md:76:separately verified the run cost/provider/tools_used enrichment through both
/home/missy/missy/LAST_SESSION_SUMMARY.md:87:- Safe controls cover providers, scheduler (pause/resume/remove), and now
/home/missy/missy/LAST_SESSION_SUMMARY.md:88:  memory turns, but not tools/skills/plugins/Discord/voice/vision/webhooks/
/home/missy/missy/LAST_SESSION_SUMMARY.md:99:with tool/skill enable-disable controls to keep closing the "full
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:1:"""Coverage gap tests for missy/tools/builtin/atspi_tools.py.
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:77:    def test_get_desktop_calls_registry(self):
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:79:        from missy.tools.builtin import atspi_tools
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:87:            result = atspi_tools._get_desktop()
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:101:        from missy.tools.builtin import atspi_tools
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:113:            result = atspi_tools._get_focused_application(desktop)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:119:        from missy.tools.builtin import atspi_tools
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:130:            result = atspi_tools._get_focused_application(desktop)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:136:        from missy.tools.builtin import atspi_tools
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:143:            result = atspi_tools._get_focused_application(desktop)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:156:        from missy.tools.builtin import atspi_tools
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:173:            result = atspi_tools._walk_tree(node, max_depth=1)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:180:        from missy.tools.builtin.atspi_tools import _walk_tree
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:191:        from missy.tools.builtin.atspi_tools import _walk_tree
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:197:        from missy.tools.builtin.atspi_tools import _walk_tree
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:211:        from missy.tools.builtin.atspi_tools import _find_element
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:231:        from missy.tools.builtin.atspi_tools import _find_element
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:239:        from missy.tools.builtin.atspi_tools import _find_element
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:258:        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:268:        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:274:                "missy.tools.builtin.atspi_tools._get_desktop",
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:285:        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:293:            patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:302:        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:310:            patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:319:        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:327:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:329:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:331:                "missy.tools.builtin.atspi_tools._walk_tree",
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:342:        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:350:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:352:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:354:                "missy.tools.builtin.atspi_tools._walk_tree",
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:371:        from missy.tools.builtin.atspi_tools import AtSpiClickTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:379:        from missy.tools.builtin.atspi_tools import AtSpiClickTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:388:        from missy.tools.builtin.atspi_tools import AtSpiClickTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:396:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:398:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:399:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=None),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:407:        from missy.tools.builtin.atspi_tools import AtSpiClickTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:418:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:420:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:421:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=element),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:429:        from missy.tools.builtin.atspi_tools import AtSpiClickTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:443:            patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:444:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=element),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:451:        from missy.tools.builtin.atspi_tools import AtSpiClickTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:459:            patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:467:        from missy.tools.builtin.atspi_tools import AtSpiClickTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:477:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:479:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:480:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=element),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:495:        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:503:        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:512:        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:517:            patch("missy.tools.builtin.atspi_tools._get_desktop", side_effect=Exception("no dbus")),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:526:        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:534:            patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:543:        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:551:            patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:560:        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:568:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:570:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:571:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=None),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:580:        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:588:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:590:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:592:                "missy.tools.builtin.atspi_tools._find_element",
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:603:        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:612:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:614:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:615:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=element),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:624:        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:634:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:636:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:637:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=element),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:652:        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:660:        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:666:                "missy.tools.builtin.atspi_tools._get_desktop", side_effect=Exception("dbus gone")
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:676:        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:684:            patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:693:        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:701:            patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:710:        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:718:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:720:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:721:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=None),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:730:        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:738:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:740:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:742:                "missy.tools.builtin.atspi_tools._find_element",
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:753:        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:776:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:778:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:779:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=element),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:791:        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:807:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:809:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:810:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=element),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:820:        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:834:                "missy.tools.builtin.atspi_tools._get_desktop", return_value=MagicMock(childCount=0)
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:836:            patch("missy.tools.builtin.atspi_tools._get_focused_application", return_value=app),
/home/missy/missy/tests/tools/test_atspi_tools_coverage.py:837:            patch("missy.tools.builtin.atspi_tools._find_element", return_value=element),
/home/missy/missy/README.md:5:Missy is a production-grade agentic platform that runs entirely on your hardware. Default-deny network, filesystem sandboxing, shell whitelisting, encrypted vault, and structured audit logging — every capability is locked down until you explicitly allow it. Connect any AI provider. Deploy voice nodes throughout your home. Automate with scheduled jobs. Extend with tools, skills, and plugins.
/home/missy/missy/README.md:13:Most AI assistants trust the network, trust the model, and trust the plugins. Missy trusts nothing by default.
/home/missy/missy/README.md:18:- **No plugins** unless you approve them individually
/home/missy/missy/README.md:19:- **Every action** logged as structured JSONL with full audit trail
/home/missy/missy/README.md:20:- **Every audit event** signed with the agent's Ed25519 identity
/home/missy/missy/README.md:29:- **Multi-provider** — Anthropic (Claude), OpenAI (GPT), Ollama (local models) with automatic fallback and runtime hot-swap (`missy providers switch`)
/home/missy/missy/README.md:30:- **API key rotation** — multiple keys per provider, round-robin distribution
/home/missy/missy/README.md:32:- **Agentic runtime** — tool-augmented loops with done-criteria verification, learnings extraction, and self-tuning prompt patches
/home/missy/missy/README.md:33:- **AI Playbook** — auto-captures successful tool patterns, injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals
/home/missy/missy/README.md:34:- **Attention system** — 5 brain-inspired subsystems (alerting, orienting, sustained, selective, executive) that track urgency, extract topics, maintain focus, and prioritize tools
/home/missy/missy/README.md:39:- **Interactive approval TUI** — real-time Rich terminal prompt for policy-denied operations (allow once / deny / allow always)
/home/missy/missy/README.md:40:- **Circuit breaker** — automatic backoff on provider failures (threshold=5, exponential to 300s)
/home/missy/missy/README.md:45:- **Failure tracking** — per-tool consecutive failure counts with automatic strategy rotation
/home/missy/missy/README.md:48:- **Code evolution** — self-evolving code modification engine with approval workflow and git-backed rollback
/home/missy/missy/README.md:49:- **Structured output** — Pydantic schema enforcement on LLM responses with automatic retry
/home/missy/missy/README.md:56:- **Multi-layer policy engine** — network (CIDR + domain + host), filesystem (per-path R/W), shell (command whitelist), L7 REST (HTTP method + path per host)
/home/missy/missy/README.md:57:- **Network presets** — `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
/home/missy/missy/README.md:58:- **Gateway enforcement** — all HTTP flows through `PolicyHTTPClient` with DNS rebinding protection, redirect blocking, scheme restrictions, interactive approval
/home/missy/missy/README.md:60:- **Prompt drift detection** — SHA-256 hashes system prompts, detects tampering between tool loop iterations
/home/missy/missy/README.md:63:- **Agent identity** — Ed25519 keypair at `~/.missy/identity.pem`, signs audit events, JWK export
/home/missy/missy/README.md:64:- **Trust scoring** — 0-1000 reliability tracking per tool/provider/MCP server with threshold warnings
/home/missy/missy/README.md:65:- **Container sandbox** — optional Docker-based isolation for tool execution (`--network=none`, memory/CPU limits)
/home/missy/missy/README.md:66:- **Landlock LSM** — Linux kernel-level filesystem enforcement via Landlock syscalls, complementing userspace policy
/home/missy/missy/README.md:67:- **Security scanner** — `missy security scan` audits installation for permission issues, config hygiene, exposed secrets
/home/missy/missy/README.md:68:- **MCP digest pinning** — SHA-256 verification of tool manifests; mismatches refuse to load
/home/missy/missy/README.md:72:- **CLI** — interactive REPL and single-shot queries with Rich formatting, capability modes (full/safe-chat/no-tools)
/home/missy/missy/README.md:73:- **Discord** — full Gateway WebSocket API, slash commands (`/ask`, `/status`, `/model`), DM allowlist, guild/role policies, image analysis
/home/missy/missy/README.md:75:- **Voice** — WebSocket server for edge nodes, faster-whisper STT, Piper TTS, device registry with PBKDF2 auth
/home/missy/missy/README.md:80:- **MCP servers** — connect external tool servers via `~/.missy/mcp.json`, auto-restart, digest pinning
/home/missy/missy/README.md:81:- **SKILL.md discovery** — scan directories for cross-agent portable skill definitions (`missy skills scan`)
/home/missy/missy/README.md:82:- **Tools, skills, plugins** — three extension tiers with increasing isolation and permission requirements
/home/missy/missy/README.md:85:- **Persona system** — YAML-backed agent identity/tone/style with backup, rollback, and audit logging
/home/missy/missy/README.md:93:- **Multi-provider** — Anthropic/OpenAI/Ollama image message formatting
/home/missy/missy/README.md:95:- **CLI tools** — `missy vision capture|inspect|review|doctor|health|benchmark|validate|memory`
/home/missy/missy/README.md:98:- **Browser tools** — Playwright-based Firefox automation (`pip install -e ".[desktop]"`)
/home/missy/missy/README.md:99:- **X11 tools** — window management and application launching
/home/missy/missy/README.md:100:- **Accessibility** — AT-SPI toolkit integration for GUI interaction
/home/missy/missy/README.md:103:- **Config presets** — `presets: ["anthropic", "github"]` replaces manual host lists
/home/missy/missy/README.md:106:- **Non-interactive setup** — `missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt`
/home/missy/missy/README.md:109:- **Audit logger** — every policy decision, provider call, and tool execution as JSONL, signed by agent identity
/home/missy/missy/README.md:110:- **Application logs** — rotating Python/provider diagnostics at `~/.missy/missy.log` (`missy logs tail`)
/home/missy/missy/README.md:130:The setup wizard walks you through configuring API keys, providers, network policy, and workspace paths. Once complete:
/home/missy/missy/README.md:152:missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt
/home/missy/missy/README.md:165:pip install -e ".[dev]"           # pytest, ruff, mypy, hypothesis, coverage tools
/home/missy/missy/README.md:182:(network,     (Anthropic, OpenAI,        (built-in tools,
/home/missy/missy/README.md:183: filesystem,   Ollama + fallback)         skills, plugins,
/home/missy/missy/README.md:199: Network ──► AuditLogger (signed) ──► ~/.missy/audit.jsonl
/home/missy/missy/README.md:205:Every outbound request — from providers, tools, plugins, MCP servers, Discord — passes through `PolicyHTTPClient`. No exceptions.
/home/missy/missy/README.md:217:  default_deny: true
/home/missy/missy/README.md:219:    - anthropic                  # auto-expands to api.anthropic.com + anthropic.com
/home/missy/missy/README.md:234:  enabled: false
/home/missy/missy/README.md:237:providers:
/home/missy/missy/README.md:238:  anthropic:
/home/missy/missy/README.md:239:    name: anthropic
/home/missy/missy/README.md:246:  enabled: false
/home/missy/missy/README.md:266:missy setup --no-prompt             # Non-interactive (--provider, --api-key-env, --model)
/home/missy/missy/README.md:267:missy ask PROMPT                    # Single-turn query (--provider, --session, --mode)
/home/missy/missy/README.md:268:missy run                           # Interactive REPL (--provider, --mode)
/home/missy/missy/README.md:269:missy providers list                # List providers and availability
/home/missy/missy/README.md:270:missy providers switch NAME         # Hot-swap active provider
/home/missy/missy/README.md:277:# Security & audit
/home/missy/missy/README.md:278:missy audit recent                  # Recent events (--limit, --category)
/home/missy/missy/README.md:279:missy audit security                # Policy violations
/home/missy/missy/README.md:289:missy discord status | probe | register-commands | audit
/home/missy/missy/README.md:293:missy devices list | pair | unpair | status | policy
/home/missy/missy/README.md:295:# MCP & skills
/home/missy/missy/README.md:297:missy skills                        # List registered skills
/home/missy/missy/README.md:298:missy skills scan                   # Discover SKILL.md files
/home/missy/missy/README.md:302:missy vision health | benchmark | validate | memory
/home/missy/missy/README.md:344:missy devices policy ID --mode full|safe-chat|muted
/home/missy/missy/README.md:354:python3 -m pytest tests/ -k "test_policy" -v         # Filter by name
/home/missy/missy/README.md:371:| [Configuration](https://missylabs.github.io/configuration/) | 7 | Full YAML reference, network/fs/shell policy, presets, providers |
/home/missy/missy/README.md:376:| [Providers](https://missylabs.github.io/providers/) | 5 | Anthropic, OpenAI, Ollama, runtime switching |
/home/missy/missy/README.md:377:| [Extending](https://missylabs.github.io/extending/) | 4 | Tools, plugins, MCP servers, SKILL.md |
/home/missy/missy/README.md:384:Developer-facing references in [`docs/`](docs/) — architecture, implementation deep-dives, persistence schema, module map.
/home/missy/missy/README.md:393:│                    attention, progress, approval, persona, behavior, hatching,
/home/missy/missy/README.md:404:├── policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy/README.md:405:├── providers/       Anthropic, OpenAI, Ollama + registry with fallback & hot-swap
/home/missy/missy/README.md:406:├── scheduler/       APScheduler integration, human schedule parser
/home/missy/missy/README.md:409:├── skills/          Skill registry + SKILL.md discovery
/home/missy/missy/README.md:410:├── plugins/         Security-gated external plugin loader
/home/missy/missy/README.md:411:├── tools/           Built-in tools + registry (18+ tools)
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:1:"""Coverage-gap tests for missy/tools/builtin/__init__.py.
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:4:  194-196: register_builtin_tools(registry=None) → fetches process-level registry
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:5:            via get_tool_registry() when no explicit registry is provided.
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:14:from missy.tools.builtin import _ALL_TOOL_CLASSES, register_builtin_tools
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:18:    """register_builtin_tools(registry=explicit) path — already covered; sanity checks."""
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:20:    def test_registers_all_tool_classes_into_provided_registry(self):
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:22:        mock_registry = MagicMock()
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:23:        register_builtin_tools(registry=mock_registry)
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:25:        # register() should have been called once per tool class.
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:26:        assert mock_registry.register.call_count == len(_ALL_TOOL_CLASSES)
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:28:    def test_registered_objects_are_tool_instances(self):
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:29:        """Each argument to registry.register() is an instance of the corresponding class."""
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:30:        from missy.tools.base import BaseTool
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:32:        mock_registry = MagicMock()
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:33:        register_builtin_tools(registry=mock_registry)
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:35:        for call_args in mock_registry.register.call_args_list:
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:41:    """Lines 194-196: register_builtin_tools() with registry=None uses get_tool_registry().
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:44:        from missy.tools.registry import get_tool_registry
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:45:    so we must patch at the registry module level, not on the __init__ module.
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:48:    def test_calls_get_tool_registry_when_no_registry_provided(self):
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:49:        """When registry=None, get_tool_registry() is called to obtain the singleton."""
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:50:        mock_registry = MagicMock()
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:53:            "missy.tools.registry.get_tool_registry",
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:54:            return_value=mock_registry,
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:56:            register_builtin_tools()
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:60:    def test_registers_all_tools_into_process_level_registry(self):
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:61:        """When registry=None, all tools are registered into the returned singleton."""
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:62:        mock_registry = MagicMock()
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:65:            "missy.tools.registry.get_tool_registry",
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:66:            return_value=mock_registry,
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:68:            register_builtin_tools()
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:70:        assert mock_registry.register.call_count == len(_ALL_TOOL_CLASSES)
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:72:    def test_get_tool_registry_raises_runtime_error_propagates(self):
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:73:        """RuntimeError from get_tool_registry (not yet initialised) propagates to caller."""
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:76:                "missy.tools.registry.get_tool_registry",
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:77:                side_effect=RuntimeError("registry not initialised"),
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:79:            pytest.raises(RuntimeError, match="registry not initialised"),
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:81:            register_builtin_tools()
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:83:    def test_none_is_treated_same_as_omitting_registry(self):
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:84:        """Explicitly passing registry=None is identical to calling with no argument."""
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:85:        mock_registry = MagicMock()
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:88:            "missy.tools.registry.get_tool_registry",
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:89:            return_value=mock_registry,
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:91:            register_builtin_tools(registry=None)
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:94:        assert mock_registry.register.call_count == len(_ALL_TOOL_CLASSES)
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:100:    def test_all_tool_classes_non_empty(self):
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:109:    def test_tts_tools_present(self):
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:110:        from missy.tools.builtin.tts_speak import (
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:120:    def test_discord_voice_tools_present(self):
/home/missy/missy/tests/tools/test_builtin_init_coverage.py:121:        from missy.tools.builtin.discord_voice import (
/home/missy/missy/HUMANIZE_TEST_PLAN.md:9:- Mock LLM/provider calls. Behavioral tests should assert prompt fragments, state transitions, audit entries, cooldown decisions, or emitted channel timing calls.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:10:- Keep security and reliability separate from style: humanistic behaviors must not bypass policy, mutate tool results, or hide errors.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:22:  - Block flush at `text_end` and before tool execution.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:27:- A2 policy coverage: `tests/policy/test_tool_policy_pipeline.py`
/home/missy/missy/HUMANIZE_TEST_PLAN.md:29:  - Glob allow rules and inline `-tool` deny syntax compose in one layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:30:  - `alsoAllow` can restore matching tools after a restrictive layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:31:  - Unknown plugin-only allowlists warn without hiding core tools.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:32:  - Standard profile → provider → global → agent → group → sandbox → subagent layer ordering records trace labels.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:33:  - Config-backed provider/global/agent/sandbox/subagent layers preserve ordering and source labels.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:34:  - Custom `tools.groups` definitions extend the built-in group map.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:37:  - `tools.*`, `tools.byProvider`, nested `byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools` parse from YAML.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:38:  - Invalid tool profiles fail with a configuration error.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:40:  - `AgentRuntime._get_tools()` records a `ToolPolicyDecision` and filters `safe-chat` through the A2 profile layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:42:  - `AgentRuntime._get_tools()` consumes config-backed global and agent policy surfaces.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:54:| H_G | Apology appears for a tool failure once; gratitude and hedging do not duplicate in the same exchange. |
/home/missy/missy/tests/tools/test_registry_hardening.py:1:"""Hardening tests for ToolRegistry: execution paths, policy checks, audit events.
/home/missy/missy/tests/tools/test_registry_hardening.py:4:audit event emission, and singleton management.
/home/missy/missy/tests/tools/test_registry_hardening.py:14:from missy.tools.base import BaseTool, ToolPermissions, ToolResult
/home/missy/missy/tests/tools/test_registry_hardening.py:15:from missy.tools.registry import ToolRegistry, get_tool_registry, init_tool_registry
/home/missy/missy/tests/tools/test_registry_hardening.py:18:# Test tools
/home/missy/missy/tests/tools/test_registry_hardening.py:32:    name = "network_tool"
/home/missy/missy/tests/tools/test_registry_hardening.py:93:        tool = EchoTool()
/home/missy/missy/tests/tools/test_registry_hardening.py:94:        reg.register(tool)
/home/missy/missy/tests/tools/test_registry_hardening.py:95:        assert reg.get("echo") is tool
/home/missy/missy/tests/tools/test_registry_hardening.py:101:    def test_list_tools_sorted(self):
/home/missy/missy/tests/tools/test_registry_hardening.py:106:        names = reg.list_tools()
/home/missy/missy/tests/tools/test_registry_hardening.py:113:        tool1 = EchoTool()
/home/missy/missy/tests/tools/test_registry_hardening.py:114:        tool2 = EchoTool()
/home/missy/missy/tests/tools/test_registry_hardening.py:115:        reg.register(tool1)
/home/missy/missy/tests/tools/test_registry_hardening.py:116:        reg.register(tool2)
/home/missy/missy/tests/tools/test_registry_hardening.py:117:        assert reg.get("echo") is tool2
/home/missy/missy/tests/tools/test_registry_hardening.py:119:    def test_list_tools_empty(self):
/home/missy/missy/tests/tools/test_registry_hardening.py:121:        assert reg.list_tools() == []
/home/missy/missy/tests/tools/test_registry_hardening.py:130:    def test_execute_simple_tool(self):
/home/missy/missy/tests/tools/test_registry_hardening.py:142:    def test_execute_crashing_tool_returns_error(self):
/home/missy/missy/tests/tools/test_registry_hardening.py:149:    def test_execute_failing_tool(self):
/home/missy/missy/tests/tools/test_registry_hardening.py:157:        """session_id and task_id should not be passed to tool.execute()."""
/home/missy/missy/tests/tools/test_registry_hardening.py:172:    def test_no_permissions_skips_policy(self):
/home/missy/missy/tests/tools/test_registry_hardening.py:173:        """Tools with no permissions should not trigger policy checks."""
/home/missy/missy/tests/tools/test_registry_hardening.py:176:        # Even without policy engine, this should work
/home/missy/missy/tests/tools/test_registry_hardening.py:177:        with patch("missy.tools.registry.get_policy_engine", side_effect=RuntimeError("no engine")):
/home/missy/missy/tests/tools/test_registry_hardening.py:182:        """Network tool denied by policy returns failure result."""
/home/missy/missy/tests/tools/test_registry_hardening.py:190:        with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine):
```
