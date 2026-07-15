#!/usr/bin/env node
// Minimal, purpose-built ACP client bridging Missy to @zed-industries/claude-agent-acp
// directly, bypassing acpx's CLI.
//
// Why this exists (6th tool-specific validation run headline finding): acpx's own CLI
// only forwards `model`/`allowedTools`/`maxTurns` into the ACP `session/new` request's
// `_meta.claudeCode.options` -- it has no flag for the two fields that actually matter
// here, which claude-agent-acp genuinely supports at the protocol level:
//   - `_meta.systemPrompt`: a real system-role prompt, delivered through the agent's
//     own system-prompt channel rather than smuggled as plain text inside a single
//     user-role message (which is what acpx's CLI-based invocation forces Missy into,
//     and which the delegate sometimes -- correctly, from its own safety standpoint --
//     flagged as a jailbreak/identity-override attempt against itself).
//   - `_meta.disableBuiltInTools`: genuinely removes the agent's own native
//     Read/Write/Bash/etc. tools from its own tool list, rather than exposing them and
//     denying every call after the fact (acpx's `--deny-all`), which is what let the
//     delegate see genuine, repeated tool-permission-denial events and react to them
//     unpredictably.
//
// Protocol reference: https://agentclientprotocol.com/ , using the official
// @agentclientprotocol/sdk client-side helpers (the same package claude-agent-acp
// itself is built against), not a hand-rolled JSON-RPC transport.
//
// Usage: node acp_bridge.mjs < json-request-on-stdin
//   Request: {"cwd": "...", "systemPrompt": "...", "prompt": "...", "timeoutMs": 120000}
//   Output (stdout, NDJSON -- one JSON object per line, matching the event shapes
//   AcpxProvider._extract_text_from_event() already recognises, so no parsing changes
//   are needed on the Python side for either buffered or real-time-streaming callers):
//     {"type": "text_delta", "delta": "..."}   -- zero or more, as text arrives
//     {"type": "result", "ok": true, "stopReason": "..."}       -- success, final line
//     {"type": "result", "ok": false, "error": "..."}           -- failure, final line
//     (process exit code is 0 on success, non-zero on failure, mirroring the final line)
// Every permission request is auto-denied (fail-closed), mirroring acpx's --deny-all
// posture as defense-in-depth even with disableBuiltInTools set.

import { spawn } from "node:child_process";
import { Writable, Readable } from "node:stream";
import * as acp from "@agentclientprotocol/sdk";

// Matches acpx's own resolution exactly (acpx spawns "@latest" today,
// confirmed via `acpx --verbose`) rather than a version-pinned range --
// note that for a 0.x package, npm's caret range (`^0.21.0`) only allows
// patch bumps (0.21.x), NOT the 0.23.x releases actually in use, so a
// stale pin here would silently resolve to an incompatible version whose
// session/new call fails outright ("Query closed before response
// received") rather than a same-version, working one.
const AGENT_COMMAND = "npx";
const AGENT_ARGS = ["-y", "@zed-industries/claude-agent-acp@latest"];

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function main() {
  const raw = await readStdin();
  let request;
  try {
    request = JSON.parse(raw);
  } catch (err) {
    process.stdout.write(
      JSON.stringify({ type: "result", ok: false, error: `invalid request JSON: ${err.message}` }) + "\n",
    );
    process.exitCode = 1;
    return;
  }

  const { cwd, systemPrompt, prompt, timeoutMs } = request;
  const effectiveTimeout = typeof timeoutMs === "number" && timeoutMs > 0 ? timeoutMs : 120000;

  const agentProcess = spawn(AGENT_COMMAND, AGENT_ARGS, {
    cwd,
    stdio: ["pipe", "pipe", "pipe"],
  });

  let stderrBuf = "";
  agentProcess.stderr.on("data", (d) => {
    stderrBuf += d.toString();
  });

  const input = Writable.toWeb(agentProcess.stdin);
  const output = Readable.toWeb(agentProcess.stdout);
  const stream = acp.ndJsonStream(input, output);

  // Every permission request is denied outright: disableBuiltInTools should mean
  // the agent's own tool list never even offers Read/Write/Bash/etc., so this
  // should rarely if ever fire -- but if it does (a future SDK version exposing a
  // tool disableBuiltInTools doesn't cover, or an MCP server tool if any is ever
  // configured here), fail closed rather than silently allowing it.
  function denyPermission(params) {
    const rejectOption = params.options.find((o) => o.kind === "reject_once" || o.kind === "reject_always");
    const optionId = rejectOption ? rejectOption.optionId : params.options[0]?.optionId;
    return { outcome: { outcome: "selected", optionId } };
  }

  let stopReason = "unknown";
  let ok = true;
  let errorMessage = null;

  try {
    await withTimeout(
      acp
        .client({ name: "missy-acp-bridge" })
        .onRequest(acp.methods.client.session.requestPermission, (ctx) => denyPermission(ctx.params))
        .connectWith(stream, async (ctx) => {
          await ctx.request(acp.methods.agent.initialize, {
            protocolVersion: acp.PROTOCOL_VERSION,
            clientCapabilities: { fs: { readTextFile: false, writeTextFile: false } },
          });

          const sessionRequest = {
            cwd,
            mcpServers: [],
            _meta: {
              systemPrompt: systemPrompt ? { append: systemPrompt } : undefined,
              disableBuiltInTools: true,
            },
          };

          await ctx.buildSession(sessionRequest).withSession(async (session) => {
            session.prompt(prompt);
            for (;;) {
              const message = await session.nextUpdate();
              if (message.kind === "stop") {
                stopReason = message.stopReason;
                return;
              }
              const update = message.update;
              if (update.sessionUpdate === "agent_message_chunk" && update.content?.type === "text") {
                process.stdout.write(JSON.stringify({ type: "text_delta", delta: update.content.text }) + "\n");
              }
            }
          });
        }),
      effectiveTimeout,
      "acp_bridge session",
    );
  } catch (err) {
    ok = false;
    errorMessage = `${err.message}${stderrBuf ? ` (stderr: ${stderrBuf.slice(0, 2000)})` : ""}`;
  } finally {
    agentProcess.kill();
  }

  if (ok) {
    process.stdout.write(JSON.stringify({ type: "result", ok: true, stopReason }) + "\n");
  } else {
    process.stdout.write(JSON.stringify({ type: "result", ok: false, error: errorMessage }) + "\n");
    process.exitCode = 1;
  }
  // npx's own wrapper process (and/or lingering stdio handles from the
  // killed agent subprocess) can otherwise keep the event loop alive
  // past agentProcess.kill(), even though the actual work is done.
  process.exit(process.exitCode ?? 0);
}

main().catch((err) => {
  process.stdout.write(
    JSON.stringify({ type: "result", ok: false, error: `unhandled: ${err.stack || err.message}` }) + "\n",
  );
  process.exit(1);
});
