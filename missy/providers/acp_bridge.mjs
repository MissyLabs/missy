#!/usr/bin/env node
// Minimal, purpose-built ACP client bridging Missy to @agentclientprotocol/claude-agent-acp
// directly, bypassing acpx's CLI.
//
// Modern acpx exposes equivalent system/no-tools controls. Missy keeps this bridge
// because it needs one temporary turn with its own process-group cancellation and no
// persistent session/queue-owner layer. The two key session fields are:
//   - `_meta.systemPrompt`: a real system-role prompt, delivered through the agent's
//     own system-prompt channel rather than plain text in the user prompt.
//   - `_meta.disableBuiltInTools`: genuinely removes the agent's own native
//     Read/Write/Bash/etc. tools from its own tool list, rather than exposing them and
//     denying every call after the fact.
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

// Match acpx 0.19.x's built-in Claude registry range.  This accepts the
// adapter releases upstream validates while avoiding an unbounded @latest
// upgrade in a paid-provider path.
const AGENT_COMMAND = "npx";
const AGENT_ARGS = ["-y", "@agentclientprotocol/claude-agent-acp@^0.76.0"];

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

  const { cwd, systemPrompt, prompt, model, timeoutMs } = request;
  const effectiveTimeout = typeof timeoutMs === "number" && timeoutMs > 0 ? timeoutMs : 120000;

  // detached: true makes this the leader of its own process group (setsid),
  // rather than a plain child of this script's own process. `npx` does not
  // exec-replace itself with the resolved `claude-agent-acp` binary -- it
  // spawns it as a genuine child process, which in turn spawns the actual
  // `claude` CLI as a further child. A plain agentProcess.kill() only
  // signals the immediate npx PID, leaving both descendant levels running
  // as orphans (reparented to init) the instant npx itself exits -- exactly
  // the leaked `claude`/`claude-agent-acp` process pairs observed piling up
  // in production. Killing the whole group via a negative PID (see
  // killAgentProcessGroup below) is required to actually stop the tree.
  const agentProcess = spawn(AGENT_COMMAND, AGENT_ARGS, {
    cwd,
    stdio: ["pipe", "pipe", "pipe"],
    detached: true,
  });

  function killAgentProcessGroup(signal = "SIGKILL") {
    if (agentProcess.pid == null) return;
    try {
      // Negative PID signals the whole process group (only valid because
      // detached: true made agentProcess its own group leader above).
      process.kill(-agentProcess.pid, signal);
    } catch {
      // ESRCH (already exited) or some other reason the group signal
      // failed -- fall back to at least killing the immediate process.
      try {
        agentProcess.kill(signal);
      } catch {
        // Already gone; nothing left to clean up.
      }
    }
  }

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
  let usage = null;

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
            if (typeof model === "string" && model.trim()) {
              await ctx.request(acp.methods.agent.session.setConfigOption, {
                sessionId: session.sessionId,
                configId: "model",
                value: model.trim(),
              });
            }
            session.prompt(prompt);
            for (;;) {
              const message = await session.nextUpdate();
              if (message.kind === "stop") {
                stopReason = message.stopReason;
                usage = message.usage ?? message._meta?.usage ?? message._meta?.claudeCode?.usage ?? null;
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
    killAgentProcessGroup();
  }

  if (ok) {
    process.stdout.write(JSON.stringify({ type: "result", ok: true, stopReason, usage }) + "\n");
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
