"""Dashboard page: runtime posture, run console, controls, approvals."""

from __future__ import annotations


def content() -> str:
    """Return the dashboard page body."""
    return """
    <section class="hero">
      <div class="hero-status">
        <p class="eyebrow">Runtime posture</p>
        <h2 id="runtime-status">Loading status...</h2>
        <p id="runtime-summary" class="muted">Checking providers, tools, sessions, and memory.</p>
      </div>
      <div class="status-grid" aria-label="Runtime metrics">
        <article><a class="tile-link" href="/providers"><div class="tile-head"><span class="led info" aria-hidden="true"></span><p>Providers</p></div><span id="provider-count" class="value">-</span></a></article>
        <article><a class="tile-link" href="/diagnostics"><div class="tile-head"><span class="led info" aria-hidden="true"></span><p>Tools</p></div><span id="tool-count" class="value">-</span></a></article>
        <article><a class="tile-link" href="/sessions"><div class="tile-head"><span class="led info" aria-hidden="true"></span><p>Sessions</p></div><span id="session-count" class="value">-</span></a></article>
        <article><a class="tile-link" href="/memory"><div class="tile-head"><span class="led info" aria-hidden="true"></span><p>Memory</p></div><span id="memory-state" class="value">-</span></a></article>
      </div>
    </section>
    <section class="panel run-console" aria-labelledby="run-console-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">RUN&middot;00</span><h3 id="run-console-heading">Ask Missy</h3></div>
        <span id="run-status" class="pill">Idle</span>
      </div>
      <p id="run-help" class="muted">Send a message and watch the run stream live: tool calls, completion, and errors.</p>
      <form id="run-form">
        <textarea id="run-input" aria-label="Message to send to the agent" aria-describedby="run-help" rows="2" placeholder="Ask the agent something..." required></textarea>
        <div class="run-form-actions">
          <button type="submit" id="run-submit">Send</button>
          <button type="button" id="run-cancel" class="secondary" disabled>Stop watching</button>
        </div>
      </form>
      <div id="run-log" class="run-log" role="log" aria-live="polite" aria-relevant="additions" aria-atomic="false" aria-label="Run activity"></div>
      <pre id="run-response" class="detail" tabindex="0" aria-label="Latest agent response">No runs yet.</pre>
    </section>
    <section class="panel-grid">
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">CTL&middot;05</span><h3>Controls</h3></div><span id="controls-health" class="pill">Loading</span></div><div id="controls" class="list list-scroll"></div></article>
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">APR&middot;10</span><h3>Approvals</h3></div><span id="approvals-health" class="pill">Loading</span></div><div id="approvals" class="list list-scroll"></div></article>
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">PAIR&middot;11</span><h3>Discord Pairing</h3></div><span id="pairing-health" class="pill">Loading</span></div><div id="pairing" class="list list-scroll"></div></article>
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">SEC&middot;09</span><h3>Security</h3></div><span class="pill secure">Local</span></div><div class="list">
        <div class="row"><div class="row-static"><strong>Authentication</strong></div><span class="meta" title="Cookie session + API key">Cookie session + API key</span></div>
        <div class="row"><div class="row-static"><strong>CSRF</strong></div><span class="meta" title="Required for browser actions">Required for browser actions</span></div>
        <div class="row"><div class="row-static"><strong>Headers</strong></div><span class="meta" title="CSP, no-store, frame deny">CSP, no-store, frame deny</span></div>
        <div class="row"><div class="row-static"><strong>Network</strong></div><span class="meta" title="Loopback by default">Loopback by default</span></div>
      </div></article>
    </section>
"""


def script() -> str:
    """Return the dashboard page script."""
    return r"""
async function loadConsole() {
  try {
    const [status, controls, approvals, pairing] = await Promise.all([
      api('/status'), api('/controls'), api('/approvals'), api('/discord/pairing')
    ]);
    const s = status.data;
    setText('runtime-status', 'Runtime online');
    setText('runtime-summary', `Default provider: ${s.default_provider || 'not configured'}`);
    setText('provider-count', (s.providers_available || []).length);
    setText('tool-count', s.tool_count || 0);
    setText('session-count', s.session_count || 0);
    setText('memory-state', s.memory && s.memory.has_memory ? 'On' : 'Idle');

    const controlRows = controls.data.controls.flatMap(control => control.targets.map(target => {
      const disabled = !control.enabled || !target.available || target.is_current ? 'disabled' : '';
      const label = target.is_current ? 'Current' : (target.action_label || control.label);
      const state = target.state || (target.available ? (target.is_current ? 'current default' : 'available') : 'offline');
      const targetLabel = target.label || target.name;
      const meta = [target.provider, target.schedule].filter(Boolean).join(' / ');
      const title = meta ? `${targetLabel} (${meta})` : targetLabel;
      return `<div class="row"><div class="row-static"><strong>${esc(title)}</strong></div><div class="row-actions"><span class="${target.available ? 'ok' : 'warn'}">${esc(state)}</span><button class="secondary control-action" type="button" data-control-id="${esc(control.id)}" data-control-label="${esc(control.label)}" data-target="${esc(target.name)}" data-target-label="${esc(targetLabel)}" data-confirmation="${esc(target.confirmation)}" ${disabled}>${esc(label)}</button></div></div>`;
    }));
    renderRows('controls', controlRows, 'No safe controls are available.');
    setText('controls-health', controlRows.length ? `${controlRows.length} targets` : 'Empty');

    const approvalRows = approvals.data.approvals.map(a => `<div class="row row-stack"><div class="row-title"><span class="led warn pulse" aria-hidden="true"></span><strong>${esc(a.action)}</strong></div><span class="meta-line" title="${esc(a.reason || 'No reason given')}">${esc(a.reason || 'No reason given')} &middot; id ${esc(a.id)}</span><div class="row-actions"><button class="secondary approval-action" type="button" data-approval-id="${esc(a.id)}" data-approve="true">Approve</button><button class="secondary danger approval-action" type="button" data-approval-id="${esc(a.id)}" data-approve="false">Deny</button></div></div>`);
    renderRows('approvals', approvalRows, 'No pending approvals.');
    setText('approvals-health', approvalRows.length ? `${approvalRows.length} pending` : 'Empty');

    const pairingRows = pairing.data.pending.map(p => `<div class="row"><div class="row-static"><strong>${esc(p.user_id)}</strong></div><div class="row-actions"><span class="meta">${esc(p.account || 'default account')}</span><button class="secondary pairing-action" type="button" data-user-id="${esc(p.user_id)}" data-approve="true">Approve</button><button class="secondary danger pairing-action" type="button" data-user-id="${esc(p.user_id)}" data-approve="false">Deny</button></div></div>`);
    renderRows('pairing', pairingRows, 'No pending Discord pairing requests.');
    setText('pairing-health', pairingRows.length ? `${pairingRows.length} pending` : 'Empty');
  } catch (error) {
    setText('runtime-status', 'Console degraded');
    setText('runtime-summary', error.message);
  }
}
document.getElementById('controls').addEventListener('click', async event => {
  const button = event.target.closest('[data-control-id]');
  if (!button || button.disabled) return;
  const target = button.dataset.target;
  const confirmation = button.dataset.confirmation;
  const targetLabel = button.dataset.targetLabel || target;
  const controlLabel = button.dataset.controlLabel || 'Run control';
  if (!window.confirm(`${controlLabel}: ${targetLabel}?`)) return;
  button.disabled = true;
  try {
    await api('/controls/' + encodeURIComponent(button.dataset.controlId), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({target, confirm: confirmation})
    });
  } catch (error) {
    window.alert('Control failed: ' + error.message);
  }
  await loadConsole();
});
document.getElementById('approvals').addEventListener('click', async event => {
  const button = event.target.closest('.approval-action');
  if (!button || button.disabled) return;
  const approvalId = button.dataset.approvalId;
  const approve = button.dataset.approve === 'true';
  if (!window.confirm(`${approve ? 'Approve' : 'Deny'} pending action ${approvalId}?`)) return;
  button.disabled = true;
  try {
    await api(`/approvals/${encodeURIComponent(approvalId)}/${approve ? 'approve' : 'deny'}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({})
    });
  } catch (error) {
    window.alert('Could not resolve approval: ' + error.message);
  }
  await loadConsole();
});
document.getElementById('pairing').addEventListener('click', async event => {
  const button = event.target.closest('.pairing-action');
  if (!button || button.disabled) return;
  const userId = button.dataset.userId;
  const approve = button.dataset.approve === 'true';
  if (!window.confirm(`${approve ? 'Approve' : 'Deny'} Discord pairing for user ${userId}?`)) return;
  button.disabled = true;
  try {
    await api(`/discord/pairing/${encodeURIComponent(userId)}/${approve ? 'approve' : 'deny'}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({})
    });
  } catch (error) {
    window.alert('Could not resolve pairing request: ' + error.message);
  }
  await loadConsole();
});

let activeRunSource = null;
function setRunStatus(text, cls) {
  const pill = document.getElementById('run-status');
  pill.textContent = text;
  pill.className = 'pill' + (cls ? ' ' + cls : '');
}
function appendRunEvent(label, detail, cls) {
  const log = document.getElementById('run-log');
  const row = document.createElement('div');
  row.className = 'run-event' + (cls ? ' ' + cls : '');
  row.textContent = detail ? `${label} — ${detail}` : label;
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}
function closeRunStream() {
  if (activeRunSource) {
    activeRunSource.close();
    activeRunSource = null;
  }
}
function setRunBusy(busy) {
  document.getElementById('run-submit').disabled = busy;
  document.getElementById('run-cancel').disabled = !busy;
}
async function startRun(message) {
  closeRunStream();
  document.getElementById('run-log').innerHTML = '';
  document.getElementById('run-response').textContent = 'Waiting for response...';
  setRunStatus('Starting...', 'warn');
  setRunBusy(true);
  let started;
  try {
    started = await api('/runs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message})
    });
  } catch (error) {
    setRunStatus('Failed to start', 'warn');
    appendRunEvent('Error', error.message, 'error');
    setRunBusy(false);
    return;
  }
  const runId = started.data.run_id;
  setRunStatus('Running', 'warn');
  appendRunEvent('Run started', message);
  const source = new EventSource('/api/v1/runs/' + encodeURIComponent(runId) + '/events');
  activeRunSource = source;
  source.addEventListener('run.start', () => {
    appendRunEvent('Agent picked up the task');
  });
  source.addEventListener('tool.request', event => {
    const data = JSON.parse(event.data);
    appendRunEvent('Tool call', data.tool, 'tool');
  });
  source.addEventListener('tool.result', event => {
    const data = JSON.parse(event.data);
    appendRunEvent('Tool result', `${data.tool} ${data.is_error ? 'failed' : 'ok'}`, data.is_error ? 'error' : 'tool');
  });
  source.addEventListener('run.complete', event => {
    const data = JSON.parse(event.data);
    setRunStatus('Complete', 'ok');
    const cost = data.cost && data.cost.total_cost_usd != null ? `$${Number(data.cost.total_cost_usd).toFixed(4)}` : null;
    const summary = [data.provider ? `provider: ${data.provider}` : null, (data.tools_used || []).length ? `tools: ${data.tools_used.join(', ')}` : null, cost ? `cost: ${cost}` : null].filter(Boolean).join(' · ');
    appendRunEvent('Run complete', summary, 'complete');
    document.getElementById('run-response').textContent = data.response || '(empty response)';
    setRunBusy(false);
    closeRunStream();
    loadConsole();
  });
  source.addEventListener('run.error', event => {
    const data = JSON.parse(event.data);
    setRunStatus('Error', 'warn');
    appendRunEvent('Run failed', data.error, 'error');
    document.getElementById('run-response').textContent = 'Run failed: ' + (data.error || 'unknown error');
    setRunBusy(false);
    closeRunStream();
  });
  source.onerror = () => {
    if (!activeRunSource) return;
    setRunStatus('Connection lost', 'warn');
    appendRunEvent('Stream connection lost', '', 'error');
    setRunBusy(false);
    closeRunStream();
  };
}
document.getElementById('run-form').addEventListener('submit', event => {
  event.preventDefault();
  const input = document.getElementById('run-input');
  const message = input.value.trim();
  if (!message) return;
  startRun(message);
  input.value = '';
});
document.getElementById('run-input').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    document.getElementById('run-form').requestSubmit();
  }
});
document.getElementById('run-cancel').addEventListener('click', () => {
  closeRunStream();
  setRunStatus('Stopped watching', 'warn');
  setRunBusy(false);
});
loadConsole();
setInterval(loadConsole, 15000);
"""
