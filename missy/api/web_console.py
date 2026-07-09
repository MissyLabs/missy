"""HTML helpers for the local operator console."""

from __future__ import annotations

import html


def console_css() -> str:
    """Return the embedded Web TUI stylesheet."""
    return """
:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2e;--panel2:#18243c;--text:#edf4ff;--muted:#9fb0cc;--line:#2a3858;--accent:#63d2ff;--ok:#7ee787;--warn:#ffd166;--bad:#ff7b72}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at top left,#193158 0,#0b1020 36rem);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:1.25rem clamp(1rem,4vw,2.5rem);border-bottom:1px solid var(--line);background:rgba(11,16,32,.82);backdrop-filter:blur(14px);position:sticky;top:0;z-index:2}
h1,h2,h3,p{margin:0}h1{font-size:clamp(1.3rem,3vw,2rem)}h2{font-size:clamp(1.8rem,4vw,3rem);letter-spacing:0}h3{font-size:1rem}.eyebrow{color:var(--accent);font-size:.75rem;text-transform:uppercase;font-weight:700;letter-spacing:.12em}.muted{color:var(--muted)}
button,.button-link{border:1px solid #3b82f6;background:#1d4ed8;color:white;border-radius:8px;padding:.7rem 1rem;font-weight:700;cursor:pointer;text-decoration:none;display:inline-block}button:hover,.button-link:hover{background:#2563eb}button.secondary{background:#0f172a;border-color:var(--line);color:var(--text)}button.secondary:hover{background:#17213a}
.console-shell{width:min(1180px,100%);margin:0 auto;padding:clamp(1rem,3vw,2rem)}.hero{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,520px);gap:1rem;align-items:stretch;margin-bottom:1rem}
.hero>div,.panel{background:linear-gradient(180deg,rgba(24,36,60,.95),rgba(18,26,46,.95));border:1px solid var(--line);border-radius:8px;padding:1rem;box-shadow:0 16px 40px rgba(0,0,0,.22)}
.status-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem}.status-grid article{background:#0f172a;border:1px solid var(--line);border-radius:8px;padding:.9rem}.status-grid span{display:block;font-size:1.8rem;font-weight:800}.status-grid p{color:var(--muted);font-size:.85rem}
.panel-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}.panel-head{display:flex;align-items:center;justify-content:space-between;gap:.75rem;margin-bottom:.75rem}.pill{border:1px solid var(--line);border-radius:999px;padding:.25rem .55rem;color:var(--muted);font-size:.78rem}.pill.secure{color:var(--ok);border-color:#2f6f48}
.list{display:grid;gap:.5rem}.row{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;border-top:1px solid var(--line);padding:.7rem 0}.row:first-child{border-top:0}.row strong{min-width:0;overflow-wrap:anywhere}.row span{color:var(--muted);text-align:right;overflow-wrap:anywhere}.row-actions{display:flex;align-items:center;justify-content:flex-end;gap:.5rem;flex-wrap:wrap}.row-actions span{text-align:left}.ok{color:var(--ok)!important}.warn{color:var(--warn)!important}.empty{border:1px dashed var(--line);border-radius:8px;color:var(--muted);padding:1rem;text-align:center}
.filter-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem;margin-bottom:.5rem}.filter-row select,.filter-row input{min-width:0;border:1px solid var(--line);background:#0f172a;color:var(--text);border-radius:8px;padding:.6rem}.diagnostics-panel .row span{font-size:.82rem}.diagnostics-panel em{display:block;color:var(--muted);font-style:normal;margin-top:.25rem}.audit-actions{display:flex;gap:.5rem;margin:.25rem 0 .5rem}.audit-actions button{padding:.45rem .7rem}.audit-actions button:disabled{opacity:.45;cursor:not-allowed}.audit-row{width:100%;background:transparent;border:0;border-top:1px solid var(--line);border-radius:0;color:var(--text);padding:.7rem 0;text-align:left}.audit-row:hover,.audit-row:focus{background:rgba(99,210,255,.08);outline:1px solid var(--line)}.audit-row span{font-size:.82rem}.detail{max-height:18rem;overflow:auto;margin:.75rem 0 0;border:1px solid var(--line);border-radius:8px;background:#0f172a;color:var(--muted);padding:.75rem;white-space:pre-wrap;overflow-wrap:anywhere}
.run-console{margin-bottom:1rem}.run-console textarea{width:100%;min-height:4.5rem;resize:vertical;border:1px solid var(--line);background:#0f172a;color:var(--text);border-radius:8px;padding:.7rem;font:inherit}.run-form-actions{display:flex;gap:.5rem;margin-top:.6rem}.run-log{display:grid;gap:.4rem;margin-top:.85rem;max-height:14rem;overflow:auto}.run-log:empty{display:none}.run-log .run-event{border-left:3px solid var(--line);padding:.35rem .6rem;font-size:.82rem;color:var(--muted);background:#0f172a;border-radius:0 6px 6px 0}.run-log .run-event.tool{border-left-color:var(--accent)}.run-log .run-event.error{border-left-color:var(--bad);color:var(--bad)}.run-log .run-event.complete{border-left-color:var(--ok);color:var(--ok)}.run-console .detail{margin-top:.75rem}
.login-body{display:grid;place-items:center;padding:1rem}.login-panel{width:min(440px,100%);background:rgba(18,26,46,.96);border:1px solid var(--line);border-radius:8px;padding:1.25rem;box-shadow:0 20px 60px rgba(0,0,0,.32)}.brand-mark{width:3rem;height:3rem;display:grid;place-items:center;border-radius:8px;background:#1d4ed8;font-weight:900;margin-bottom:1rem}.login-panel form{display:grid;gap:.75rem;margin-top:1rem}.login-panel label{font-weight:700}.login-panel input{width:100%;border:1px solid var(--line);background:#0f172a;color:var(--text);border-radius:8px;padding:.8rem}.error{color:var(--bad);margin-top:.75rem}
@media (max-width:820px){.hero,.panel-grid{grid-template-columns:1fr}.status-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.topbar{position:static;align-items:flex-start}.row{display:grid}.row span{text-align:left}.row-actions{justify-content:flex-start}}
@media (max-width:520px){.filter-row{grid-template-columns:1fr}}
"""


def render_console(*, csrf_token: str) -> str:
    """Render the authenticated operator console shell."""
    csrf = html.escape(csrf_token, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Missy Operator Console</title>
  <style>{console_css()}</style>
</head>
<body>
  <header class="topbar">
    <div><p class="eyebrow">Local control plane</p><h1>Missy Operator Console</h1></div>
    <button id="logout" type="button">Sign out</button>
  </header>
  <main class="console-shell" data-csrf="{csrf}">
    <section class="hero">
      <div>
        <p class="eyebrow">Runtime posture</p>
        <h2 id="runtime-status">Loading status...</h2>
        <p id="runtime-summary" class="muted">Checking providers, tools, sessions, and memory.</p>
      </div>
      <div class="status-grid" aria-label="Runtime metrics">
        <article><span id="provider-count">-</span><p>Providers</p></article>
        <article><span id="tool-count">-</span><p>Tools</p></article>
        <article><span id="session-count">-</span><p>Sessions</p></article>
        <article><span id="memory-state">-</span><p>Memory</p></article>
      </div>
    </section>
    <section class="panel run-console" aria-labelledby="run-console-heading">
      <div class="panel-head">
        <h3 id="run-console-heading">Ask Missy</h3>
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
      <article class="panel"><div class="panel-head"><h3>Providers</h3><span id="provider-health" class="pill">Loading</span></div><div id="providers" class="list"></div></article>
      <article class="panel"><div class="panel-head"><h3>Tools</h3><span id="tool-health" class="pill">Loading</span></div><div id="tools" class="list"></div></article>
      <article class="panel"><div class="panel-head"><h3>Sessions</h3><span class="pill">Recent</span></div><div id="sessions" class="list"></div></article>
      <article class="panel diagnostics-panel"><div class="panel-head"><h3>Diagnostics</h3><span id="diagnostics-health" class="pill">Loading</span></div><div id="diagnostics" class="list"></div></article>
      <article class="panel"><div class="panel-head"><h3>Controls</h3><span id="controls-health" class="pill">Loading</span></div><div id="controls" class="list"></div></article>
      <article class="panel audit-panel"><div class="panel-head"><h3>Audit Trail</h3><span id="audit-health" class="pill">Loading</span></div>
        <div class="filter-row" aria-label="Audit filters">
          <select id="audit-result" aria-label="Audit result"><option value="">All results</option><option value="deny">Denied</option><option value="allow">Allowed</option><option value="error">Errors</option></select>
          <select id="audit-severity" aria-label="Audit severity"><option value="">All severities</option><option value="critical">Critical</option><option value="warning">Warning</option><option value="info">Info</option></select>
          <select id="audit-subsystem" aria-label="Audit subsystem"><option value="">All subsystems</option><option value="auth">Auth</option><option value="security">Security</option><option value="network">Network</option><option value="tool">Tools</option><option value="provider">Providers</option></select>
          <input id="audit-actor" type="search" placeholder="Actor" aria-label="Audit actor">
          <input id="audit-source" type="search" placeholder="Source" aria-label="Audit source">
          <input id="audit-query" type="search" placeholder="Search redacted events" aria-label="Audit search">
          <input id="audit-since" type="datetime-local" aria-label="Audit since timestamp">
          <input id="audit-until" type="datetime-local" aria-label="Audit until timestamp">
        </div>
        <div class="audit-actions"><button id="audit-prev" type="button">Previous</button><button id="audit-next" type="button">Next</button></div>
        <div id="audit" class="list"></div>
        <pre id="audit-detail" class="detail" tabindex="0" aria-label="Selected audit event detail">Select an event to inspect details.</pre>
      </article>
      <article class="panel"><div class="panel-head"><h3>Security</h3><span class="pill secure">Local</span></div><div class="list">
        <div class="row"><strong>Authentication</strong><span>Cookie session + API key</span></div>
        <div class="row"><strong>CSRF</strong><span>Required for browser actions</span></div>
        <div class="row"><strong>Headers</strong><span>CSP, no-store, frame deny</span></div>
        <div class="row"><strong>Network</strong><span>Loopback by default</span></div>
      </div></article>
    </section>
  </main>
  <script>{console_script()}</script>
</body>
</html>"""


def console_script() -> str:
    """Return the embedded Web TUI JavaScript."""
    return r"""
const root = document.querySelector('.console-shell');
const csrf = root.dataset.csrf;
async function api(path, options = {}) {
  const response = await fetch('/api/v1' + path, {
    ...options,
    headers: {'Accept': 'application/json', 'X-CSRF-Token': csrf, ...(options.headers || {})},
    credentials: 'same-origin'
  });
  if (!response.ok) throw new Error(path + ' returned ' + response.status);
  return response.json();
}
function setText(id, value) { document.getElementById(id).textContent = value; }
function empty(label) { return `<div class="empty">${label}</div>`; }
function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
function renderRows(id, rows, fallback) {
  document.getElementById(id).innerHTML = rows.length ? rows.join('') : empty(fallback);
}
let auditOffset = 0;
let latestAuditEvents = [];
function isoLocalValue(id) {
  const value = document.getElementById(id).value;
  return value ? new Date(value).toISOString() : '';
}
function auditPath() {
  const params = new URLSearchParams({limit: '8', offset: String(auditOffset)});
  for (const [id, key] of [
    ['audit-result', 'result'],
    ['audit-severity', 'severity'],
    ['audit-subsystem', 'subsystem'],
    ['audit-actor', 'actor'],
    ['audit-source', 'source'],
    ['audit-query', 'q']
  ]) {
    const value = document.getElementById(id).value.trim();
    if (value) params.set(key, value);
  }
  const since = isoLocalValue('audit-since');
  const until = isoLocalValue('audit-until');
  if (since) params.set('since', since);
  if (until) params.set('until', until);
  return '/audit?' + params.toString();
}
function renderAuditDetail(event) {
  const detail = document.getElementById('audit-detail');
  detail.textContent = event ? JSON.stringify(event, null, 2) : 'Select an event to inspect details.';
}
async function loadConsole() {
  try {
    const [status, providers, tools, sessions, diagnostics, controls, audit] = await Promise.all([
      api('/status'), api('/providers'), api('/tools'), api('/sessions?limit=8'), api('/diagnostics'), api('/controls'), api(auditPath())
    ]);
    const s = status.data;
    setText('runtime-status', 'Runtime online');
    setText('runtime-summary', `Default provider: ${s.default_provider || 'not configured'}`);
    setText('provider-count', (s.providers_available || []).length);
    setText('tool-count', s.tool_count || 0);
    setText('session-count', s.session_count || 0);
    setText('memory-state', s.memory && s.memory.has_memory ? 'On' : 'Idle');
    const providerRows = providers.data.providers.map(p => `<div class="row"><strong>${esc(p.name)}</strong><span class="${p.available ? 'ok' : 'warn'}">${p.available ? 'available' : 'offline'}${p.is_default ? ' / default' : ''}</span></div>`);
    renderRows('providers', providerRows, 'No providers registered.');
    setText('provider-health', providerRows.length ? 'Ready' : 'Empty');
    const toolRows = tools.data.tools.slice(0, 12).map(t => `<div class="row"><strong>${esc(t.name)}</strong><span>${esc(t.description || 'No description')}</span></div>`);
    renderRows('tools', toolRows, 'No tools registered.');
    setText('tool-health', `${tools.data.tools.length} total`);
    const sessionRows = sessions.data.sessions.map(s => `<div class="row"><strong>${esc(s.name || s.session_id.slice(0, 8))}</strong><span>${esc(s.provider || 'provider unset')} / ${s.turn_count} turns</span></div>`);
    renderRows('sessions', sessionRows, 'No API sessions yet.');
    const diagnosticRows = diagnostics.data.sections.map(section => {
      const summary = section.checks.slice(0, 3).map(check => `${check.name}: ${typeof check.summary === 'object' ? JSON.stringify(check.summary) : check.summary}`).join(' / ');
      const remediation = section.checks.find(check => check.remediation)?.remediation || '';
      const statusClass = section.status === 'ok' ? 'ok' : 'warn';
      const hint = remediation ? `<em>${esc(remediation)}</em>` : '';
      return `<div class="row"><strong>${esc(section.label)}</strong><span class="${statusClass}">${esc(section.status)} &middot; ${esc(summary)}${hint}</span></div>`;
    });
    renderRows('diagnostics', diagnosticRows, 'Diagnostics are unavailable.');
    setText('diagnostics-health', diagnostics.data.overall);
    const controlRows = controls.data.controls.flatMap(control => control.targets.map(target => {
      const disabled = !control.enabled || !target.available || target.is_current ? 'disabled' : '';
      const label = target.is_current ? 'Current' : (target.action_label || control.label);
      const state = target.state || (target.available ? (target.is_current ? 'current default' : 'available') : 'offline');
      const targetLabel = target.label || target.name;
      const meta = [target.provider, target.schedule].filter(Boolean).join(' / ');
      const title = meta ? `${targetLabel} (${meta})` : targetLabel;
      return `<div class="row"><strong>${esc(title)}</strong><div class="row-actions"><span class="${target.available ? 'ok' : 'warn'}">${esc(state)}</span><button class="secondary control-action" type="button" data-control-id="${esc(control.id)}" data-control-label="${esc(control.label)}" data-target="${esc(target.name)}" data-target-label="${esc(targetLabel)}" data-confirmation="${esc(target.confirmation)}" ${disabled}>${esc(label)}</button></div></div>`;
    }));
    renderRows('controls', controlRows, 'No safe controls are available.');
    setText('controls-health', controlRows.length ? `${controlRows.length} targets` : 'Empty');
    latestAuditEvents = audit.data.events;
    const auditRows = latestAuditEvents.map(e => {
      const d = e.detail || {};
      const resultClass = e.result === 'deny' || e.result === 'error' ? 'warn' : 'ok';
      return `<button class="row audit-row" type="button" data-event-id="${esc(e.id)}"><strong>${esc(d.subsystem || e.category)} / ${esc(d.action || e.event_type)}</strong><span class="${resultClass}">${esc(e.result)} &middot; ${esc(d.severity || 'info')} &middot; ${esc(d.actor || 'system')} &middot; ${esc(e.timestamp || '')}</span></button>`;
    });
    renderRows('audit', auditRows, 'No audit events match these filters.');
    setText('audit-health', `${audit.data.total} total`);
    document.getElementById('audit-prev').disabled = auditOffset <= 0;
    document.getElementById('audit-next').disabled = !audit.data.has_more;
    if (!latestAuditEvents.some(e => e.id === (document.getElementById('audit-detail').dataset.eventId || ''))) {
      renderAuditDetail(latestAuditEvents[0] || null);
      document.getElementById('audit-detail').dataset.eventId = latestAuditEvents[0]?.id || '';
    }
  } catch (error) {
    setText('runtime-status', 'Console degraded');
    setText('runtime-summary', error.message);
  }
}
document.getElementById('logout').addEventListener('click', async () => {
  await fetch('/logout', {method: 'POST', headers: {'X-CSRF-Token': csrf}, credentials: 'same-origin'});
  window.location = '/login';
});
document.getElementById('audit-result').addEventListener('change', loadConsole);
document.getElementById('audit-severity').addEventListener('change', () => { auditOffset = 0; loadConsole(); });
document.getElementById('audit-subsystem').addEventListener('change', loadConsole);
for (const id of ['audit-result', 'audit-subsystem', 'audit-actor', 'audit-source', 'audit-query', 'audit-since', 'audit-until']) {
  document.getElementById(id).addEventListener('input', () => { auditOffset = 0; loadConsole(); });
}
document.getElementById('audit-prev').addEventListener('click', () => {
  auditOffset = Math.max(0, auditOffset - 8);
  loadConsole();
});
document.getElementById('audit-next').addEventListener('click', () => {
  auditOffset += 8;
  loadConsole();
});
document.getElementById('audit').addEventListener('click', event => {
  const row = event.target.closest('[data-event-id]');
  if (!row) return;
  const selected = latestAuditEvents.find(e => e.id === row.dataset.eventId);
  renderAuditDetail(selected);
  document.getElementById('audit-detail').dataset.eventId = selected?.id || '';
});
document.getElementById('controls').addEventListener('click', async event => {
  const button = event.target.closest('[data-control-id]');
  if (!button || button.disabled) return;
  const target = button.dataset.target;
  const confirmation = button.dataset.confirmation;
  const targetLabel = button.dataset.targetLabel || target;
  const controlLabel = button.dataset.controlLabel || 'Run control';
  if (!window.confirm(`${controlLabel}: ${targetLabel}?`)) return;
  button.disabled = true;
  await api('/controls/' + encodeURIComponent(button.dataset.controlId), {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({target, confirm: confirmation})
  });
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
    appendRunEvent('Run complete', '', 'complete');
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


def render_login(*, error: bool = False) -> str:
    """Render the operator login page."""
    error_html = (
        '<p class="error" role="alert">Invalid operator key. Try again.</p>' if error else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Missy Operator Login</title>
  <style>{console_css()}</style>
</head>
<body class="login-body">
  <main class="login-panel" aria-labelledby="login-title">
    <div class="brand-mark">M</div>
    <h1 id="login-title">Missy Operator Console</h1>
    <p class="muted">Local control plane access requires the configured API key.</p>
    {error_html}
    <form method="post" action="/login">
      <label for="api_key">Operator key</label>
      <input id="api_key" name="api_key" type="password" autocomplete="current-password" required autofocus>
      <button type="submit">Enter Console</button>
    </form>
  </main>
</body>
</html>"""


def render_message(title: str, message: str) -> str:
    """Render a compact operator message page."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{console_css()}</style></head>
<body class="login-body"><main class="login-panel"><h1>{html.escape(title)}</h1>
<p class="muted">{html.escape(message)}</p><a class="button-link" href="/">Back to console</a></main></body></html>"""
