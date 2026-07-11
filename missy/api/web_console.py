"""HTML helpers for the local operator console."""

from __future__ import annotations

import html


def console_css() -> str:
    """Return the embedded Web TUI stylesheet.

    Visual identity: a rack-console / signal-light aesthetic — square LED
    indicators (ok/warn/crit/info) are the one recurring motif tying every
    module together, monospace type carries data (ids, counts, JSON),
    and a display grotesk carries structure (module codes, headings).
    """
    return """
:root{color-scheme:dark;
  --void:#08090c;--panel:#101317;--panel-raised:#161b21;--line:#242a31;--line-soft:#1a1f25;
  --text:#e8ebef;--muted:#838d99;--muted-dim:#5b636e;
  --ok:#3ddc84;--warn:#f5b942;--crit:#ef5350;--info:#4fb3ff;
  --mono:ui-monospace,"SF Mono","Cascadia Code","JetBrains Mono",Consolas,monospace;
  --sans:ui-sans-serif,"Segoe UI",system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
}
*{box-sizing:border-box}
html,body{overflow-x:hidden}
body{margin:0;min-height:100vh;background:
    radial-gradient(1100px 500px at 12% -8%,rgba(79,179,255,.05),transparent 60%),
    radial-gradient(900px 460px at 100% 0%,rgba(61,220,132,.04),transparent 55%),
    var(--void);
  color:var(--text);font-family:var(--sans);line-height:1.45;-webkit-font-smoothing:antialiased}
h1,h2,h3,p{margin:0}
h1{font-size:clamp(1.15rem,2.4vw,1.5rem);font-weight:800;letter-spacing:-.01em}
h2{font-family:var(--mono);font-size:clamp(1.5rem,3.4vw,2.4rem);font-weight:600;letter-spacing:-.01em}
h3{font-size:.92rem;font-weight:800;letter-spacing:.01em}
.eyebrow{color:var(--info);font-size:.72rem;text-transform:uppercase;font-weight:800;letter-spacing:.16em;font-family:var(--mono)}
.muted{color:var(--muted)}
a{color:var(--info)}
:focus-visible{outline:2px solid var(--info);outline-offset:2px}

/* Signature device: square signal-light indicators */
.led{display:inline-block;width:.5rem;height:.5rem;border-radius:1px;background:var(--muted-dim);flex:none}
.led.ok{background:var(--ok);box-shadow:0 0 0 2px rgba(61,220,132,.12),0 0 8px rgba(61,220,132,.55)}
.led.warn{background:var(--warn);box-shadow:0 0 0 2px rgba(245,185,66,.12),0 0 8px rgba(245,185,66,.55)}
.led.crit{background:var(--crit);box-shadow:0 0 0 2px rgba(239,83,80,.12),0 0 8px rgba(239,83,80,.55)}
.led.info{background:var(--info);box-shadow:0 0 0 2px rgba(79,179,255,.12),0 0 8px rgba(79,179,255,.5)}
.led.pulse{animation:pulse 1.8s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){.led.pulse{animation:none}}

button,.button-link{border:1px solid #2f6fe0;background:#1f56c9;color:#fff;border-radius:3px;padding:.6rem .95rem;font-weight:700;font-size:.85rem;cursor:pointer;text-decoration:none;display:inline-block;font-family:var(--sans);transition:background .12s ease}
button:hover,.button-link:hover{background:#2764e8}
button.secondary{background:var(--panel-raised);border-color:var(--line);color:var(--text)}
button.secondary:hover{background:#1c222a}
button:disabled{opacity:.4;cursor:not-allowed}
button.danger{border-color:#6b2323;background:#3a1414;color:#ffb4b0}
button.danger:hover{background:#4a1818}

.topbar{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:1rem clamp(1rem,4vw,2.5rem);border-bottom:1px solid var(--line);background:rgba(8,9,12,.9);backdrop-filter:blur(14px);position:sticky;top:0;z-index:20}
.brand{display:flex;align-items:center;gap:.65rem}
.brand-mark{width:2.1rem;height:2.1rem;display:grid;place-items:center;border-radius:3px;background:linear-gradient(160deg,#1f56c9,#123a8f);font-weight:900;font-family:var(--mono);font-size:.95rem;border:1px solid #2f6fe0}
.console-shell{width:100%;margin:0;padding:1.25rem clamp(1rem,2.4vw,2.5rem) 3rem}

.hero{display:flex;flex-wrap:wrap;gap:1rem;align-items:stretch;margin-bottom:1rem}
.hero-status{flex:1 1 360px;max-width:680px}
.hero>div,.panel{background:linear-gradient(180deg,var(--panel-raised),var(--panel));border:1px solid var(--line);border-radius:3px;padding:1.1rem;position:relative}
.hero>div::before,.panel::before{content:"";position:absolute;inset:0 0 auto 0;height:1px;background:linear-gradient(90deg,rgba(255,255,255,.08),transparent 60%)}
.status-grid{flex:2 1 480px;display:flex;flex-wrap:wrap;gap:.6rem}
.status-grid article{flex:1 1 160px;background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:.85rem}
.status-grid .tile-head{display:flex;align-items:center;gap:.4rem;margin-bottom:.4rem}
.status-grid span.value{display:block;font-family:var(--mono);font-size:1.6rem;font-weight:600}
.status-grid p{color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.08em;font-family:var(--mono)}

.panel-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));grid-auto-flow:dense;gap:1rem;align-items:start}
.panel-grid .span-2{grid-column:span 2}
@media (max-width:1100px){.panel-grid .span-2{grid-column:span 1}}
.panel-head{display:flex;align-items:center;justify-content:space-between;gap:.75rem;margin-bottom:.75rem;padding-bottom:.6rem;border-bottom:1px solid var(--line-soft)}
.panel-id{display:flex;align-items:center;gap:.55rem;min-width:0}
.mod-code{font-family:var(--mono);font-size:.7rem;color:var(--muted-dim);letter-spacing:.06em;flex:none}
.pill{border:1px solid var(--line);border-radius:2px;padding:.2rem .5rem;color:var(--muted);font-size:.72rem;font-family:var(--mono);white-space:nowrap}
.pill.secure{color:var(--ok);border-color:#1d4a34}

.list{display:grid;grid-template-columns:minmax(0,1fr);gap:0}
.list-scroll{max-height:19rem;overflow-y:auto;overflow-x:hidden}
.row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,230px);align-items:center;column-gap:1rem;border-top:1px solid var(--line-soft);padding:0}
.row:first-child{border-top:0}
.row-title{min-width:0;text-align:left;background:transparent;border:0;border-radius:0;color:var(--text);padding:.62rem 0;font:inherit;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:.5rem}
.row-title:hover,.row-title:focus-visible{color:var(--info)}
.row-title strong{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600}
.row-static{padding:.62rem 0;min-width:0}
.row-static strong{font-weight:600}
.row span.meta{color:var(--muted);font-size:.8rem;font-family:var(--mono);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.row-stack{display:block;grid-template-columns:none;padding:.55rem 0}
.row-stack .row-title{padding:0 0 .2rem}
.row-stack .meta-line{display:block;color:var(--muted);font-size:.78rem;font-family:var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-left:1.2rem}
.row-actions{display:flex;align-items:center;justify-content:flex-end;gap:.5rem;flex-wrap:wrap;padding:.5rem 0;min-width:0}
.row-actions span{text-align:left;font-family:var(--mono);font-size:.78rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.ok{color:var(--ok)!important}.warn{color:var(--warn)!important}.crit{color:var(--crit)!important}
.empty{border:1px dashed var(--line);border-radius:3px;color:var(--muted);padding:1rem;text-align:center;font-size:.85rem}

.filter-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem;margin-bottom:.6rem}
.filter-row select,.filter-row input{min-width:0;border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:3px;padding:.55rem;font:inherit;font-size:.85rem}
.diagnostics-panel .row{display:block;padding:.6rem 0}
.diagnostics-panel .row-static{display:flex;align-items:baseline;justify-content:space-between;gap:1rem}
.diagnostics-panel em{display:block;color:var(--muted);font-style:normal;margin-top:.25rem;font-size:.8rem}
.audit-actions{display:flex;gap:.5rem;margin:.15rem 0 .6rem}
.audit-actions button{padding:.4rem .7rem;font-size:.8rem}
.audit-row{width:100%;background:transparent;border:0;border-top:1px solid var(--line-soft);border-radius:0;color:var(--text);padding:.62rem 0;text-align:left;cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:1rem;font:inherit}
.audit-row:first-child{border-top:0}
.audit-row:hover,.audit-row:focus-visible{background:rgba(79,179,255,.06)}
.audit-row span{font-size:.78rem;font-family:var(--mono)}
.detail{max-height:16rem;overflow:auto;margin:.75rem 0 0;border:1px solid var(--line);border-radius:3px;background:var(--void);color:var(--muted);padding:.75rem;white-space:pre-wrap;overflow-wrap:anywhere;font-family:var(--mono);font-size:.78rem}

.run-console{margin-bottom:1rem}
.run-help,.run-form,.run-log,.run-console .detail{max-width:1100px}
.run-console textarea{width:100%;min-height:4.5rem;resize:vertical;border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:3px;padding:.7rem;font:inherit}
.run-form-actions{display:flex;gap:.5rem;margin-top:.6rem}
.run-log{display:grid;gap:.35rem;margin-top:.85rem;max-height:13rem;overflow:auto}
.run-log:empty{display:none}
.run-log .run-event{border-left:2px solid var(--line);padding:.35rem .6rem;font-size:.8rem;font-family:var(--mono);color:var(--muted);background:var(--panel);border-radius:0 2px 2px 0}
.run-log .run-event.tool{border-left-color:var(--info)}
.run-log .run-event.error{border-left-color:var(--crit);color:var(--crit)}
.run-log .run-event.complete{border-left-color:var(--ok);color:var(--ok)}
.run-console .detail{margin-top:.75rem}

.op-form{display:grid;gap:.5rem;margin-top:.85rem;border-top:1px solid var(--line-soft);padding-top:.85rem}
.op-form input,.op-form textarea{width:100%;border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:3px;padding:.6rem;font:inherit;font-size:.85rem}
.op-form textarea{resize:vertical;min-height:3.5rem}
.op-form-actions{display:flex;justify-content:flex-end}
.op-form-actions button{padding:.5rem .9rem}
.op-form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem}
.pin-marker{color:var(--warn)}

/* Inspector: slide-over detail tray, shared by every clickable module */
.inspector-backdrop{position:fixed;inset:0;background:rgba(4,5,7,.55);opacity:0;pointer-events:none;transition:opacity .16s ease;z-index:29}
.inspector-backdrop.open{opacity:1;pointer-events:auto}
.inspector{position:fixed;top:0;right:0;bottom:0;width:min(460px,100%);background:linear-gradient(180deg,var(--panel-raised),var(--panel));border-left:1px solid var(--line);box-shadow:-24px 0 60px rgba(0,0,0,.5);transform:translateX(100%);transition:transform .18s ease;z-index:30;display:flex;flex-direction:column}
.inspector.open{transform:translateX(0)}
.inspector-head{display:flex;align-items:flex-start;justify-content:space-between;gap:.75rem;padding:1.1rem 1.1rem .9rem;border-bottom:1px solid var(--line)}
.inspector-head .mod-code{display:block;margin-bottom:.3rem}
#inspector-subtitle{color:var(--muted);font-size:.82rem;font-family:var(--mono);margin-top:.2rem}
.inspector-body{padding:1.1rem;overflow-y:auto;flex:1}
.field{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;padding:.5rem 0;border-top:1px solid var(--line-soft)}
.field:first-child{border-top:0}
.field-label{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;font-family:var(--mono);flex:none}
.field-value{text-align:right;overflow-wrap:anywhere;font-size:.86rem}
.field-block{padding:.6rem 0;border-top:1px solid var(--line-soft)}
.field-block .field-label{display:block;margin-bottom:.4rem}
.json-block{margin:0;background:var(--void);border:1px solid var(--line);border-radius:3px;padding:.75rem;font-family:var(--mono);font-size:.76rem;white-space:pre-wrap;overflow-wrap:anywhere;max-height:20rem;overflow-y:auto;color:var(--muted)}
.transcript{margin-top:.6rem;display:grid;gap:.5rem}
.transcript-turn{border:1px solid var(--line-soft);border-radius:3px;padding:.55rem .65rem;background:var(--void)}
.transcript-turn .turn-role{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--info);margin-right:.5rem}
.transcript-turn .turn-meta{font-family:var(--mono);font-size:.72rem;color:var(--muted-dim)}
.transcript-turn p{margin:.35rem 0 0;font-size:.85rem;white-space:pre-wrap;overflow-wrap:anywhere}

.login-body{display:grid;place-items:center;padding:1rem}
.login-panel{width:min(420px,100%);background:linear-gradient(180deg,var(--panel-raised),var(--panel));border:1px solid var(--line);border-radius:3px;padding:1.4rem;box-shadow:0 30px 80px rgba(0,0,0,.45)}
.login-panel form{display:grid;gap:.75rem;margin-top:1rem}
.login-panel label{font-weight:700;font-size:.85rem}
.login-panel input{width:100%;border:1px solid var(--line);background:var(--void);color:var(--text);border-radius:3px;padding:.8rem;font-family:var(--mono)}
.error{color:var(--crit);margin-top:.75rem;font-size:.85rem}

@media (max-width:820px){
  .hero,.panel-grid{grid-template-columns:1fr}
  .status-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .topbar{position:static;align-items:flex-start}
  .row{flex-wrap:wrap}
  .inspector{width:100%}
}
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
    <div class="brand"><div class="brand-mark">M</div><div><p class="eyebrow">Local control plane</p><h1>Missy Operator Console</h1></div></div>
    <button id="logout" type="button">Sign out</button>
  </header>
  <main class="console-shell" data-csrf="{csrf}">
    <section class="hero">
      <div class="hero-status">
        <p class="eyebrow">Runtime posture</p>
        <h2 id="runtime-status">Loading status...</h2>
        <p id="runtime-summary" class="muted">Checking providers, tools, sessions, and memory.</p>
      </div>
      <div class="status-grid" aria-label="Runtime metrics">
        <article><div class="tile-head"><span class="led info" aria-hidden="true"></span><p>Providers</p></div><span id="provider-count" class="value">-</span></article>
        <article><div class="tile-head"><span class="led info" aria-hidden="true"></span><p>Tools</p></div><span id="tool-count" class="value">-</span></article>
        <article><div class="tile-head"><span class="led info" aria-hidden="true"></span><p>Sessions</p></div><span id="session-count" class="value">-</span></article>
        <article><div class="tile-head"><span class="led info" aria-hidden="true"></span><p>Memory</p></div><span id="memory-state" class="value">-</span></article>
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
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">PRV&middot;01</span><h3>Providers</h3></div><span id="provider-health" class="pill">Loading</span></div><div id="providers" class="list list-scroll"></div></article>
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">TL&middot;02</span><h3>Tools</h3></div><span id="tool-health" class="pill">Loading</span></div><div id="tools" class="list list-scroll"></div></article>
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">SES&middot;03</span><h3>Sessions</h3></div><span class="pill">Recent</span></div><div id="sessions" class="list list-scroll"></div></article>
      <article class="panel diagnostics-panel span-2"><div class="panel-head"><div class="panel-id"><span class="mod-code">DIA&middot;04</span><h3>Diagnostics</h3></div><span id="diagnostics-health" class="pill">Loading</span></div><div id="diagnostics" class="list"></div></article>
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">CTL&middot;05</span><h3>Controls</h3></div><span id="controls-health" class="pill">Loading</span></div><div id="controls" class="list"></div></article>
      <article class="panel scheduler-panel span-2"><div class="panel-head"><div class="panel-id"><span class="mod-code">SCH&middot;06</span><h3>Scheduled Jobs</h3></div><span id="scheduler-health" class="pill">Loading</span></div>
        <div id="scheduler-jobs" class="list list-scroll"></div>
        <form id="scheduler-form" class="op-form" aria-label="Create a scheduled job">
          <input id="job-name" type="text" placeholder="Job name" aria-label="Job name" required>
          <input id="job-schedule" type="text" placeholder="Schedule, e.g. daily at 09:00" aria-label="Job schedule" required>
          <textarea id="job-task" placeholder="Task prompt sent to the agent" aria-label="Job task" rows="2" required></textarea>
          <div class="op-form-grid">
            <input id="job-provider" type="text" placeholder="Provider (optional)" aria-label="Job provider">
            <input id="job-active-hours" type="text" placeholder="Active hours HH:MM-HH:MM (optional)" aria-label="Job active hours">
          </div>
          <div class="op-form-actions"><button type="submit">Add job</button></div>
        </form>
      </article>
      <article class="panel memory-panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">MEM&middot;07</span><h3>Memory Browser</h3></div><span id="memory-health" class="pill">Loading</span></div>
        <div class="filter-row" aria-label="Memory filters">
          <input id="memory-query" type="search" placeholder="Search memory" aria-label="Memory search query">
          <input id="memory-session" type="search" placeholder="Session ID (optional)" aria-label="Memory session filter">
        </div>
        <div id="memory-results" class="list list-scroll"><div class="empty">Search memory to see results.</div></div>
      </article>
      <article class="panel audit-panel span-2"><div class="panel-head"><div class="panel-id"><span class="mod-code">AUD&middot;08</span><h3>Audit Trail</h3></div><span id="audit-health" class="pill">Loading</span></div>
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
        <div id="audit" class="list list-scroll"></div>
        <pre id="audit-detail" class="detail" tabindex="0" aria-label="Selected audit event detail">Select an event to inspect details.</pre>
      </article>
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">SEC&middot;09</span><h3>Security</h3></div><span class="pill secure">Local</span></div><div class="list">
        <div class="row"><div class="row-static"><strong>Authentication</strong></div><span class="meta" title="Cookie session + API key">Cookie session + API key</span></div>
        <div class="row"><div class="row-static"><strong>CSRF</strong></div><span class="meta" title="Required for browser actions">Required for browser actions</span></div>
        <div class="row"><div class="row-static"><strong>Headers</strong></div><span class="meta" title="CSP, no-store, frame deny">CSP, no-store, frame deny</span></div>
        <div class="row"><div class="row-static"><strong>Network</strong></div><span class="meta" title="Loopback by default">Loopback by default</span></div>
      </div></article>
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">APR&middot;10</span><h3>Approvals</h3></div><span id="approvals-health" class="pill">Loading</span></div><div id="approvals" class="list list-scroll"></div></article>
      <article class="panel"><div class="panel-head"><div class="panel-id"><span class="mod-code">PAIR&middot;11</span><h3>Discord Pairing</h3></div><span id="pairing-health" class="pill">Loading</span></div><div id="pairing" class="list list-scroll"></div></article>
    </section>
  </main>
  <div id="inspector-backdrop" class="inspector-backdrop"></div>
  <aside id="inspector" class="inspector" aria-label="Detail inspector" aria-hidden="true">
    <div class="inspector-head">
      <div>
        <span id="inspector-code" class="mod-code">&nbsp;</span>
        <h3 id="inspector-title">Nothing selected</h3>
        <p id="inspector-subtitle"></p>
      </div>
      <button id="inspector-close" type="button" class="secondary" aria-label="Close inspector">Close</button>
    </div>
    <div id="inspector-body" class="inspector-body"><div class="empty">Select an item to inspect its details.</div></div>
  </aside>
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

// ---------------------------------------------------------------------
// Inspector: shared slide-over detail tray for providers/tools/sessions/memory
// ---------------------------------------------------------------------
function inspectorField(label, value) {
  return `<div class="field"><span class="field-label">${esc(label)}</span><span class="field-value">${esc(value)}</span></div>`;
}
function inspectorJson(label, obj) {
  const text = (obj && Object.keys(obj).length) ? JSON.stringify(obj, null, 2) : '(none)';
  return `<div class="field-block"><span class="field-label">${esc(label)}</span><pre class="json-block">${esc(text)}</pre></div>`;
}
function openInspector(code, title, subtitle, bodyHtml) {
  setText('inspector-code', code);
  setText('inspector-title', title || 'Untitled');
  setText('inspector-subtitle', subtitle || '');
  document.getElementById('inspector-body').innerHTML = bodyHtml;
  document.getElementById('inspector').classList.add('open');
  document.getElementById('inspector').setAttribute('aria-hidden', 'false');
  document.getElementById('inspector-backdrop').classList.add('open');
}
function closeInspector() {
  document.getElementById('inspector').classList.remove('open');
  document.getElementById('inspector').setAttribute('aria-hidden', 'true');
  document.getElementById('inspector-backdrop').classList.remove('open');
}
document.getElementById('inspector-close').addEventListener('click', closeInspector);
document.getElementById('inspector-backdrop').addEventListener('click', closeInspector);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeInspector();
});

let auditOffset = 0;
let latestAuditEvents = [];
let latestProviders = [];
let latestTools = [];
let latestMemoryResults = [];
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
  if (event) {
    const d = event.detail || {};
    openInspector('AUD', d.action || event.event_type || 'Audit event', `${event.result} · ${d.severity || 'info'}`, inspectorJson('Event', event));
  }
}
async function loadConsole() {
  try {
    const [status, providers, tools, sessions, diagnostics, controls, audit, jobs, approvals, pairing] = await Promise.all([
      api('/status'), api('/providers'), api('/tools'), api('/sessions?limit=8'), api('/diagnostics'), api('/controls'), api(auditPath()), api('/scheduler/jobs'), api('/approvals'), api('/discord/pairing')
    ]);
    const s = status.data;
    setText('runtime-status', 'Runtime online');
    setText('runtime-summary', `Default provider: ${s.default_provider || 'not configured'}`);
    setText('provider-count', (s.providers_available || []).length);
    setText('tool-count', s.tool_count || 0);
    setText('session-count', s.session_count || 0);
    setText('memory-state', s.memory && s.memory.has_memory ? 'On' : 'Idle');

    latestProviders = providers.data.providers;
    const providerRows = latestProviders.map((p, i) => `<div class="row"><button class="row-title" type="button" data-provider-index="${i}"><span class="led ${p.available ? 'ok' : 'crit'}" aria-hidden="true"></span><strong>${esc(p.name)}</strong></button><span class="meta ${p.available ? 'ok' : 'warn'}">${p.available ? 'available' : 'offline'}${p.is_default ? ' / default' : ''}</span></div>`);
    renderRows('providers', providerRows, 'No providers registered.');
    setText('provider-health', providerRows.length ? `${providerRows.length} ready` : 'Empty');

    latestTools = tools.data.tools;
    const toolRows = latestTools.map((t, i) => `<div class="row row-stack"><button class="row-title" type="button" data-tool-index="${i}"><span class="led info" aria-hidden="true"></span><strong>${esc(t.name)}</strong></button><span class="meta-line" title="${esc(t.description || 'No description')}">${esc(t.description || 'No description')}</span></div>`);
    renderRows('tools', toolRows, 'No tools registered.');
    setText('tool-health', `${latestTools.length} total`);

    const sessionRows = sessions.data.sessions.map(sess => `<div class="row"><button class="row-title" type="button" data-session-id="${esc(sess.session_id)}"><span class="led info" aria-hidden="true"></span><strong>${esc(sess.name || sess.session_id.slice(0, 8))}</strong></button><span class="meta">${esc(sess.provider || 'provider unset')} / ${sess.turn_count} turns</span></div>`);
    renderRows('sessions', sessionRows, 'No API sessions yet.');

    const diagnosticRows = diagnostics.data.sections.map(section => {
      const summary = section.checks.slice(0, 3).map(check => `${check.name}: ${typeof check.summary === 'object' ? JSON.stringify(check.summary) : check.summary}`).join(' / ');
      const remediation = section.checks.find(check => check.remediation)?.remediation || '';
      const statusClass = section.status === 'ok' ? 'ok' : (section.status === 'error' ? 'crit' : 'warn');
      const hint = remediation ? `<em>${esc(remediation)}</em>` : '';
      return `<div class="row"><div class="row-static"><span class="led ${statusClass}" aria-hidden="true"></span> <strong>${esc(section.label)}</strong><em class="muted">${esc(summary)}</em>${hint}</div></div>`;
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
      return `<div class="row"><div class="row-static"><strong>${esc(title)}</strong></div><div class="row-actions"><span class="${target.available ? 'ok' : 'warn'}">${esc(state)}</span><button class="secondary control-action" type="button" data-control-id="${esc(control.id)}" data-control-label="${esc(control.label)}" data-target="${esc(target.name)}" data-target-label="${esc(targetLabel)}" data-confirmation="${esc(target.confirmation)}" ${disabled}>${esc(label)}</button></div></div>`;
    }));
    renderRows('controls', controlRows, 'No safe controls are available.');
    setText('controls-health', controlRows.length ? `${controlRows.length} targets` : 'Empty');

    const jobRows = jobs.data.jobs.map(job => {
      const state = job.enabled ? 'enabled' : 'paused';
      const meta = [job.schedule, job.provider].filter(Boolean).join(' / ');
      return `<div class="row"><div class="row-static"><strong>${esc(job.name || job.id)}</strong></div><div class="row-actions"><span class="${job.enabled ? 'ok' : 'warn'}">${esc(state)} &middot; ${esc(meta)}</span><button class="secondary danger job-remove" type="button" data-job-id="${esc(job.id)}" data-job-name="${esc(job.name || job.id)}">Remove</button></div></div>`;
    });
    renderRows('scheduler-jobs', jobRows, 'No scheduled jobs yet.');
    setText('scheduler-health', jobRows.length ? `${jobRows.length} jobs` : 'Empty');

    const approvalRows = approvals.data.approvals.map(a => `<div class="row row-stack"><div class="row-title"><span class="led warn pulse" aria-hidden="true"></span><strong>${esc(a.action)}</strong></div><span class="meta-line" title="${esc(a.reason || 'No reason given')}">${esc(a.reason || 'No reason given')} &middot; id ${esc(a.id)}</span><div class="row-actions"><button class="secondary approval-action" type="button" data-approval-id="${esc(a.id)}" data-approve="true">Approve</button><button class="secondary danger approval-action" type="button" data-approval-id="${esc(a.id)}" data-approve="false">Deny</button></div></div>`);
    renderRows('approvals', approvalRows, 'No pending approvals.');
    setText('approvals-health', approvalRows.length ? `${approvalRows.length} pending` : 'Empty');

    const pairingRows = pairing.data.pending.map(p => `<div class="row"><div class="row-static"><strong>${esc(p.user_id)}</strong></div><div class="row-actions"><span class="meta">${esc(p.account || 'default account')}</span><button class="secondary pairing-action" type="button" data-user-id="${esc(p.user_id)}" data-approve="true">Approve</button><button class="secondary danger pairing-action" type="button" data-user-id="${esc(p.user_id)}" data-approve="false">Deny</button></div></div>`);
    renderRows('pairing', pairingRows, 'No pending Discord pairing requests.');
    setText('pairing-health', pairingRows.length ? `${pairingRows.length} pending` : 'Empty');

    latestAuditEvents = audit.data.events;
    const auditRows = latestAuditEvents.map(e => {
      const d = e.detail || {};
      const resultClass = e.result === 'deny' || e.result === 'error' ? 'warn' : 'ok';
      return `<button class="row audit-row" type="button" data-event-id="${esc(e.id)}"><span><span class="led ${resultClass === 'ok' ? 'ok' : 'warn'}" aria-hidden="true"></span> <strong>${esc(d.subsystem || e.category)} / ${esc(d.action || e.event_type)}</strong></span><span class="${resultClass}">${esc(e.result)} &middot; ${esc(d.severity || 'info')} &middot; ${esc(d.actor || 'system')} &middot; ${esc(e.timestamp || '')}</span></button>`;
    });
    renderRows('audit', auditRows, 'No audit events match these filters.');
    setText('audit-health', `${audit.data.total} total`);
    document.getElementById('audit-prev').disabled = auditOffset <= 0;
    document.getElementById('audit-next').disabled = !audit.data.has_more;
    if (!latestAuditEvents.some(e => e.id === (document.getElementById('audit-detail').dataset.eventId || ''))) {
      const detailEl = document.getElementById('audit-detail');
      detailEl.textContent = latestAuditEvents[0] ? JSON.stringify(latestAuditEvents[0], null, 2) : 'Select an event to inspect details.';
      detailEl.dataset.eventId = latestAuditEvents[0]?.id || '';
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
document.getElementById('providers').addEventListener('click', event => {
  const row = event.target.closest('[data-provider-index]');
  if (!row) return;
  const p = latestProviders[Number(row.dataset.providerIndex)];
  if (!p) return;
  const body = inspectorField('Status', p.available ? 'Available' : 'Offline')
    + inspectorField('Default provider', p.is_default ? 'Yes' : 'No');
  openInspector('PRV', p.name, p.available ? 'Available' : 'Offline', body);
});
document.getElementById('tools').addEventListener('click', event => {
  const row = event.target.closest('[data-tool-index]');
  if (!row) return;
  const t = latestTools[Number(row.dataset.toolIndex)];
  if (!t) return;
  const body = inspectorField('Description', t.description || 'No description')
    + inspectorJson('Parameters schema', t.schema || {});
  openInspector('TL', t.name, 'Registered tool', body);
});
document.getElementById('sessions').addEventListener('click', async event => {
  const row = event.target.closest('[data-session-id]');
  if (!row) return;
  const sessionId = row.dataset.sessionId;
  openInspector('SES', sessionId, 'Loading transcript...', empty('Loading session detail...'));
  try {
    const [detail, history] = await Promise.all([
      api('/sessions/' + encodeURIComponent(sessionId)),
      api('/sessions/' + encodeURIComponent(sessionId) + '/history?limit=20')
    ]);
    const sess = detail.data;
    const turns = history.data.turns || [];
    const fields = inspectorField('Provider', sess.provider || 'unset')
      + inspectorField('Turns', String(sess.turn_count ?? turns.length))
      + inspectorField('Created', sess.created_at || 'unknown');
    const transcript = turns.length
      ? '<div class="transcript">' + turns.map(t => `<div class="transcript-turn"><span class="turn-role">${esc(t.role)}</span><span class="turn-meta">${esc(t.timestamp || '')}</span><p>${esc(t.content)}</p></div>`).join('') + '</div>'
      : empty('No recorded turns for this session.');
    openInspector('SES', sess.name || sessionId.slice(0, 12), sess.provider || 'provider unset', fields + transcript);
  } catch (error) {
    openInspector('SES', sessionId, 'Error', empty('Could not load session: ' + esc(error.message)));
  }
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
document.getElementById('scheduler-form').addEventListener('submit', async event => {
  event.preventDefault();
  const name = document.getElementById('job-name').value.trim();
  const schedule = document.getElementById('job-schedule').value.trim();
  const task = document.getElementById('job-task').value.trim();
  const provider = document.getElementById('job-provider').value.trim();
  const activeHours = document.getElementById('job-active-hours').value.trim();
  if (!name || !schedule || !task) return;
  try {
    await api('/scheduler/jobs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, schedule, task, provider, active_hours: activeHours})
    });
    event.target.reset();
    await loadConsole();
  } catch (error) {
    window.alert('Could not create job: ' + error.message);
  }
});
document.getElementById('scheduler-jobs').addEventListener('click', async event => {
  const button = event.target.closest('.job-remove');
  if (!button || button.disabled) return;
  const jobId = button.dataset.jobId;
  const jobName = button.dataset.jobName || jobId;
  if (!window.confirm(`Remove scheduled job: ${jobName}? This cannot be undone.`)) return;
  button.disabled = true;
  await api('/scheduler/jobs/' + encodeURIComponent(jobId), {
    method: 'DELETE',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({confirm: 'remove-job:' + jobId})
  });
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
function memoryRow(turn, index) {
  const pinned = turn.pinned;
  const meta = [turn.role, turn.provider, turn.timestamp].filter(Boolean).join(' &middot; ');
  const preview = String(turn.content || '').slice(0, 120);
  return `<div class="row"><button class="row-title" type="button" data-memory-index="${index}">${pinned ? '<span class="pin-marker">&#9733;</span> ' : ''}<strong>${esc(preview)}</strong></button><div class="row-actions"><span>${meta}</span><button class="secondary memory-pin" type="button" data-turn-id="${esc(turn.id)}" data-pinned="${pinned ? '1' : '0'}">${pinned ? 'Unpin' : 'Pin'}</button><button class="secondary danger memory-delete" type="button" data-turn-id="${esc(turn.id)}">Delete</button></div></div>`;
}
async function runMemorySearch() {
  const q = document.getElementById('memory-query').value.trim();
  const sessionId = document.getElementById('memory-session').value.trim();
  if (!q) {
    latestMemoryResults = [];
    renderRows('memory-results', [], 'Search memory to see results.');
    setText('memory-health', 'Idle');
    return;
  }
  const params = new URLSearchParams({q, limit: '15'});
  if (sessionId) params.set('session_id', sessionId);
  try {
    const results = await api('/memory/search?' + params.toString());
    latestMemoryResults = results.data.results;
    const rows = latestMemoryResults.map(memoryRow);
    renderRows('memory-results', rows, 'No memory matches this search.');
    setText('memory-health', `${rows.length} results`);
  } catch (error) {
    setText('memory-health', 'Error');
  }
}
let memorySearchTimer = null;
for (const id of ['memory-query', 'memory-session']) {
  document.getElementById(id).addEventListener('input', () => {
    clearTimeout(memorySearchTimer);
    memorySearchTimer = setTimeout(runMemorySearch, 300);
  });
}
document.getElementById('memory-results').addEventListener('click', async event => {
  const pinButton = event.target.closest('.memory-pin');
  const deleteButton = event.target.closest('.memory-delete');
  const titleButton = event.target.closest('[data-memory-index]');
  if (pinButton && !pinButton.disabled) {
    const turnId = pinButton.dataset.turnId;
    const nextPinned = pinButton.dataset.pinned !== '1';
    pinButton.disabled = true;
    await api('/memory/turns/' + encodeURIComponent(turnId) + '/pin', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pinned: nextPinned})
    });
    await runMemorySearch();
    return;
  }
  if (deleteButton && !deleteButton.disabled) {
    const turnId = deleteButton.dataset.turnId;
    if (!window.confirm('Permanently delete this memory entry?')) return;
    deleteButton.disabled = true;
    await api('/memory/turns/' + encodeURIComponent(turnId), {
      method: 'DELETE',
      headers: {'Content-Type': 'application/json'}
    });
    await runMemorySearch();
    return;
  }
  if (titleButton) {
    const turn = latestMemoryResults[Number(titleButton.dataset.memoryIndex)];
    if (!turn) return;
    const body = inspectorField('Role', turn.role || 'unknown')
      + inspectorField('Provider', turn.provider || 'unset')
      + inspectorField('Session', turn.session_id || 'unknown')
      + inspectorField('Timestamp', turn.timestamp || 'unknown')
      + inspectorField('Pinned', turn.pinned ? 'Yes' : 'No')
      + `<div class="field-block"><span class="field-label">Content</span><pre class="json-block">${esc(turn.content || '')}</pre></div>`;
    openInspector('MEM', turn.role ? `${turn.role} turn` : 'Memory turn', turn.session_id || '', body);
  }
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
