"""Shared layout, stylesheet, and client helpers for the Web TUI.

Visual identity: a rack-console / signal-light aesthetic — square LED
indicators (ok/warn/crit/info) are the one recurring motif tying every
module together, monospace type carries data (ids, counts, JSON), and a
display grotesk carries structure (module codes, headings).
"""

from __future__ import annotations

import html


def console_css() -> str:
    """Return the embedded Web TUI stylesheet shared by every page."""
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
h1{font-size:clamp(1.05rem,2.2vw,1.35rem);font-weight:800;letter-spacing:-.01em}
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
button.secondary,.button-link.secondary{background:var(--panel-raised);border-color:var(--line);color:var(--text)}
button.secondary:hover,.button-link.secondary:hover{background:#1c222a}
button:disabled{opacity:.4;cursor:not-allowed}
button.danger{border-color:#6b2323;background:#3a1414;color:#ffb4b0}
button.danger:hover{background:#4a1818}
button.small{padding:.35rem .65rem;font-size:.78rem}

.topbar{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.75rem clamp(1rem,4vw,2.5rem);border-bottom:1px solid var(--line);background:rgba(8,9,12,.9);backdrop-filter:blur(14px);position:sticky;top:0;z-index:20;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:.65rem}
.brand-mark{width:2.1rem;height:2.1rem;display:grid;place-items:center;border-radius:3px;background:linear-gradient(160deg,#1f56c9,#123a8f);font-weight:900;font-family:var(--mono);font-size:.95rem;border:1px solid #2f6fe0}
.console-nav{display:flex;align-items:center;gap:.15rem;flex-wrap:wrap}
.console-nav a{font-family:var(--mono);font-size:.8rem;font-weight:600;color:var(--muted);text-decoration:none;padding:.45rem .7rem;border-radius:3px;border:1px solid transparent}
.console-nav a:hover{color:var(--text);background:var(--panel-raised)}
.console-nav a[aria-current="page"]{color:var(--text);border-color:var(--line);background:var(--panel-raised);box-shadow:inset 0 -2px 0 var(--info)}
.console-shell{width:100%;margin:0;padding:1.25rem clamp(1rem,2.4vw,2.5rem) 3rem}

.page-head{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;margin-bottom:1rem;flex-wrap:wrap}
.page-head-actions{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}

.hero{display:flex;flex-wrap:wrap;gap:1rem;align-items:stretch;margin-bottom:1rem}
.hero-status{flex:1 1 360px;max-width:680px}
.hero>div,.panel{background:linear-gradient(180deg,var(--panel-raised),var(--panel));border:1px solid var(--line);border-radius:3px;padding:1.1rem;position:relative}
.hero>div::before,.panel::before{content:"";position:absolute;inset:0 0 auto 0;height:1px;background:linear-gradient(90deg,rgba(255,255,255,.08),transparent 60%)}
.status-grid{flex:2 1 480px;display:flex;flex-wrap:wrap;gap:.6rem}
.status-grid article{flex:1 1 160px;background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:.85rem}
.status-grid a.tile-link{display:block;color:inherit;text-decoration:none}
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
.pill.ok{color:var(--ok);border-color:#1d4a34}
.pill.warn{color:var(--warn);border-color:#54401a}
.pill.crit{color:var(--crit);border-color:#5a2422}

.list{display:grid;grid-template-columns:minmax(0,1fr);gap:0}
.list-scroll{max-height:19rem;overflow-y:auto;overflow-x:hidden}
.list-tall{max-height:34rem}
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

.filter-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.5rem;margin-bottom:.6rem}
.filter-row select,.filter-row input,.op-form select{min-width:0;border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:3px;padding:.55rem;font:inherit;font-size:.85rem}
.chip-row{display:flex;flex-wrap:wrap;gap:.35rem;margin-bottom:.6rem}
.chip{border:1px solid var(--line);background:var(--panel);color:var(--muted);border-radius:2px;padding:.25rem .55rem;font-family:var(--mono);font-size:.74rem;cursor:pointer;font-weight:600}
.chip:hover{color:var(--text);background:var(--panel-raised)}
.chip.active{color:var(--info);border-color:#1c4a75;background:rgba(79,179,255,.08)}
.pager{display:flex;align-items:center;gap:.5rem;margin:.15rem 0 .6rem;flex-wrap:wrap}
.pager button{padding:.4rem .7rem;font-size:.8rem}
.pager .pager-status{color:var(--muted);font-family:var(--mono);font-size:.78rem}

.audit-row{width:100%;background:transparent;border:0;border-top:1px solid var(--line-soft);border-radius:0;color:var(--text);padding:.62rem 0;text-align:left;cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:1rem;font:inherit}
.audit-row:first-child{border-top:0}
.audit-row:hover,.audit-row:focus-visible{background:rgba(79,179,255,.06)}
.audit-row span{font-size:.78rem;font-family:var(--mono)}
.detail{max-height:16rem;overflow:auto;margin:.75rem 0 0;border:1px solid var(--line);border-radius:3px;background:var(--void);color:var(--muted);padding:.75rem;white-space:pre-wrap;overflow-wrap:anywhere;font-family:var(--mono);font-size:.78rem}

/* Diagnostics: expandable section cards */
.diag-section{border:1px solid var(--line);border-radius:3px;background:linear-gradient(180deg,var(--panel-raised),var(--panel));margin-bottom:.6rem}
.diag-head{width:100%;display:flex;align-items:center;gap:.6rem;background:transparent;border:0;color:var(--text);padding:.75rem .9rem;font:inherit;font-weight:700;cursor:pointer;text-align:left}
.diag-head:hover{background:rgba(79,179,255,.05)}
.diag-head .diag-counts{margin-left:auto;color:var(--muted);font-family:var(--mono);font-size:.76rem;white-space:nowrap}
.diag-head .diag-caret{color:var(--muted-dim);font-family:var(--mono);flex:none}
.diag-body{border-top:1px solid var(--line-soft);padding:.35rem .9rem .6rem;display:none}
.diag-section.open .diag-body{display:block}
.diag-check{display:flex;align-items:baseline;gap:.6rem;border-top:1px solid var(--line-soft);padding:.5rem 0;width:100%;background:transparent;border-left:0;border-right:0;border-bottom:0;color:var(--text);font:inherit;text-align:left;cursor:pointer}
.diag-check:first-child{border-top:0}
.diag-check:hover{color:var(--info)}
.diag-check .diag-name{font-weight:600;flex:none}
.diag-check .diag-summary{color:var(--muted);font-family:var(--mono);font-size:.78rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.diag-check em{color:var(--warn);font-style:normal;font-size:.76rem;margin-left:auto;flex:none}

/* Providers: card list with toggles */
.provider-card{border-top:1px solid var(--line-soft);padding:.7rem 0;display:flex;align-items:center;gap:.75rem;flex-wrap:wrap}
.provider-card:first-child{border-top:0}
.provider-card .provider-name{min-width:0;text-align:left;background:transparent;border:0;color:var(--text);font:inherit;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:.5rem;padding:0}
.provider-card .provider-name:hover{color:var(--info)}
.provider-card .provider-meta{color:var(--muted);font-family:var(--mono);font-size:.78rem;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.provider-card .provider-actions{margin-left:auto;display:flex;gap:.5rem;flex-wrap:wrap}

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
.inspector{position:fixed;top:0;right:0;bottom:0;width:min(520px,100%);background:linear-gradient(180deg,var(--panel-raised),var(--panel));border-left:1px solid var(--line);box-shadow:-24px 0 60px rgba(0,0,0,.5);transform:translateX(100%);transition:transform .18s ease;z-index:30;display:flex;flex-direction:column}
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
.inspector-actions{display:flex;gap:.5rem;flex-wrap:wrap;padding:.75rem 0 0}
.memory-edit-area{width:100%;min-height:9rem;resize:vertical;border:1px solid var(--line);background:var(--void);color:var(--text);border-radius:3px;padding:.7rem;font-family:var(--mono);font-size:.8rem;margin-top:.4rem}
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


def shared_script() -> str:
    """Return the client helpers shared by every operator page."""
    return r"""
const root = document.querySelector('.console-shell');
const csrf = root.dataset.csrf;
async function api(path, options = {}) {
  const response = await fetch('/api/v1' + path, {
    ...options,
    headers: {'Accept': 'application/json', 'X-CSRF-Token': csrf, ...(options.headers || {})},
    credentials: 'same-origin'
  });
  if (!response.ok) {
    let message = path + ' returned ' + response.status;
    try {
      const body = await response.json();
      if (body && body.error) message = body.error;
    } catch (ignored) {}
    throw new Error(message);
  }
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
// Inspector: shared slide-over detail tray used by every page
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
document.getElementById('logout').addEventListener('click', async () => {
  await fetch('/logout', {method: 'POST', headers: {'X-CSRF-Token': csrf}, credentials: 'same-origin'});
  window.location = '/login';
});
"""


def render_shell(
    *,
    active: str,
    title: str,
    csrf_token: str,
    content: str,
    script: str,
) -> str:
    """Wrap page *content* + *script* in the shared operator layout."""
    from missy.api.webui import PAGES

    csrf = html.escape(csrf_token, quote=True)
    nav_links = []
    for slug, (label, _module) in PAGES.items():
        href = "/" if slug == "dashboard" else f"/{slug}"
        current = ' aria-current="page"' if slug == active else ""
        nav_links.append(f'<a href="{href}"{current}>{html.escape(label)}</a>')
    nav = "".join(nav_links)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Missy Operator Console — {html.escape(title)}</title>
  <style>{console_css()}</style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><div class="brand-mark">M</div><div><p class="eyebrow">Local control plane</p><h1>Missy Operator Console</h1></div></div>
    <nav class="console-nav" aria-label="Console pages">{nav}</nav>
    <button id="logout" type="button" class="secondary">Sign out</button>
  </header>
  <main class="console-shell" data-csrf="{csrf}">
{content}
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
  <script>{shared_script()}
{script}</script>
</body>
</html>"""


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
