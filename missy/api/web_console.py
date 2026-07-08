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
.login-body{display:grid;place-items:center;padding:1rem}.login-panel{width:min(440px,100%);background:rgba(18,26,46,.96);border:1px solid var(--line);border-radius:8px;padding:1.25rem;box-shadow:0 20px 60px rgba(0,0,0,.32)}.brand-mark{width:3rem;height:3rem;display:grid;place-items:center;border-radius:8px;background:#1d4ed8;font-weight:900;margin-bottom:1rem}.login-panel form{display:grid;gap:.75rem;margin-top:1rem}.login-panel label{font-weight:700}.login-panel input{width:100%;border:1px solid var(--line);background:#0f172a;color:var(--text);border-radius:8px;padding:.8rem}.error{color:var(--bad);margin-top:.75rem}
@media (max-width:820px){.hero,.panel-grid{grid-template-columns:1fr}.status-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.topbar{position:static;align-items:flex-start}.row{display:grid}.row span{text-align:left}.row-actions{justify-content:flex-start}}
@media (max-width:520px){.filter-row{grid-template-columns:1fr}}
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
