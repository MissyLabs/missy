"""Live log tail page: polls the application log and appends new lines (F18)."""

from __future__ import annotations


def content() -> str:
    """Return the logs page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">LOG&middot;12</p>
        <h2>Live Logs</h2>
        <p class="muted">Tail of the application log, refreshed live. Secret-shaped values are redacted server-side.</p>
      </div>
      <div class="page-head-actions">
        <label class="pill"><input type="checkbox" id="logs-follow" checked> Auto-refresh</label>
        <button id="logs-refresh" type="button" class="secondary">Refresh</button>
        <span id="logs-health" class="pill">Loading</span>
      </div>
    </section>
    <section class="panel" aria-labelledby="logs-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">LOG&middot;12</span><h3 id="logs-heading">Application log</h3></div>
        <span id="logs-count" class="pill">-</span>
      </div>
      <div class="filter-row" aria-label="Log filters">
        <input id="logs-filter" type="search" placeholder="Filter visible lines (client-side)" aria-label="Filter log lines">
        <select id="logs-lines" aria-label="Lines to fetch">
          <option value="200" selected>200 lines</option>
          <option value="500">500 lines</option>
          <option value="1000">1000 lines</option>
        </select>
      </div>
      <pre id="logs-view" class="detail" tabindex="0" aria-live="polite" aria-label="Log output" style="max-height:60vh;overflow:auto">Loading log...</pre>
    </section>
"""


def script() -> str:
    """Return the logs page script."""
    return r"""
let logsTimer = null;
function logsLineCount() { return Number(document.getElementById('logs-lines').value) || 200; }

async function loadLogs() {
  try {
    const resp = await api('/logs/tail?lines=' + logsLineCount());
    const data = resp.data || {};
    const lines = data.lines || [];
    const filter = document.getElementById('logs-filter').value.trim().toLowerCase();
    const shown = filter ? lines.filter(l => l.toLowerCase().includes(filter)) : lines;
    const view = document.getElementById('logs-view');
    const atBottom = view.scrollTop + view.clientHeight >= view.scrollHeight - 8;
    view.textContent = shown.length ? shown.join('\n') : '(no matching log lines)';
    if (atBottom) view.scrollTop = view.scrollHeight;
    setText('logs-count', `${shown.length}/${lines.length} lines`);
    setText('logs-health', data.path ? 'OK' : 'No log');
  } catch (error) {
    setText('logs-health', 'Error');
    document.getElementById('logs-view').textContent = 'Log unavailable: ' + error.message;
  }
}
function scheduleLogsFollow() {
  clearInterval(logsTimer);
  if (document.getElementById('logs-follow').checked) {
    logsTimer = setInterval(loadLogs, 4000);
  }
}
document.getElementById('logs-refresh').addEventListener('click', loadLogs);
document.getElementById('logs-follow').addEventListener('change', scheduleLogsFollow);
document.getElementById('logs-filter').addEventListener('input', loadLogs);
document.getElementById('logs-lines').addEventListener('change', loadLogs);
loadLogs();
scheduleLogsFollow();
"""
