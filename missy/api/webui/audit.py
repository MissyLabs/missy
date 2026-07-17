"""Full-page audit trail: filters, facets, pagination, export."""

from __future__ import annotations


def content() -> str:
    """Return the audit trail page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">AUD&middot;08</p>
        <h2>Audit Trail</h2>
        <p class="muted">Redacted security and activity events. Facet chips reflect the current filter set; click one to drill in.</p>
      </div>
      <div class="page-head-actions">
        <label class="pill" for="audit-autorefresh"><input id="audit-autorefresh" type="checkbox"> Auto-refresh</label>
        <button id="audit-export" type="button" class="secondary">Export page (JSON)</button>
        <span id="audit-health" class="pill">Loading</span>
      </div>
    </section>
    <section class="panel" aria-labelledby="audit-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">AUD&middot;08</span><h3 id="audit-heading">Events</h3></div>
        <span id="audit-source" class="pill">-</span>
      </div>
      <div class="filter-row" aria-label="Audit filters">
        <select id="audit-result" aria-label="Audit result"><option value="">All results</option><option value="deny">Denied</option><option value="allow">Allowed</option><option value="error">Errors</option></select>
        <select id="audit-severity" aria-label="Audit severity"><option value="">All severities</option><option value="critical">Critical</option><option value="warning">Warning</option><option value="info">Info</option></select>
        <select id="audit-subsystem" aria-label="Audit subsystem"><option value="">All subsystems</option><option value="auth">Auth</option><option value="security">Security</option><option value="network">Network</option><option value="tool">Tools</option><option value="provider">Providers</option><option value="memory">Memory</option><option value="scheduler">Scheduler</option><option value="discord">Discord</option></select>
        <input id="audit-actor" type="search" placeholder="Actor" aria-label="Audit actor">
        <input id="audit-source-filter" type="search" placeholder="Source" aria-label="Audit source">
        <input id="audit-query" type="search" placeholder="Search redacted events" aria-label="Audit search">
        <input id="audit-since" type="datetime-local" aria-label="Audit since timestamp">
        <input id="audit-until" type="datetime-local" aria-label="Audit until timestamp">
        <select id="audit-limit" aria-label="Audit page size">
          <option value="25" selected>25 per page</option>
          <option value="50">50 per page</option>
          <option value="100">100 per page</option>
        </select>
        <button id="audit-clear" type="button" class="secondary">Clear filters</button>
      </div>
      <div id="audit-facets" class="chip-row" aria-label="Audit facets"></div>
      <div class="pager" aria-label="Audit pagination">
        <button id="audit-prev" type="button" class="secondary">Previous</button>
        <button id="audit-next" type="button" class="secondary">Next</button>
        <span id="audit-page-status" class="pager-status"></span>
      </div>
      <div id="audit" class="list list-scroll list-tall"></div>
      <pre id="audit-detail" class="detail" tabindex="0" aria-label="Selected audit event detail">Select an event to inspect details.</pre>
    </section>
"""


def script() -> str:
    """Return the audit trail page script."""
    return r"""
let auditOffset = 0;
let latestAuditEvents = [];
let latestAuditPayload = null;
let auditRefreshTimer = null;
let auditFilterTimer = null;

function auditLimit() {
  return Number(document.getElementById('audit-limit').value) || 25;
}
function isoLocalValue(id) {
  const value = document.getElementById(id).value;
  return value ? new Date(value).toISOString() : '';
}
function auditPath() {
  const params = new URLSearchParams({limit: String(auditLimit()), offset: String(auditOffset)});
  for (const [id, key] of [
    ['audit-result', 'result'],
    ['audit-severity', 'severity'],
    ['audit-subsystem', 'subsystem'],
    ['audit-actor', 'actor'],
    ['audit-source-filter', 'source'],
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
function renderAuditFacets(facets) {
  const rows = [];
  const filterTargets = {result: 'audit-result', severity: 'audit-severity', subsystem: 'audit-subsystem'};
  for (const [facet, targetId] of Object.entries(filterTargets)) {
    const counts = (facets || {})[facet] || {};
    const active = document.getElementById(targetId).value;
    for (const [value, count] of Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
      const isActive = active === value ? ' active' : '';
      rows.push(`<button type="button" class="chip${isActive}" data-facet-target="${esc(targetId)}" data-facet-value="${esc(value)}">${esc(facet)}: ${esc(value)} (${count})</button>`);
    }
  }
  document.getElementById('audit-facets').innerHTML = rows.join('');
}
function renderAuditDetail(event) {
  const detail = document.getElementById('audit-detail');
  detail.textContent = event ? JSON.stringify(event, null, 2) : 'Select an event to inspect details.';
  if (event) {
    const d = event.detail || {};
    openInspector('AUD', d.action || event.event_type || 'Audit event', `${event.result} · ${d.severity || 'info'}`, inspectorJson('Event', event));
  }
}
async function loadAudit() {
  try {
    const audit = await api(auditPath());
    latestAuditPayload = audit.data;
    latestAuditEvents = audit.data.events;
    const auditRows = latestAuditEvents.map(e => {
      const d = e.detail || {};
      const resultClass = e.result === 'deny' || e.result === 'error' ? 'warn' : 'ok';
      return `<button class="row audit-row" type="button" data-event-id="${esc(e.id)}"><span><span class="led ${resultClass === 'ok' ? 'ok' : 'warn'}" aria-hidden="true"></span> <strong>${esc(d.subsystem || e.category)} / ${esc(d.action || e.event_type)}</strong></span><span class="${resultClass}">${esc(e.result)} &middot; ${esc(d.severity || 'info')} &middot; ${esc(d.actor || 'system')} &middot; ${esc(e.timestamp || '')}</span></button>`;
    });
    renderRows('audit', auditRows, 'No audit events match these filters.');
    renderAuditFacets(audit.data.facets);
    setText('audit-health', `${audit.data.total} matching`);
    setText('audit-source', audit.data.source === 'file' ? 'Signed log file' : 'In-memory bus');
    const from = audit.data.total ? auditOffset + 1 : 0;
    const to = auditOffset + latestAuditEvents.length;
    setText('audit-page-status', `${from}-${to} of ${audit.data.total}`);
    document.getElementById('audit-prev').disabled = auditOffset <= 0;
    document.getElementById('audit-next').disabled = !audit.data.has_more;
    if (!latestAuditEvents.some(e => e.id === (document.getElementById('audit-detail').dataset.eventId || ''))) {
      const detailEl = document.getElementById('audit-detail');
      detailEl.textContent = latestAuditEvents[0] ? JSON.stringify(latestAuditEvents[0], null, 2) : 'Select an event to inspect details.';
      detailEl.dataset.eventId = latestAuditEvents[0]?.id || '';
    }
  } catch (error) {
    setText('audit-health', 'Error');
    renderRows('audit', [], 'Audit unavailable: ' + error.message);
  }
}
function scheduleAuditLoad() {
  clearTimeout(auditFilterTimer);
  auditFilterTimer = setTimeout(() => { auditOffset = 0; loadAudit(); }, 250);
}
for (const id of ['audit-result', 'audit-severity', 'audit-subsystem', 'audit-limit']) {
  document.getElementById(id).addEventListener('change', () => { auditOffset = 0; loadAudit(); });
}
for (const id of ['audit-actor', 'audit-source-filter', 'audit-query', 'audit-since', 'audit-until']) {
  document.getElementById(id).addEventListener('input', scheduleAuditLoad);
}
document.getElementById('audit-clear').addEventListener('click', () => {
  for (const id of ['audit-result', 'audit-severity', 'audit-subsystem']) document.getElementById(id).value = '';
  for (const id of ['audit-actor', 'audit-source-filter', 'audit-query', 'audit-since', 'audit-until']) document.getElementById(id).value = '';
  auditOffset = 0;
  loadAudit();
});
document.getElementById('audit-facets').addEventListener('click', event => {
  const chip = event.target.closest('[data-facet-target]');
  if (!chip) return;
  const select = document.getElementById(chip.dataset.facetTarget);
  select.value = select.value === chip.dataset.facetValue ? '' : chip.dataset.facetValue;
  auditOffset = 0;
  loadAudit();
});
document.getElementById('audit-prev').addEventListener('click', () => {
  auditOffset = Math.max(0, auditOffset - auditLimit());
  loadAudit();
});
document.getElementById('audit-next').addEventListener('click', () => {
  auditOffset += auditLimit();
  loadAudit();
});
document.getElementById('audit').addEventListener('click', event => {
  const row = event.target.closest('[data-event-id]');
  if (!row) return;
  const selected = latestAuditEvents.find(e => e.id === row.dataset.eventId);
  renderAuditDetail(selected);
  document.getElementById('audit-detail').dataset.eventId = selected?.id || '';
});
document.getElementById('audit-export').addEventListener('click', () => {
  if (!latestAuditPayload) return;
  const blob = new Blob([JSON.stringify(latestAuditPayload, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'missy-audit-page.json';
  link.click();
  URL.revokeObjectURL(url);
});
document.getElementById('audit-autorefresh').addEventListener('change', event => {
  clearInterval(auditRefreshTimer);
  auditRefreshTimer = null;
  if (event.target.checked) auditRefreshTimer = setInterval(loadAudit, 15000);
});
loadAudit();
"""
