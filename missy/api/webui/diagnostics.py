"""Interactive diagnostics page: expandable sections, status filter, refresh."""

from __future__ import annotations


def content() -> str:
    """Return the diagnostics page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">DIA&middot;04</p>
        <h2>Diagnostics</h2>
        <p class="muted">Operator doctor report across every subsystem. Expand a section for per-check detail; click a check for the raw record.</p>
      </div>
      <div class="page-head-actions">
        <button id="diag-refresh" type="button" class="secondary">Refresh</button>
        <span id="diagnostics-health" class="pill">Loading</span>
      </div>
    </section>
    <section class="panel" aria-labelledby="diag-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">DIA&middot;04</span><h3 id="diag-heading">Subsystem checks</h3></div>
        <span id="diag-counts" class="pill">-</span>
      </div>
      <div class="chip-row" role="group" aria-label="Filter checks by status">
        <button type="button" class="chip active" data-status-filter="">All</button>
        <button type="button" class="chip" data-status-filter="ok">OK</button>
        <button type="button" class="chip" data-status-filter="warn">Warnings</button>
        <button type="button" class="chip" data-status-filter="error">Errors</button>
        <button type="button" class="chip" data-diag-expand="1">Expand all</button>
        <button type="button" class="chip" data-diag-expand="0">Collapse all</button>
      </div>
      <div id="diagnostics"><div class="empty">Loading diagnostics...</div></div>
    </section>
"""


def script() -> str:
    """Return the diagnostics page script."""
    return r"""
let latestDiagnostics = null;
let diagStatusFilter = '';
let diagOpenSections = new Set();

function summarize(value) {
  return typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '');
}
function renderDiagnostics() {
  const container = document.getElementById('diagnostics');
  if (!latestDiagnostics) {
    container.innerHTML = empty('Diagnostics are unavailable.');
    return;
  }
  const sections = latestDiagnostics.sections.map((section, sectionIndex) => {
    const checks = section.checks.filter(check => !diagStatusFilter || check.status === diagStatusFilter);
    if (diagStatusFilter && !checks.length) return '';
    const counts = {ok: 0, warn: 0, error: 0};
    for (const check of section.checks) counts[check.status] = (counts[check.status] || 0) + 1;
    const statusClass = section.status === 'ok' ? 'ok' : (section.status === 'error' ? 'crit' : 'warn');
    const isOpen = diagOpenSections.has(section.key) || Boolean(diagStatusFilter);
    const checkRows = checks.map((check, checkIndex) => {
      const checkClass = check.status === 'ok' ? 'ok' : (check.status === 'error' ? 'crit' : 'warn');
      const remediation = check.remediation ? `<em>${esc(check.remediation)}</em>` : '';
      return `<button class="diag-check" type="button" data-section-index="${sectionIndex}" data-check-name="${esc(check.name)}"><span class="led ${checkClass}" aria-hidden="true"></span><span class="diag-name">${esc(check.name)}</span><span class="diag-summary">${esc(summarize(check.summary))}</span>${remediation}</button>`;
    }).join('');
    return `<div class="diag-section${isOpen ? ' open' : ''}" data-section-key="${esc(section.key)}">
      <button class="diag-head" type="button" data-section-toggle="${esc(section.key)}" aria-expanded="${isOpen}">
        <span class="led ${statusClass}" aria-hidden="true"></span>
        <span>${esc(section.label)}</span>
        <span class="diag-counts">${counts.ok || 0} ok &middot; ${counts.warn || 0} warn &middot; ${counts.error || 0} error</span>
        <span class="diag-caret">${isOpen ? '&#9662;' : '&#9656;'}</span>
      </button>
      <div class="diag-body">${checkRows || empty('No checks match this filter.')}</div>
    </div>`;
  }).filter(Boolean);
  container.innerHTML = sections.length ? sections.join('') : empty('No sections match this filter.');
}
async function loadDiagnostics() {
  setText('diagnostics-health', 'Loading');
  try {
    const diagnostics = await api('/diagnostics');
    latestDiagnostics = diagnostics.data;
    const counts = latestDiagnostics.counts || {};
    setText('diagnostics-health', latestDiagnostics.overall);
    setText('diag-counts', `${counts.ok || 0} ok · ${counts.warn || 0} warn · ${counts.error || 0} error`);
    renderDiagnostics();
  } catch (error) {
    setText('diagnostics-health', 'Error');
    document.getElementById('diagnostics').innerHTML = empty('Diagnostics unavailable: ' + error.message);
  }
}
document.getElementById('diag-refresh').addEventListener('click', loadDiagnostics);
document.querySelector('.chip-row').addEventListener('click', event => {
  const filterChip = event.target.closest('[data-status-filter]');
  if (filterChip) {
    diagStatusFilter = filterChip.dataset.statusFilter;
    for (const chip of document.querySelectorAll('[data-status-filter]')) {
      chip.classList.toggle('active', chip === filterChip);
    }
    renderDiagnostics();
    return;
  }
  const expandChip = event.target.closest('[data-diag-expand]');
  if (expandChip && latestDiagnostics) {
    diagOpenSections = expandChip.dataset.diagExpand === '1'
      ? new Set(latestDiagnostics.sections.map(section => section.key))
      : new Set();
    renderDiagnostics();
  }
});
document.getElementById('diagnostics').addEventListener('click', event => {
  const toggle = event.target.closest('[data-section-toggle]');
  if (toggle) {
    const key = toggle.dataset.sectionToggle;
    if (diagOpenSections.has(key)) diagOpenSections.delete(key);
    else diagOpenSections.add(key);
    renderDiagnostics();
    return;
  }
  const checkButton = event.target.closest('[data-check-name]');
  if (checkButton && latestDiagnostics) {
    const section = latestDiagnostics.sections[Number(checkButton.dataset.sectionIndex)];
    const check = section?.checks.find(c => c.name === checkButton.dataset.checkName);
    if (!check) return;
    const body = inspectorField('Section', section.label)
      + inspectorField('Status', check.status)
      + (check.remediation ? inspectorField('Remediation', check.remediation) : '')
      + inspectorJson('Check record', check);
    openInspector('DIA', check.name, `${section.label} · ${check.status}`, body);
  }
});
loadDiagnostics();
"""
