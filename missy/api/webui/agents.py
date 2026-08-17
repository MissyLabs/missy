"""Per-agent live activity page for primary and generated agents."""

from __future__ import annotations


def content() -> str:
    """Return the agent activity page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">AGT&middot;02</p>
        <h2>Agent Activity</h2>
        <p class="muted">Live, per-agent visibility across Web, CLI, scheduler, Discord, and voice runs.</p>
      </div>
      <div class="page-head-actions">
        <label class="pill"><input type="checkbox" id="agents-follow" checked> Live</label>
        <button id="agents-refresh" type="button" class="secondary">Refresh</button>
        <span id="agents-health" class="pill">Loading</span>
      </div>
    </section>
    <section class="hero agent-summary">
      <div><p class="eyebrow">Active now</p><span id="agents-active" class="agent-metric">-</span></div>
      <div><p class="eyebrow">Visible agents</p><span id="agents-total" class="agent-metric">-</span></div>
      <div><p class="eyebrow">Observed events</p><span id="agents-events" class="agent-metric">-</span></div>
    </section>
    <section class="panel" aria-labelledby="agents-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">AGT&middot;02</span><h3 id="agents-heading">Primary and generated agents</h3></div>
        <span id="agents-source" class="pill">-</span>
      </div>
      <div class="filter-row">
        <select id="agents-status" aria-label="Agent status">
          <option value="">All statuses</option><option value="running">Running</option><option value="complete">Complete</option><option value="error">Error</option>
        </select>
        <input id="agents-query" type="search" placeholder="Filter name, goal, tool, session" aria-label="Filter agents">
      </div>
      <div id="agents-grid" class="agent-grid" aria-live="polite"></div>
    </section>
    """


def script() -> str:
    """Return the live agent activity script."""
    return r"""
let agentTimer = null;
let visibleAgents = [];
function duration(agent) {
  if (!agent.started_at) return '-';
  const end = agent.finished_at ? new Date(agent.finished_at) : new Date();
  const seconds = Math.max(0, Math.round((end - new Date(agent.started_at)) / 1000));
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}
function renderAgents() {
  const query = document.getElementById('agents-query').value.trim().toLowerCase();
  const filtered = visibleAgents.filter(agent => !query || JSON.stringify(agent).toLowerCase().includes(query));
  const rows = filtered.map(agent => {
    const statusClass = agent.status === 'running' ? 'warn pulse' : (agent.status === 'error' ? 'crit' : 'ok');
    const parent = agent.parent_agent_id ? `<span>parent ${esc(agent.parent_agent_id)}</span>` : '<span>root agent</span>';
    const tools = (agent.tools || []).map(tool => `<span class="agent-tool">${esc(tool)}</span>`).join('') || '<span class="muted">No tools used</span>';
    return `<button type="button" class="agent-card" data-agent-key="${esc(agent.session_id + ':' + agent.agent_id)}">
      <div class="agent-card-head"><span class="led ${statusClass}" aria-hidden="true"></span><strong>${esc(agent.name || agent.agent_id)}</strong><span class="pill">${esc(agent.status)}</span></div>
      <p class="agent-goal">${esc(agent.goal || 'Goal not recorded')}</p>
      <div class="agent-action"><span class="muted">Now</span><strong>${esc(agent.current_action || 'Waiting')}</strong></div>
      <div class="agent-tools">${tools}</div>
      <div class="agent-meta"><span>depth ${esc(agent.depth)}</span>${parent}<span>${esc(duration(agent))}</span><span>${esc(agent.session_id)}</span></div>
    </button>`;
  });
  renderRows('agents-grid', rows, 'No agent activity matches these filters.');
}
async function loadAgents() {
  try {
    const status = document.getElementById('agents-status').value;
    const response = await api('/agents/activity?limit=200' + (status ? '&status=' + encodeURIComponent(status) : ''));
    visibleAgents = response.data.agents || [];
    setText('agents-active', response.data.active_count || 0);
    setText('agents-total', response.data.count || 0);
    setText('agents-events', response.data.event_count || 0);
    setText('agents-source', response.data.source || 'audit bus');
    setText('agents-health', response.data.active_count ? `${response.data.active_count} working` : 'Idle');
    renderAgents();
  } catch (error) {
    setText('agents-health', 'Error');
    renderRows('agents-grid', [], 'Agent activity unavailable: ' + error.message);
  }
}
function showAgent(agent) {
  const actions = (agent.actions || []).map(action => `<div class="timeline-item"><span class="${action.result === 'error' || action.result === 'deny' ? 'crit' : 'ok'}">${esc(action.result)}</span><strong>${esc(action.label)}</strong><time>${esc(action.timestamp)}</time></div>`).join('') || empty('No recorded actions.');
  openInspector('AGT', agent.name || agent.agent_id, `${agent.status} · depth ${agent.depth}`, 
    inspectorField('Agent ID', agent.agent_id) + inspectorField('Task ID', agent.task_id) + inspectorField('Session', agent.session_id) + inspectorField('Parent', agent.parent_agent_id || 'root') + inspectorField('Provider', agent.provider || '-') + inspectorField('Current action', agent.current_action || '-') + `<div class="field-block"><span class="field-label">Goal</span><p>${esc(agent.goal || '-')}</p></div><div class="field-block"><span class="field-label">Timeline</span><div class="agent-timeline">${actions}</div></div>`);
}
document.getElementById('agents-grid').addEventListener('click', event => {
  const card = event.target.closest('[data-agent-key]');
  if (!card) return;
  const selected = visibleAgents.find(agent => agent.session_id + ':' + agent.agent_id === card.dataset.agentKey);
  if (selected) showAgent(selected);
});
document.getElementById('agents-refresh').addEventListener('click', loadAgents);
document.getElementById('agents-status').addEventListener('change', loadAgents);
document.getElementById('agents-query').addEventListener('input', renderAgents);
document.getElementById('agents-follow').addEventListener('change', event => {
  clearInterval(agentTimer); agentTimer = event.target.checked ? setInterval(loadAgents, 2000) : null;
});
loadAgents();
agentTimer = setInterval(loadAgents, 2000);
"""
