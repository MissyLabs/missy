"""Sessions page: API session list with transcript inspection and deletion."""

from __future__ import annotations


def content() -> str:
    """Return the sessions page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">SES&middot;03</p>
        <h2>Sessions</h2>
        <p class="muted">Active API sessions with their recorded transcripts. Click a session to read its history.</p>
      </div>
      <div class="page-head-actions">
        <button id="sessions-refresh" type="button" class="secondary">Refresh</button>
        <span id="sessions-health" class="pill">Loading</span>
      </div>
    </section>
    <section class="panel" aria-labelledby="sessions-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">SES&middot;03</span><h3 id="sessions-heading">Active sessions</h3></div>
        <span class="pill">Most recent first</span>
      </div>
      <div id="sessions" class="list list-scroll list-tall"></div>
    </section>
"""


def script() -> str:
    """Return the sessions page script."""
    return r"""
async function loadSessions() {
  try {
    const sessions = await api('/sessions?limit=50');
    const rows = sessions.data.sessions.map(sess => `<div class="row"><button class="row-title" type="button" data-session-id="${esc(sess.session_id)}"><span class="led info" aria-hidden="true"></span><strong>${esc(sess.name || sess.session_id.slice(0, 8))}</strong></button><div class="row-actions"><span class="meta">${esc(sess.provider || 'provider unset')} / ${esc(sess.turn_count)} turns</span><button class="secondary small danger session-delete" type="button" data-session-id="${esc(sess.session_id)}">End</button></div></div>`);
    renderRows('sessions', rows, 'No API sessions yet.');
    setText('sessions-health', rows.length ? `${rows.length} sessions` : 'Empty');
  } catch (error) {
    setText('sessions-health', 'Error');
    renderRows('sessions', [], 'Sessions unavailable: ' + error.message);
  }
}
document.getElementById('sessions').addEventListener('click', async event => {
  const deleteButton = event.target.closest('.session-delete');
  if (deleteButton && !deleteButton.disabled) {
    const sessionId = deleteButton.dataset.sessionId;
    if (!window.confirm(`End session ${sessionId.slice(0, 12)}? Recorded memory is kept.`)) return;
    deleteButton.disabled = true;
    try {
      await api('/sessions/' + encodeURIComponent(sessionId), {method: 'DELETE'});
    } catch (error) {
      window.alert('Could not end session: ' + error.message);
    }
    await loadSessions();
    return;
  }
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
document.getElementById('sessions-refresh').addEventListener('click', loadSessions);
loadSessions();
setInterval(loadSessions, 20000);
"""
