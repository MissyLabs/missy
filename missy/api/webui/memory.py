"""Full-page memory browser: search, browse, edit, pin, delete."""

from __future__ import annotations


def content() -> str:
    """Return the memory browser page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">MEM&middot;07</p>
        <h2>Memory Browser</h2>
        <p class="muted">Search or page through stored conversation turns. Click a turn to inspect, edit, pin, or delete it.</p>
      </div>
      <div class="page-head-actions"><span id="memory-health" class="pill">Loading</span></div>
    </section>
    <section class="panel" aria-labelledby="memory-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">MEM&middot;07</span><h3 id="memory-heading">Stored turns</h3></div>
        <span id="memory-mode" class="pill">Browse</span>
      </div>
      <div class="filter-row" aria-label="Memory filters">
        <input id="memory-query" type="search" placeholder="Search memory (full-text)" aria-label="Memory search query">
        <select id="memory-session" aria-label="Memory session filter"><option value="">All sessions</option></select>
        <select id="memory-limit" aria-label="Memory page size">
          <option value="25" selected>25 per page</option>
          <option value="50">50 per page</option>
          <option value="100">100 per page</option>
        </select>
      </div>
      <div class="pager" aria-label="Memory pagination">
        <button id="memory-prev" type="button" class="secondary" disabled>Previous</button>
        <button id="memory-next" type="button" class="secondary" disabled>Next</button>
        <span id="memory-page-status" class="pager-status"></span>
      </div>
      <div id="memory-results" class="list list-scroll list-tall"><div class="empty">Loading memory...</div></div>
    </section>
"""


def script() -> str:
    """Return the memory browser page script."""
    return r"""
let memoryOffset = 0;
let latestMemoryResults = [];
let memorySearchTimer = null;

function memoryLimit() {
  return Number(document.getElementById('memory-limit').value) || 25;
}
function memoryRow(turn, index) {
  const pinned = turn.pinned;
  const edited = turn.edited_at ? 'edited' : '';
  const meta = [turn.role, turn.provider, edited, turn.timestamp].filter(Boolean).map(esc).join(' &middot; ');
  const preview = String(turn.content || '').slice(0, 140);
  return `<div class="row"><button class="row-title" type="button" data-memory-index="${index}">${pinned ? '<span class="pin-marker">&#9733;</span> ' : ''}<strong>${esc(preview)}</strong></button><div class="row-actions"><span>${meta}</span><button class="secondary small memory-pin" type="button" data-turn-id="${esc(turn.id)}" data-pinned="${pinned ? '1' : '0'}">${pinned ? 'Unpin' : 'Pin'}</button><button class="secondary small danger memory-delete" type="button" data-turn-id="${esc(turn.id)}">Delete</button></div></div>`;
}
async function loadMemorySessions() {
  try {
    const sessions = await api('/memory/sessions?limit=100');
    const select = document.getElementById('memory-session');
    const current = select.value;
    const options = ['<option value="">All sessions</option>'];
    for (const sess of sessions.data.sessions) {
      const label = sess.name || String(sess.session_id).slice(0, 12);
      options.push(`<option value="${esc(sess.session_id)}">${esc(label)} (${esc(sess.turn_count)} turns)</option>`);
    }
    select.innerHTML = options.join('');
    select.value = current;
  } catch (ignored) {}
}
async function loadMemory() {
  const q = document.getElementById('memory-query').value.trim();
  const sessionId = document.getElementById('memory-session').value.trim();
  const limit = memoryLimit();
  try {
    let results, total, hasMore;
    if (q) {
      const params = new URLSearchParams({q, limit: String(limit)});
      if (sessionId) params.set('session_id', sessionId);
      const response = await api('/memory/search?' + params.toString());
      results = response.data.results;
      total = results.length;
      hasMore = false;
      setText('memory-mode', 'Search');
      setText('memory-page-status', `${results.length} matches`);
    } else {
      const params = new URLSearchParams({limit: String(limit), offset: String(memoryOffset)});
      if (sessionId) params.set('session_id', sessionId);
      const response = await api('/memory/recent?' + params.toString());
      results = response.data.results;
      total = response.data.total;
      hasMore = response.data.has_more;
      setText('memory-mode', 'Browse');
      const from = total ? memoryOffset + 1 : 0;
      const to = memoryOffset + results.length;
      setText('memory-page-status', `${from}-${to} of ${total}`);
    }
    latestMemoryResults = results;
    renderRows('memory-results', results.map(memoryRow), q ? 'No memory matches this search.' : 'No stored memory yet.');
    setText('memory-health', `${total} turns`);
    document.getElementById('memory-prev').disabled = Boolean(q) || memoryOffset <= 0;
    document.getElementById('memory-next').disabled = Boolean(q) || !hasMore;
  } catch (error) {
    setText('memory-health', 'Error');
    renderRows('memory-results', [], 'Memory unavailable: ' + error.message);
  }
}
function scheduleMemoryLoad() {
  clearTimeout(memorySearchTimer);
  memorySearchTimer = setTimeout(() => { memoryOffset = 0; loadMemory(); }, 300);
}
document.getElementById('memory-query').addEventListener('input', scheduleMemoryLoad);
document.getElementById('memory-session').addEventListener('change', () => { memoryOffset = 0; loadMemory(); });
document.getElementById('memory-limit').addEventListener('change', () => { memoryOffset = 0; loadMemory(); });
document.getElementById('memory-prev').addEventListener('click', () => {
  memoryOffset = Math.max(0, memoryOffset - memoryLimit());
  loadMemory();
});
document.getElementById('memory-next').addEventListener('click', () => {
  memoryOffset += memoryLimit();
  loadMemory();
});

function openMemoryInspector(turn) {
  const fields = inspectorField('Role', turn.role || 'unknown')
    + inspectorField('Provider', turn.provider || 'unset')
    + inspectorField('Session', turn.session_id || 'unknown')
    + inspectorField('Timestamp', turn.timestamp || 'unknown')
    + inspectorField('Pinned', turn.pinned ? 'Yes' : 'No')
    + (turn.edited_at ? inspectorField('Edited', turn.edited_at) : '');
  const contentBlock = `<div class="field-block"><span class="field-label">Content</span><pre id="memory-content-view" class="json-block">${esc(turn.content || '')}</pre><textarea id="memory-content-edit" class="memory-edit-area" aria-label="Edit memory content" hidden></textarea></div>`;
  const actions = `<div class="inspector-actions">
    <button id="memory-edit-start" type="button" class="secondary" data-turn-id="${esc(turn.id)}">Edit content</button>
    <button id="memory-edit-save" type="button" data-turn-id="${esc(turn.id)}" hidden>Save changes</button>
    <button id="memory-edit-cancel" type="button" class="secondary" hidden>Cancel</button>
  </div>`;
  openInspector('MEM', turn.role ? `${turn.role} turn` : 'Memory turn', turn.session_id || '', fields + contentBlock + actions);
  const view = document.getElementById('memory-content-view');
  const edit = document.getElementById('memory-content-edit');
  const startBtn = document.getElementById('memory-edit-start');
  const saveBtn = document.getElementById('memory-edit-save');
  const cancelBtn = document.getElementById('memory-edit-cancel');
  startBtn.addEventListener('click', () => {
    edit.value = turn.content || '';
    view.hidden = true;
    edit.hidden = false;
    startBtn.hidden = true;
    saveBtn.hidden = false;
    cancelBtn.hidden = false;
    edit.focus();
  });
  cancelBtn.addEventListener('click', () => {
    view.hidden = false;
    edit.hidden = true;
    startBtn.hidden = false;
    saveBtn.hidden = true;
    cancelBtn.hidden = true;
  });
  saveBtn.addEventListener('click', async () => {
    const contentText = edit.value;
    if (!contentText.trim()) {
      window.alert('Content cannot be empty. Use Delete to remove a memory.');
      return;
    }
    saveBtn.disabled = true;
    try {
      await api('/memory/turns/' + encodeURIComponent(turn.id), {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({content: contentText})
      });
      closeInspector();
      await loadMemory();
    } catch (error) {
      window.alert('Could not save memory edit: ' + error.message);
      saveBtn.disabled = false;
    }
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
    try {
      await api('/memory/turns/' + encodeURIComponent(turnId) + '/pin', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pinned: nextPinned})
      });
    } catch (error) {
      window.alert('Could not update pin: ' + error.message);
    }
    await loadMemory();
    return;
  }
  if (deleteButton && !deleteButton.disabled) {
    const turnId = deleteButton.dataset.turnId;
    if (!window.confirm('Permanently delete this memory entry?')) return;
    deleteButton.disabled = true;
    try {
      await api('/memory/turns/' + encodeURIComponent(turnId), {
        method: 'DELETE',
        headers: {'Content-Type': 'application/json'}
      });
    } catch (error) {
      window.alert('Could not delete memory: ' + error.message);
    }
    await loadMemory();
    return;
  }
  if (titleButton) {
    const turn = latestMemoryResults[Number(titleButton.dataset.memoryIndex)];
    if (turn) openMemoryInspector(turn);
  }
});
loadMemorySessions();
loadMemory();
"""
