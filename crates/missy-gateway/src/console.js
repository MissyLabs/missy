'use strict';
(() => {
  let token = '';
  let pollGeneration = 0;
  const element = id => document.getElementById(id);
  const show = (value, notice) => {
    element('notice').textContent = notice;
    element('output').textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    if (value && value.turn_id) element('turn').value = value.turn_id;
  };
  const session = () => encodeURIComponent(element('session').value);
  async function pollTurn(id, activeSession, generation) {
    // Bounded read-only polling. Never replay chat/resume after transport failure.
    for (let attempt = 0; attempt < 720; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      if (!token || generation !== pollGeneration) return;
      try {
        const response = await fetch('/api/v1/turns/' + encodeURIComponent(id) + '?session=' + encodeURIComponent(activeSession), {
          credentials: 'omit', cache: 'no-store', redirect: 'error', headers: {Authorization: 'Bearer ' + token}
        });
        if (generation !== pollGeneration) return;
        if (!response.ok) {
          if (response.status === 429) { await new Promise(resolve => setTimeout(resolve, 60000)); continue; }
          show(await response.json(), 'Polling stopped: HTTP ' + response.status); return;
        }
        const value = await response.json();
        show(value, 'HTTP ' + response.status);
        if (!value.turn || !['admitted', 'running'].includes(value.turn.status)) return;
      } catch (_) { show('', 'Status connection failed. Use Status to retry; the turn was not replayed.'); return; }
    }
    element('notice').textContent = 'Polling limit reached. The turn was not cancelled; use Status to inspect it.';
  }
  async function request(path, method = 'GET', body) {
    if (!token) { show('', 'Unlock with your bearer token first.'); return; }
    try {
      const response = await fetch('/api/v1/' + path, {
        method, credentials: 'omit', cache: 'no-store', redirect: 'error',
        headers: {Authorization: 'Bearer ' + token, ...(body ? {'Content-Type': 'application/json'} : {})},
        body: body ? JSON.stringify(body) : undefined
      });
      const value = await response.json();
      show(value, 'HTTP ' + response.status);
      if (response.status === 202 && value.turn_id && ['admitted', 'running'].includes(value.status)) {
        const generation = ++pollGeneration;
        void pollTurn(value.turn_id, body.session, generation);
      }
    } catch (_) { show('', 'Request failed. Check the service; no message was automatically replayed.'); }
  }
  element('auth').addEventListener('submit', event => {
    event.preventDefault(); pollGeneration++; token = element('token').value; element('token').value = ''; show('', 'Token set in page memory.');
  });
  element('forget').addEventListener('click', () => { token = ''; pollGeneration++; element('token').value = ''; show('', 'Locked'); });
  element('health').addEventListener('click', () => request('health'));
  element('sessions').addEventListener('click', () => request('sessions?session=' + session()));
  element('history').addEventListener('click', () => request('sessions/' + session() + '/history'));
  element('chat').addEventListener('submit', async event => {
    event.preventDefault(); element('send').disabled = true;
    try { await request('chat', 'POST', {session: element('session').value, prompt: element('prompt').value, request_id: crypto.randomUUID()}); }
    finally { element('send').disabled = false; }
  });
  for (const action of ['status', 'cancel', 'resume']) element(action).addEventListener('click', () => {
    const path = 'turns/' + encodeURIComponent(element('turn').value);
    if (action === 'status') request(path + '?session=' + session());
    else request(path + '/' + action, 'POST', {session: element('session').value});
  });
})();
