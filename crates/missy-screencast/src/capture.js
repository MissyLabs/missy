'use strict';
(() => {
  const input = document.getElementById('pairing');
  const start = document.getElementById('start');
  const stop = document.getElementById('stop');
  const status = document.getElementById('status');
  const video = document.getElementById('preview');
  const canvas = document.getElementById('frame');
  let stream = null, socket = null, timer = null, waiting = false, busy = false, generation = 0;
  function finish(message = 'Stopped. Create a new pairing to share again.') {
    generation += 1;
    clearTimeout(timer); timer = null;
    const old = socket; socket = null; if (old) { old.onclose = null; old.close(); }
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null; video.srcObject = null; waiting = false; busy = false;
    canvas.width = 0; canvas.height = 0; input.value = '';
    start.disabled = false; stop.disabled = true; status.textContent = message;
  }
  async function frame() {
    if (!socket || socket.readyState !== WebSocket.OPEN || !stream || waiting) return;
    if (socket.bufferedAmount > 262144) { timer = setTimeout(frame, 2000); return; }
    const scale = Math.min(1, 1600 / Math.max(video.videoWidth, video.videoHeight));
    if (!Number.isFinite(scale) || !video.videoWidth) { timer = setTimeout(frame, 2000); return; }
    canvas.width = Math.max(1, Math.floor(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.floor(video.videoHeight * scale));
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
    if (!blob || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (blob.size > 2097152) { finish('Frame exceeds policy. Stopped.'); return; }
    waiting = true; socket.send(blob);
    timer = setTimeout(() => finish('Server acknowledgement timed out. Stopped.'), 180000);
  }
  start.addEventListener('click', async () => {
    if (busy) return; busy = true; start.disabled = true; stop.disabled = false;
    const current = ++generation;
    try {
      const raw = input.value; input.value = '';
      let auth = JSON.parse(raw);
      if (!auth.session_id || !auth.token || !auth.scope || Object.keys(auth).some(k => !['session_id','token','scope'].includes(k))) throw new Error('pairing');
      // This is the only getDisplayMedia call. It occurs in a real user-click
      // handler. No camera/microphone, persisted grants, ambient tabs, or auto-start.
      const pending = navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      status.textContent = 'Choose the surface you consent to share in the browser picker.';
      const granted = await pending;
      if (!busy || current !== generation) { granted.getTracks().forEach(track => track.stop()); return; }
      stream = granted; video.srcObject = stream;
      stream.getVideoTracks().forEach(track => track.addEventListener('ended', () => finish()));
      await video.play();
      if (current !== generation) return;
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
      socket.onopen = () => { socket.send(JSON.stringify(auth)); auth = null; };
      socket.onmessage = event => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'authenticated') { status.textContent = 'Sharing the surface you selected.'; frame(); }
          else if (data.type === 'frame') { clearTimeout(timer); waiting = false; status.textContent = `Sharing: ${data.status}`; timer = setTimeout(frame, 2500); }
        } catch (_) { finish('Invalid server response. Stopped.'); }
      };
      socket.onclose = () => finish('Disconnected. Create a new pairing to share again.');
      socket.onerror = () => finish('Connection failed. Stopped.');
    } catch (_) { if (current === generation) finish('Pairing or browser permission failed. No capture is running.'); }
  });
  stop.addEventListener('click', () => { if (socket && socket.readyState === WebSocket.OPEN) socket.send('stop'); finish(); });
  window.addEventListener('pagehide', () => finish());
})();
