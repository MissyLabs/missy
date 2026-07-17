"""Scheduler page: job list with pause/resume/remove and a create form."""

from __future__ import annotations


def content() -> str:
    """Return the scheduler page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">SCH&middot;06</p>
        <h2>Scheduled Jobs</h2>
        <p class="muted">Unattended agent runs on a schedule. Jobs default to the read-only safe-chat capability mode.</p>
      </div>
      <div class="page-head-actions">
        <button id="scheduler-refresh" type="button" class="secondary">Refresh</button>
        <span id="scheduler-health" class="pill">Loading</span>
      </div>
    </section>
    <section class="panel" aria-labelledby="scheduler-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">SCH&middot;06</span><h3 id="scheduler-heading">Jobs</h3></div>
        <span id="scheduler-count" class="pill">-</span>
      </div>
      <div id="scheduler-jobs" class="list list-scroll list-tall"></div>
      <form id="scheduler-form" class="op-form" aria-label="Create a scheduled job">
        <input id="job-name" type="text" placeholder="Job name" aria-label="Job name" required>
        <input id="job-schedule" type="text" placeholder="Schedule, e.g. daily at 09:00" aria-label="Job schedule" required>
        <textarea id="job-task" placeholder="Task prompt sent to the agent" aria-label="Job task" rows="2" required></textarea>
        <div class="op-form-grid">
          <input id="job-provider" type="text" placeholder="Provider (optional)" aria-label="Job provider">
          <input id="job-active-hours" type="text" placeholder="Active hours HH:MM-HH:MM (optional)" aria-label="Job active hours">
        </div>
        <div class="op-form-actions"><button type="submit">Add job</button></div>
      </form>
    </section>
"""


def script() -> str:
    """Return the scheduler page script."""
    return r"""
let latestJobs = [];

function jobRow(job, index) {
  const state = job.enabled ? 'enabled' : 'paused';
  const meta = [job.schedule, job.provider, job.capability_mode].filter(Boolean).join(' / ');
  const pauseResume = job.enabled
    ? `<button class="secondary small job-pause" type="button" data-job-id="${esc(job.id)}" data-job-name="${esc(job.name || job.id)}">Pause</button>`
    : `<button class="secondary small job-resume" type="button" data-job-id="${esc(job.id)}" data-job-name="${esc(job.name || job.id)}">Resume</button>`;
  return `<div class="row"><button class="row-title" type="button" data-job-index="${index}"><span class="led ${job.enabled ? 'ok' : 'warn'}" aria-hidden="true"></span><strong>${esc(job.name || job.id)}</strong></button><div class="row-actions"><span class="${job.enabled ? 'ok' : 'warn'}">${esc(state)} &middot; ${esc(meta)}</span>${pauseResume}<button class="secondary small danger job-remove" type="button" data-job-id="${esc(job.id)}" data-job-name="${esc(job.name || job.id)}">Remove</button></div></div>`;
}
async function loadScheduler() {
  try {
    const jobs = await api('/scheduler/jobs');
    latestJobs = jobs.data.jobs;
    renderRows('scheduler-jobs', latestJobs.map(jobRow), 'No scheduled jobs yet.');
    const enabledCount = latestJobs.filter(job => job.enabled).length;
    setText('scheduler-health', latestJobs.length ? `${latestJobs.length} jobs` : 'Empty');
    setText('scheduler-count', `${enabledCount} enabled · ${latestJobs.length - enabledCount} paused`);
  } catch (error) {
    setText('scheduler-health', 'Error');
    renderRows('scheduler-jobs', [], 'Scheduler unavailable: ' + error.message);
  }
}
async function runJobControl(controlId, jobId, confirmation) {
  await api('/controls/' + encodeURIComponent(controlId), {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({target: jobId, confirm: confirmation})
  });
}
document.getElementById('scheduler-jobs').addEventListener('click', async event => {
  const titleButton = event.target.closest('[data-job-index]');
  if (titleButton) {
    const job = latestJobs[Number(titleButton.dataset.jobIndex)];
    if (!job) return;
    openInspector('SCH', job.name || job.id, job.schedule || '', inspectorJson('Job record', job));
    return;
  }
  const pauseButton = event.target.closest('.job-pause');
  const resumeButton = event.target.closest('.job-resume');
  const removeButton = event.target.closest('.job-remove');
  try {
    if (pauseButton && !pauseButton.disabled) {
      const jobId = pauseButton.dataset.jobId;
      if (!window.confirm(`Pause scheduled job: ${pauseButton.dataset.jobName}?`)) return;
      pauseButton.disabled = true;
      await runJobControl('scheduler.pause_job', jobId, 'pause-job:' + jobId);
    } else if (resumeButton && !resumeButton.disabled) {
      const jobId = resumeButton.dataset.jobId;
      if (!window.confirm(`Resume scheduled job: ${resumeButton.dataset.jobName}?`)) return;
      resumeButton.disabled = true;
      await runJobControl('scheduler.resume_job', jobId, 'resume-job:' + jobId);
    } else if (removeButton && !removeButton.disabled) {
      const jobId = removeButton.dataset.jobId;
      if (!window.confirm(`Remove scheduled job: ${removeButton.dataset.jobName}? This cannot be undone.`)) return;
      removeButton.disabled = true;
      await api('/scheduler/jobs/' + encodeURIComponent(jobId), {
        method: 'DELETE',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({confirm: 'remove-job:' + jobId})
      });
    } else {
      return;
    }
  } catch (error) {
    window.alert('Scheduler action failed: ' + error.message);
  }
  await loadScheduler();
});
document.getElementById('scheduler-form').addEventListener('submit', async event => {
  event.preventDefault();
  const name = document.getElementById('job-name').value.trim();
  const schedule = document.getElementById('job-schedule').value.trim();
  const task = document.getElementById('job-task').value.trim();
  const provider = document.getElementById('job-provider').value.trim();
  const activeHours = document.getElementById('job-active-hours').value.trim();
  if (!name || !schedule || !task) return;
  try {
    await api('/scheduler/jobs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, schedule, task, provider, active_hours: activeHours})
    });
    event.target.reset();
    await loadScheduler();
  } catch (error) {
    window.alert('Could not create job: ' + error.message);
  }
});
document.getElementById('scheduler-refresh').addEventListener('click', loadScheduler);
loadScheduler();
"""
