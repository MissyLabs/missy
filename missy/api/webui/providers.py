"""Providers page: usage stats, capacity, config, availability, and controls."""

from __future__ import annotations


def content() -> str:
    """Return the providers page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">PRV&middot;01</p>
        <h2>Providers</h2>
        <p class="muted">Registered AI providers: usage stats, remaining rate-limit capacity, and configuration. Toggle a provider out of dispatch, switch the default, set a weight for balancing, or click a name to edit its configuration.</p>
      </div>
      <div class="page-head-actions">
        <button id="providers-refresh" type="button" class="secondary">Refresh</button>
        <span id="provider-health" class="pill">Loading</span>
      </div>
    </section>
    <section class="panel" aria-labelledby="usage-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">PRV&middot;02</span><h3 id="usage-heading">Usage over time</h3></div>
        <div class="usage-metric-toggle" id="usage-metric-toggle">
          <button type="button" class="secondary small active" data-metric="call_count">Calls</button>
          <button type="button" class="secondary small" data-metric="total_tokens">Tokens</button>
          <button type="button" class="secondary small" data-metric="total_cost_usd">Cost</button>
        </div>
        <select id="usage-days" aria-label="Time range">
          <option value="7">7 days</option>
          <option value="14" selected>14 days</option>
          <option value="30">30 days</option>
          <option value="90">90 days</option>
        </select>
      </div>
      <p class="muted">Which providers are actually serving calls, at a glance. Click a legend entry to isolate/hide that provider.</p>
      <div class="usage-chart-wrap"><div id="usage-chart"><div class="empty">Loading usage...</div></div></div>
      <div class="usage-legend" id="usage-legend"></div>
    </section>
    <section class="panel" aria-labelledby="providers-heading">
      <div class="panel-head">
        <div class="panel-id"><span class="mod-code">PRV&middot;01</span><h3 id="providers-heading">Registered providers</h3></div>
        <span id="provider-default" class="pill">-</span>
      </div>
      <p class="muted">Weight governs the provider-preference hierarchy's weighted fallback/balancing pool: when the default provider (or a candidate in the same tier) is unavailable, the next pick is drawn proportionally to weight. Weight has no effect on which provider is the sticky default.</p>
      <div id="providers"><div class="empty">Loading providers...</div></div>
    </section>
"""


def script() -> str:
    """Return the providers page script."""
    return r"""
let latestUsage = null;
let chartMetric = 'call_count';
let chartDays = 14;
const hiddenProviders = new Set();

const PALETTE = ['#4fb3ff', '#3ddc84', '#f5b942', '#ef5350', '#a78bfa', '#f472b6', '#22d3ee', '#fb923c'];
function colorForProvider(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return PALETTE[hash % PALETTE.length];
}
function fmtNum(n) { return Number(n || 0).toLocaleString(); }
function fmtUsd(n) {
  const value = Number(n || 0);
  return '$' + value.toFixed(value < 1 ? 4 : 2);
}
const METRIC_LABEL = {call_count: 'calls', total_tokens: 'tokens', total_cost_usd: 'cost'};
function fmtMetric(metric, value) { return metric === 'total_cost_usd' ? fmtUsd(value) : fmtNum(value); }

// ---------------------------------------------------------------------
// Capacity gauges
// ---------------------------------------------------------------------
function capacityBar(label, capacity, limit, unlimited) {
  if (unlimited) {
    return `<div class="usage-bar-row"><div class="usage-bar-label"><span>${esc(label)}</span><span>unlimited</span></div></div>`;
  }
  if (!limit) return '';
  const pct = Math.max(0, Math.min(100, (capacity / limit) * 100));
  const cls = pct < 15 ? 'crit' : (pct < 40 ? 'warn' : '');
  return `<div class="usage-bar-row">
    <div class="usage-bar-label"><span>${esc(label)}</span><span>${fmtNum(Math.round(capacity))} / ${fmtNum(limit)}</span></div>
    <div class="usage-bar"><div class="usage-bar-fill ${cls}" style="width:${pct.toFixed(1)}%"></div></div>
  </div>`;
}
function rateLimitBars(rl) {
  if (!rl) return '<p class="muted" style="font-size:.78rem;margin-top:.4rem">No rate limiter configured for this provider.</p>';
  return capacityBar('Requests/min', rl.request_capacity, rl.requests_per_minute, rl.request_unlimited)
    + capacityBar('Tokens/min', rl.token_capacity, rl.tokens_per_minute, rl.token_unlimited);
}

// ---------------------------------------------------------------------
// Provider cards
// ---------------------------------------------------------------------
function usageStatChip(label, value) {
  return `<div class="usage-stat"><span class="value">${esc(value)}</span><span class="label">${esc(label)}</span></div>`;
}
function providerCard(p) {
  const statusLed = !p.enabled ? 'crit' : (p.available ? 'ok' : 'warn');
  const statusLabel = !p.enabled ? 'disabled' : (p.available ? 'available' : 'offline');
  const defaultPill = p.is_default ? '<span class="pill ok">default</span>' : '';
  const model = p.model ? `model ${p.model}` : 'model unset';
  const toggleLabel = p.enabled ? 'Disable' : 'Enable';
  const toggleClass = p.enabled ? 'danger' : '';
  const toggleDisabled = p.enabled && p.is_default ? 'disabled title="Switch the default provider before disabling this one"' : '';
  const defaultDisabled = p.is_default || !p.enabled || !p.available ? 'disabled' : '';
  const weight = typeof p.weight === 'number' ? p.weight : 1;
  const unhealthyCount = p.is_multi_account && p.accounts_healthy != null ? (p.account_count - p.accounts_healthy) : 0;
  const healthNote = unhealthyCount > 0 ? ` &middot; <span class="warn">${unhealthyCount} backing off</span>` : '';
  const balancingNote = p.is_multi_account ? `<span class="provider-meta">round_robin &middot; ${p.account_count} accounts${healthNote}</span>` : '';
  const usage = p.usage || {call_count: 0, total_tokens: 0, total_cost_usd: 0};
  const usageStats = usageStatChip('calls', fmtNum(usage.call_count))
    + usageStatChip('tokens', fmtNum(usage.total_tokens))
    + usageStatChip('cost', fmtUsd(usage.total_cost_usd));
  const capacityMini = p.rate_limit ? `<div class="capacity-mini">${rateLimitBars(p.rate_limit)}</div>` : '';
  return `<div class="provider-card">
    <div class="provider-card-row">
      <button class="provider-name" type="button" data-provider-detail="${esc(p.name)}"><span class="led ${statusLed}" aria-hidden="true"></span>${esc(p.name)}</button>
      ${defaultPill}
      <span class="provider-meta">${esc(statusLabel)} &middot; ${esc(model)}</span>
      ${balancingNote}
      <div class="provider-actions">
        <button class="secondary small provider-default" type="button" data-provider="${esc(p.name)}" ${defaultDisabled}>Make default</button>
        <button class="secondary small ${toggleClass} provider-toggle" type="button" data-provider="${esc(p.name)}" data-enable="${p.enabled ? '0' : '1'}" ${toggleDisabled}>${toggleLabel}</button>
        <label class="provider-weight-label">weight
          <input class="provider-weight-input" type="number" min="0" step="0.1" value="${weight}" data-provider="${esc(p.name)}" aria-label="Weight for ${esc(p.name)}">
        </label>
        <button class="secondary small provider-weight-set" type="button" data-provider="${esc(p.name)}">Set weight</button>
      </div>
    </div>
    <div class="usage-mini">${usageStats}${capacityMini}</div>
  </div>`;
}

// ---------------------------------------------------------------------
// Usage-over-time chart (dependency-free inline SVG, stacked per provider)
// ---------------------------------------------------------------------
function renderLegend() {
  const providers = (latestUsage && latestUsage.providers) || [];
  const items = providers.map(p => {
    const off = hiddenProviders.has(p.name) ? ' off' : '';
    return `<button type="button" class="usage-legend-item${off}" data-legend="${esc(p.name)}">
      <span class="usage-legend-swatch" style="background:${colorForProvider(p.name)}"></span>${esc(p.name)}
    </button>`;
  });
  document.getElementById('usage-legend').innerHTML = items.join('');
}
function renderChart() {
  const container = document.getElementById('usage-chart');
  const series = (latestUsage && latestUsage.series) || [];
  const providers = (latestUsage && latestUsage.providers) || [];
  const visibleNames = providers.map(p => p.name).filter(name => !hiddenProviders.has(name));
  if (!series.length || !visibleNames.length) {
    container.innerHTML = '<div class="usage-chart-empty muted">No usage recorded in this window yet.</div>';
    return;
  }
  const dates = [...new Set(series.map(row => row.date))].sort();
  const byDateProvider = {};
  for (const row of series) {
    byDateProvider[row.date] = byDateProvider[row.date] || {};
    byDateProvider[row.date][row.provider] = row;
  }
  const totals = dates.map(date => visibleNames.reduce((sum, name) => {
    const row = (byDateProvider[date] || {})[name];
    return sum + (row ? Number(row[chartMetric] || 0) : 0);
  }, 0));
  const maxTotal = Math.max(1, ...totals);

  const barWidth = 26, gap = 10, padLeft = 44, padBottom = 24, padTop = 10;
  const chartHeight = 200;
  const chartWidth = padLeft + dates.length * (barWidth + gap);
  const labelEvery = Math.max(1, Math.ceil(dates.length / 14));

  let bars = '';
  dates.forEach((date, i) => {
    const x = padLeft + i * (barWidth + gap);
    let y = chartHeight - padBottom;
    visibleNames.forEach(name => {
      const row = (byDateProvider[date] || {})[name];
      const value = row ? Number(row[chartMetric] || 0) : 0;
      if (value <= 0) return;
      const segHeight = (value / maxTotal) * (chartHeight - padTop - padBottom);
      y -= segHeight;
      const label = `${esc(name)}: ${fmtMetric(chartMetric, value)} on ${esc(date)}`;
      bars += `<rect x="${x}" y="${y.toFixed(1)}" width="${barWidth}" height="${segHeight.toFixed(1)}" fill="${colorForProvider(name)}" rx="1"><title>${label}</title></rect>`;
    });
    if (i % labelEvery === 0) {
      bars += `<text x="${x + barWidth / 2}" y="${chartHeight - 6}" text-anchor="middle" font-size="10" fill="var(--muted)" font-family="var(--mono)">${esc(date.slice(5))}</text>`;
    }
  });
  const axisLabel = `<text x="4" y="${padTop + 8}" font-size="10" fill="var(--muted)" font-family="var(--mono)">${fmtMetric(chartMetric, maxTotal)}</text>`;
  container.innerHTML = `<svg viewBox="0 0 ${chartWidth} ${chartHeight}" width="${chartWidth}" height="${chartHeight}" role="img" aria-label="Provider usage over time (${esc(METRIC_LABEL[chartMetric])})">
    <line x1="${padLeft - 6}" y1="${chartHeight - padBottom}" x2="${chartWidth}" y2="${chartHeight - padBottom}" stroke="var(--line)" stroke-width="1"></line>
    ${axisLabel}
    ${bars}
  </svg>`;
}
function refreshChart() { renderLegend(); renderChart(); }

// ---------------------------------------------------------------------
// Load + inspector
// ---------------------------------------------------------------------
async function loadProviders() {
  try {
    const resp = await api(`/providers/usage?days=${chartDays}`);
    latestUsage = resp.data;
    const providers = latestUsage.providers || [];
    const cards = providers.map(providerCard);
    document.getElementById('providers').innerHTML = cards.length ? cards.join('') : empty('No providers registered.');
    const enabledCount = providers.filter(p => p.enabled).length;
    setText('provider-health', `${enabledCount}/${providers.length} enabled`);
    const currentDefault = providers.find(p => p.is_default);
    setText('provider-default', currentDefault ? `default: ${currentDefault.name}` : 'no default');
    refreshChart();
  } catch (error) {
    setText('provider-health', 'Error');
    document.getElementById('providers').innerHTML = empty('Providers unavailable: ' + error.message);
    document.getElementById('usage-chart').innerHTML = empty('Usage unavailable: ' + error.message);
  }
}

const EDITABLE_FIELDS = [
  {key: 'model', label: 'Model', type: 'text'},
  {key: 'fast_model', label: 'Fast model', type: 'text'},
  {key: 'premium_model', label: 'Premium model', type: 'text'},
  {key: 'base_url', label: 'Base URL', type: 'text'},
  {key: 'timeout', label: 'Timeout (s, 0=default)', type: 'number'},
  {key: 'requests_per_minute', label: 'Requests/min (0=unlimited)', type: 'number'},
  {key: 'tokens_per_minute', label: 'Tokens/min (0=unlimited)', type: 'number'},
];
function editFormHtml(name, current) {
  const rows = EDITABLE_FIELDS.map(f => {
    const value = current[f.key] != null ? current[f.key] : '';
    return `<div><label class="field-label" for="pf-${f.key}">${esc(f.label)}</label>
      <input id="pf-${f.key}" type="${f.type}" ${f.type === 'number' ? 'min="0"' : ''} value="${esc(value)}" data-field="${f.key}" data-original="${esc(value)}">
    </div>`;
  }).join('');
  return `<form class="provider-edit-form op-form" data-provider="${esc(name)}">
    <div class="op-form-grid">${rows}</div>
    <div class="op-form-actions"><button type="submit" class="secondary small">Save configuration</button></div>
  </form>`;
}
function accountRowHtml(account) {
  const led = !account.healthy ? 'crit' : (account.client_ready ? 'ok' : 'warn');
  const statusLabel = !account.healthy
    ? `backing off (${account.consecutive_failures} consecutive failures)`
    : (account.client_ready ? 'healthy' : 'healthy · not yet used this run');
  const usage = account.usage || {call_count: 0, total_prompt_tokens: 0, total_completion_tokens: 0, total_cost_usd: 0, last_call: null};
  const weightNote = account.weight != null && account.weight !== 1 ? ` &middot; weight ${account.weight}` : '';
  return `<div class="account-card">
    <div class="account-card-head">
      <span class="led ${led}" aria-hidden="true"></span>
      <strong>${esc(account.name)}</strong>
      <span class="provider-meta">${esc(statusLabel)}${weightNote}</span>
    </div>
    <div class="usage-stats" style="margin:.4rem 0">
      ${usageStatChip('calls', fmtNum(usage.call_count))}
      ${usageStatChip('prompt tok', fmtNum(usage.total_prompt_tokens))}
      ${usageStatChip('completion tok', fmtNum(usage.total_completion_tokens))}
      ${usageStatChip('cost', fmtUsd(usage.total_cost_usd))}
    </div>
    ${rateLimitBars(account.rate_limit)}
    ${usage.last_call ? `<p class="muted" style="font-size:.74rem;margin-top:.35rem">Last call: ${esc(usage.last_call)}</p>` : ''}
  </div>`;
}
function accountsBlockHtml(accounts) {
  if (!accounts || !accounts.length) return '';
  const rows = accounts.map(accountRowHtml).join('');
  return `<div class="field-block"><span class="field-label">Per-account breakdown (${accounts.length} balanced accounts)</span>
    <div class="account-list">${rows}</div>
  </div>`;
}
async function openProviderInspector(name) {
  openInspector('PRV', name, 'Loading configuration...', empty('Loading provider detail...'));
  try {
    const detail = await api('/providers/' + encodeURIComponent(name));
    const p = detail.data;
    const config = p.config || {};
    const keySummary = config.api_key_configured
      ? 'configured'
      : (config.api_keys_count ? `${config.api_keys_count} rotation keys` : 'not configured');
    const usage = p.usage || {call_count: 0, total_prompt_tokens: 0, total_completion_tokens: 0, total_tokens: 0, total_cost_usd: 0, last_call: null};
    const usageBlock = `<div class="field-block"><span class="field-label">Usage (lifetime, all accounts combined)</span>
      <div class="usage-stats" style="margin-top:.35rem">
        ${usageStatChip('calls', fmtNum(usage.call_count))}
        ${usageStatChip('prompt tok', fmtNum(usage.total_prompt_tokens))}
        ${usageStatChip('completion tok', fmtNum(usage.total_completion_tokens))}
        ${usageStatChip('cost', fmtUsd(usage.total_cost_usd))}
      </div>
      ${usage.last_call ? `<p class="muted" style="font-size:.76rem;margin-top:.35rem">Last call: ${esc(usage.last_call)}</p>` : ''}
    </div>`;
    const capacityLabel = p.accounts && p.accounts.length ? 'Rate-limit capacity (live, combined across accounts)' : 'Rate-limit capacity (live)';
    const capacityBlock = `<div class="field-block"><span class="field-label">${esc(capacityLabel)}</span>${rateLimitBars(p.rate_limit)}</div>`;
    const body = inspectorField('Status', !p.enabled ? 'Disabled' : (p.available ? 'Available' : 'Offline'))
      + inspectorField('Default provider', p.is_default ? 'Yes' : 'No')
      + inspectorField('Balancing weight', config.weight != null ? `${config.weight}` : '1')
      + inspectorField('API key', keySummary)
      + (config.api_keys_count > 1 ? inspectorField('Key rotation', config.key_rotation_strategy || 'failover') : '')
      + (config.account_weights && config.account_weights.length ? inspectorField('Account weights', config.account_weights.join(', ')) : '')
      + usageBlock
      + capacityBlock
      + accountsBlockHtml(p.accounts)
      + `<div class="field-block"><span class="field-label">Edit configuration</span>${editFormHtml(p.name, config)}</div>`
      + (p.diagnostics ? inspectorJson('Diagnostics', p.diagnostics) : '');
    openInspector('PRV', p.name, p.available ? 'Available' : 'Offline', body);
  } catch (error) {
    openInspector('PRV', name, 'Error', empty('Could not load provider: ' + esc(error.message)));
  }
}

// ---------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------
document.getElementById('providers').addEventListener('click', async event => {
  const detailButton = event.target.closest('[data-provider-detail]');
  if (detailButton) {
    openProviderInspector(detailButton.dataset.providerDetail);
    return;
  }
  const defaultButton = event.target.closest('.provider-default');
  if (defaultButton && !defaultButton.disabled) {
    const name = defaultButton.dataset.provider;
    if (!window.confirm(`Set default provider: ${name}?`)) return;
    defaultButton.disabled = true;
    try {
      await api('/controls/provider.set_default', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({target: name, confirm: 'set-default:' + name})
      });
    } catch (error) {
      window.alert('Could not set default: ' + error.message);
    }
    await loadProviders();
    return;
  }
  const toggleButton = event.target.closest('.provider-toggle');
  if (toggleButton && !toggleButton.disabled) {
    const name = toggleButton.dataset.provider;
    const enable = toggleButton.dataset.enable === '1';
    const action = enable ? 'enable' : 'disable';
    if (!window.confirm(`${enable ? 'Enable' : 'Disable'} provider ${name}?`)) return;
    toggleButton.disabled = true;
    try {
      await api('/controls/provider.' + action, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({target: name, confirm: action + '-provider:' + name})
      });
    } catch (error) {
      window.alert(`Could not ${action} provider: ` + error.message);
    }
    await loadProviders();
    return;
  }
  const weightButton = event.target.closest('.provider-weight-set');
  if (weightButton && !weightButton.disabled) {
    const name = weightButton.dataset.provider;
    const card = weightButton.closest('.provider-card');
    const input = card ? card.querySelector('.provider-weight-input') : null;
    const value = input ? parseFloat(input.value) : NaN;
    if (!Number.isFinite(value) || value < 0) {
      window.alert('Weight must be a number >= 0.');
      return;
    }
    if (!window.confirm(`Set weight for ${name} to ${value}? A running gateway picks this up via config hot-reload.`)) return;
    weightButton.disabled = true;
    try {
      await api('/controls/provider.set_weight', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({target: name, value: value, confirm: `set-weight:${name}:${value}`})
      });
    } catch (error) {
      window.alert('Could not set weight: ' + error.message);
    }
    await loadProviders();
  }
});
document.getElementById('inspector-body').addEventListener('submit', async event => {
  const form = event.target.closest('.provider-edit-form');
  if (!form) return;
  event.preventDefault();
  const name = form.dataset.provider;
  const inputs = [...form.querySelectorAll('[data-field]')];
  const changes = inputs
    .map(input => ({field: input.dataset.field, value: input.value, original: input.dataset.original}))
    .filter(change => change.value !== change.original);
  if (!changes.length) {
    window.alert('No changes to save.');
    return;
  }
  const summary = changes.map(c => `${c.field}: ${c.original || '(unset)'} -> ${c.value || '(unset)'}`).join('\n');
  if (!window.confirm(`Save configuration changes for ${name}?\n\n${summary}\n\nA running gateway picks this up via config hot-reload.`)) return;
  const submitButton = form.querySelector('button[type="submit"]');
  submitButton.disabled = true;
  try {
    for (const change of changes) {
      await api('/controls/provider.set_field', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          target: name,
          field: change.field,
          value: change.value,
          confirm: `set-field:${name}:${change.field}:${change.value}`
        })
      });
    }
    await openProviderInspector(name);
    await loadProviders();
  } catch (error) {
    window.alert('Could not save configuration: ' + error.message);
    submitButton.disabled = false;
  }
});
document.getElementById('usage-legend').addEventListener('click', event => {
  const item = event.target.closest('[data-legend]');
  if (!item) return;
  const name = item.dataset.legend;
  if (hiddenProviders.has(name)) hiddenProviders.delete(name); else hiddenProviders.add(name);
  refreshChart();
});
document.getElementById('usage-metric-toggle').addEventListener('click', event => {
  const button = event.target.closest('[data-metric]');
  if (!button) return;
  chartMetric = button.dataset.metric;
  [...document.querySelectorAll('#usage-metric-toggle button')].forEach(b => b.classList.toggle('active', b === button));
  renderChart();
});
document.getElementById('usage-days').addEventListener('change', event => {
  chartDays = parseInt(event.target.value, 10) || 14;
  loadProviders();
});
document.getElementById('providers-refresh').addEventListener('click', loadProviders);
loadProviders();
"""
