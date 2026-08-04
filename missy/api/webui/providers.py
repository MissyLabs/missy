"""Providers page: availability, runtime enable/disable, default, weight, config."""

from __future__ import annotations


def content() -> str:
    """Return the providers page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">PRV&middot;01</p>
        <h2>Providers</h2>
        <p class="muted">Registered AI providers. Toggle a provider out of dispatch, switch the default, set a weight for balancing, or click a name for its redacted configuration.</p>
      </div>
      <div class="page-head-actions">
        <button id="providers-refresh" type="button" class="secondary">Refresh</button>
        <span id="provider-health" class="pill">Loading</span>
      </div>
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
let latestProviders = [];

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
  const balancingNote = p.is_multi_account ? `<span class="provider-meta">round_robin &middot; ${p.account_count} accounts</span>` : '';
  return `<div class="provider-card">
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
  </div>`;
}
async function loadProviders() {
  try {
    const providers = await api('/providers');
    latestProviders = providers.data.providers;
    const cards = latestProviders.map(providerCard);
    document.getElementById('providers').innerHTML = cards.length ? cards.join('') : empty('No providers registered.');
    const enabledCount = latestProviders.filter(p => p.enabled).length;
    setText('provider-health', `${enabledCount}/${latestProviders.length} enabled`);
    const currentDefault = latestProviders.find(p => p.is_default);
    setText('provider-default', currentDefault ? `default: ${currentDefault.name}` : 'no default');
  } catch (error) {
    setText('provider-health', 'Error');
    document.getElementById('providers').innerHTML = empty('Providers unavailable: ' + error.message);
  }
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
    const body = inspectorField('Status', !p.enabled ? 'Disabled' : (p.available ? 'Available' : 'Offline'))
      + inspectorField('Default provider', p.is_default ? 'Yes' : 'No')
      + inspectorField('Balancing weight', config.weight != null ? `${config.weight}` : '1')
      + inspectorField('Model', config.model || 'unset')
      + (config.fast_model ? inspectorField('Fast model', config.fast_model) : '')
      + (config.premium_model ? inspectorField('Premium model', config.premium_model) : '')
      + (config.base_url ? inspectorField('Base URL', config.base_url) : '')
      + inspectorField('Timeout', config.timeout != null ? `${config.timeout}s` : 'default')
      + inspectorField('API key', keySummary)
      + (config.api_keys_count > 1 ? inspectorField('Key rotation', config.key_rotation_strategy || 'failover') : '')
      + (config.account_weights && config.account_weights.length ? inspectorField('Account weights', config.account_weights.join(', ')) : '')
      + (config.requests_per_minute != null ? inspectorField('Rate limit', `${config.requests_per_minute} rpm`) : '')
      + (p.diagnostics ? inspectorJson('Diagnostics', p.diagnostics) : '');
    openInspector('PRV', p.name, p.available ? 'Available' : 'Offline', body);
  } catch (error) {
    openInspector('PRV', name, 'Error', empty('Could not load provider: ' + esc(error.message)));
  }
}
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
document.getElementById('providers-refresh').addEventListener('click', loadProviders);
loadProviders();
"""
