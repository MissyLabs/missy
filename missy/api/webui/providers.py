"""Providers page: availability, runtime enable/disable, default, config."""

from __future__ import annotations


def content() -> str:
    """Return the providers page body."""
    return """
    <section class="page-head">
      <div>
        <p class="eyebrow">PRV&middot;01</p>
        <h2>Providers</h2>
        <p class="muted">Registered AI providers. Toggle a provider out of dispatch, switch the default, or click a name for its redacted configuration.</p>
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
  return `<div class="provider-card">
    <button class="provider-name" type="button" data-provider-detail="${esc(p.name)}"><span class="led ${statusLed}" aria-hidden="true"></span>${esc(p.name)}</button>
    ${defaultPill}
    <span class="provider-meta">${esc(statusLabel)} &middot; ${esc(model)}</span>
    <div class="provider-actions">
      <button class="secondary small provider-default" type="button" data-provider="${esc(p.name)}" ${defaultDisabled}>Make default</button>
      <button class="secondary small ${toggleClass} provider-toggle" type="button" data-provider="${esc(p.name)}" data-enable="${p.enabled ? '0' : '1'}" ${toggleDisabled}>${toggleLabel}</button>
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
      + inspectorField('Model', config.model || 'unset')
      + (config.fast_model ? inspectorField('Fast model', config.fast_model) : '')
      + (config.premium_model ? inspectorField('Premium model', config.premium_model) : '')
      + (config.base_url ? inspectorField('Base URL', config.base_url) : '')
      + inspectorField('Timeout', config.timeout != null ? `${config.timeout}s` : 'default')
      + inspectorField('API key', keySummary)
      + (config.api_keys_count > 1 ? inspectorField('Key rotation', config.key_rotation_strategy || 'failover') : '')
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
  }
});
document.getElementById('providers-refresh').addEventListener('click', loadProviders);
loadProviders();
"""
