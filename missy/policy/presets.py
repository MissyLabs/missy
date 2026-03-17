"""Network policy presets for common services.

Each preset maps to a set of hosts, domains, and CIDRs that should be
allowed for the named service.  ``resolve_presets()`` merges multiple
presets into deduplicated lists suitable for injection into a
:class:`~missy.config.settings.NetworkPolicy`.
"""

from __future__ import annotations

PRESETS: dict[str, dict[str, list[str]]] = {
    "anthropic": {
        "hosts": ["api.anthropic.com"],
        "domains": ["anthropic.com"],
        "cidrs": [],
    },
    "openai": {
        "hosts": ["api.openai.com", "auth.openai.com", "chatgpt.com"],
        "domains": ["openai.com"],
        "cidrs": [],
    },
    "ollama": {
        "hosts": ["localhost:11434", "127.0.0.1:11434"],
        "domains": [],
        "cidrs": ["127.0.0.0/8"],
    },
    "github": {
        "hosts": ["api.github.com", "github.com"],
        "domains": ["github.com", "githubusercontent.com"],
        "cidrs": [],
    },
    "discord": {
        "hosts": ["discord.com", "gateway.discord.gg"],
        "domains": ["discord.com", "discord.gg", "discordapp.com"],
        "cidrs": [],
    },
    "home-assistant": {
        "hosts": ["localhost:8123", "127.0.0.1:8123"],
        "domains": [],
        "cidrs": ["127.0.0.0/8"],
    },
    "pypi": {
        "hosts": ["pypi.org", "files.pythonhosted.org"],
        "domains": ["pypi.org", "pythonhosted.org"],
        "cidrs": [],
    },
    "npm": {
        "hosts": ["registry.npmjs.org"],
        "domains": ["npmjs.org", "npmjs.com"],
        "cidrs": [],
    },
    "docker-hub": {
        "hosts": ["registry-1.docker.io", "auth.docker.io", "index.docker.io"],
        "domains": ["docker.io", "docker.com"],
        "cidrs": [],
    },
    "huggingface": {
        "hosts": ["huggingface.co", "cdn-lfs.huggingface.co"],
        "domains": ["huggingface.co"],
        "cidrs": [],
    },
}


def resolve_presets(
    names: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Resolve a list of preset names into merged network policy entries.

    Args:
        names: Preset names to resolve (e.g. ``["anthropic", "github"]``).

    Returns:
        A 4-tuple of ``(hosts, domains, cidrs, unknown)`` where *unknown*
        contains any names that did not match a built-in preset.
    """
    hosts: list[str] = []
    domains: list[str] = []
    cidrs: list[str] = []
    unknown: list[str] = []

    seen_hosts: set[str] = set()
    seen_domains: set[str] = set()
    seen_cidrs: set[str] = set()

    for name in names:
        preset = PRESETS.get(name)
        if preset is None:
            unknown.append(name)
            continue
        for h in preset.get("hosts", []):
            if h not in seen_hosts:
                hosts.append(h)
                seen_hosts.add(h)
        for d in preset.get("domains", []):
            if d not in seen_domains:
                domains.append(d)
                seen_domains.add(d)
        for c in preset.get("cidrs", []):
            if c not in seen_cidrs:
                cidrs.append(c)
                seen_cidrs.add(c)

    return hosts, domains, cidrs, unknown
