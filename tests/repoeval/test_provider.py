import pytest

from missy.repoeval.provider import (
    AdapterRequest,
    AdapterResponse,
    BrokerError,
    ProviderBroker,
    ProviderFailure,
    ProviderRequest,
    ProviderSpec,
    RateLimitError,
)

SECRET = "super-secret-token-value"


class FakeAdapter:
    def __init__(self, response=None, error=None):
        self.response = response or AdapterResponse(
            "answer",
            actual_model="model-v1",
            request_id="req-123",
            input_tokens=3,
            output_tokens=2,
            cost_microusd=5,
        )
        self.error = error
        self.calls = []

    def complete(self, request: AdapterRequest, credential: str):
        self.calls.append((request, credential))
        if self.error:
            raise self.error
        return self.response


def make_broker(adapters=None, *, authorized=True, credential=SECRET):
    spec = ProviderSpec(
        "test",
        "secret://test",
        frozenset({"model-v1", "model-v2"}),
        max_timeout_seconds=10,
        max_output_tokens=100,
        max_cost_microusd=500,
    )
    return ProviderBroker(
        {"test": spec},
        adapters or {"test": FakeAdapter()},
        lambda ref: credential,
        lambda project, key: authorized,
        max_response_chars=40,
    )


def request():
    return ProviderRequest(
        messages=({"role": "user", "content": "Hello"},), settings={"temperature": 0}, tools=()
    )


def run(broker, **kw):
    return broker.execute(
        project_id="p",
        registry_key="test",
        model="model-v1",
        request=request(),
        budget_microusd=100,
        timeout_seconds=3,
        token_cap=10,
        **kw,
    )


def test_execute_resolves_secret_at_adapter_boundary_and_clamps_limits():
    adapter = FakeAdapter()
    result = make_broker({"test": adapter}).execute(
        project_id="p",
        registry_key="test",
        model="model-v1",
        request=request(),
        budget_microusd=100,
        timeout_seconds=100,
        token_cap=1000,
    )
    call, credential = adapter.calls[0]
    assert credential == SECRET
    assert call.timeout_seconds == 10 and call.max_output_tokens == 100
    assert result.content == "answer" and result.request_digest == request().digest
    assert SECRET not in repr(result)


def test_fanout_uses_identical_canonical_request():
    a, b = FakeAdapter(), FakeAdapter()
    broker = make_broker({"test": a})
    broker._registry["other"] = ProviderSpec("other", "secret://other", frozenset({"model-v2"}))
    broker._adapters["other"] = b
    results = broker.execute_fanout(
        project_id="p",
        targets=(("test", "model-v1"), ("other", "model-v2")),
        request=request(),
        budget_microusd=100,
        timeout_seconds=5,
        token_cap=20,
    )
    assert results[0].request_digest == results[1].request_digest == request().digest
    for attr in ("messages", "settings", "tools"):
        assert getattr(a.calls[0][0], attr) == getattr(b.calls[0][0], attr)


@pytest.mark.parametrize(
    "overrides,category",
    [
        ({"authorized": False}, ProviderFailure.AUTHORIZATION),
        ({"registry_key": "missing"}, ProviderFailure.UNKNOWN_PROVIDER),
        ({"model": "not-approved"}, ProviderFailure.MODEL_NOT_ALLOWED),
        ({"budget_microusd": 501}, ProviderFailure.BUDGET),
    ],
)
def test_policy_refusals_precede_provider_call(overrides, category):
    adapter = FakeAdapter()
    authorized = overrides.pop("authorized", True)
    broker = make_broker({"test": adapter}, authorized=authorized)
    args = {
        "project_id": "p",
        "registry_key": "test",
        "model": "model-v1",
        "request": request(),
        "budget_microusd": 100,
        "timeout_seconds": 3,
        "token_cap": 10,
    }
    args.update(overrides)
    with pytest.raises(BrokerError) as err:
        broker.execute(**args)
    assert err.value.category == category
    assert not adapter.calls


def test_timeout_and_provider_errors_do_not_expose_adapter_exception():
    timed = run(make_broker({"test": FakeAdapter(error=TimeoutError(SECRET))}))
    assert timed.error_category == "timeout" and SECRET not in repr(timed)
    failed = run(make_broker({"test": FakeAdapter(error=RuntimeError(SECRET))}))
    assert failed.error_category == "provider_error" and SECRET not in repr(failed)


def test_response_secret_redaction_and_bounded_output():
    result = run(make_broker({"test": FakeAdapter(AdapterResponse(SECRET + "x" * 100))}))
    assert SECRET not in result.content and len(result.content) <= 40
    assert result.content_truncated


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_model", f"model-{SECRET}"),
        ("request_id", f"prefix-{SECRET}-suffix"),
        ("actual_model", "Bearer abcdefghijk"),
        ("request_id", "api_key=abcdefghijk"),
        ("request_id", "token=abcdefghijk"),
        ("actual_model", "password=abcdefghijk"),
        ("request_id", "ghp_abcdefghijklmnopqrstuvwxyz123456"),
        ("request_id", "x" * 256 + SECRET),
    ],
)
def test_untrusted_provider_metadata_never_emits_secrets_or_partial_values(field, value):
    response = AdapterResponse("safe", **{field: value})
    result = run(make_broker({"test": FakeAdapter(response)}))
    assert getattr(result, "provider_request_id" if field == "request_id" else field) is None
    assert SECRET not in repr(result)
    assert value not in repr(result)


def test_safe_provider_identifiers_preserved_and_error_categories_stable():
    result = run(
        make_broker(
            {
                "test": FakeAdapter(
                    AdapterResponse("safe", actual_model="model-v1", request_id="req-123")
                )
            }
        )
    )
    assert result.actual_model == "model-v1" and result.provider_request_id == "req-123"
    assert result.error_category is None
    with pytest.raises(BrokerError) as error:
        make_broker(authorized=False).execute(
            project_id="p",
            registry_key="test",
            model="model-v1",
            request=request(),
            budget_microusd=100,
            timeout_seconds=3,
            token_cap=10,
        )
    assert error.value.category == ProviderFailure.AUTHORIZATION
    assert error.value.category.value == "authorization_denied"
    assert SECRET not in error.value.category.value


@pytest.mark.parametrize("field", ["actual_model", "request_id"])
def test_exact_real_credential_hidden_in_both_provider_identifiers(field):
    credential = "mortalVendorKey9876543210"
    response = AdapterResponse("safe", **{field: f"prefix-{credential}-suffix"})
    result = run(make_broker({"test": FakeAdapter(response)}, credential=credential))
    assert getattr(result, "provider_request_id" if field == "request_id" else field) is None
    assert credential not in repr(result)


@pytest.mark.parametrize(
    "response", [AdapterResponse("x", output_tokens=-1), AdapterResponse("x", output_tokens=11)]
)
def test_bad_usage_and_token_overrun_are_malformed(response):
    assert run(make_broker({"test": FakeAdapter(response)})).error_category == "malformed_response"


def test_refusal_and_rate_limited_provider_signals():
    assert (
        run(make_broker({"test": FakeAdapter(AdapterResponse("", refused=True))})).error_category
        == "refusal"
    )
    limited = run(make_broker({"test": FakeAdapter(error=RateLimitError(SECRET))}))
    assert limited.error_category == "rate_limit" and SECRET not in repr(limited)


def test_canonical_request_rejects_non_json_values():
    bad = ProviderRequest(messages=({"content": object()},))
    with pytest.raises(BrokerError) as err:
        bad.canonical_bytes()
    assert err.value.category == ProviderFailure.MALFORMED
