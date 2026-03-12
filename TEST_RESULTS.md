# TEST_RESULTS

- Timestamp: 2026-03-12 (Session 5)

## pytest
```
1097 passed, 1 warning in 5.30s
```

## New Tests Added (Session 5)

### tests/providers/test_rate_limiter.py (18 tests)
- TestRateLimiterInit: default_values, custom_values, unlimited_when_zero
- TestAcquire: succeeds_under_limit, blocks_when_near_limit, raises_on_timeout, with_token_budget, unlimited_requests_with_token_limit, unlimited_tokens_with_request_limit
- TestCapacityProperties: request_capacity_starts_at_max, decreases_after_acquire, token_capacity_starts_at_max, unlimited_capacity_is_inf
- TestOnRateLimitResponse: drains_request_bucket, drains_token_bucket
- TestRecordUsage: does_not_crash, with_no_limit
- TestThreadSafety: concurrent_acquire

### tests/memory/test_sqlite_costs.py (11 tests)
- TestRecordCost: record_and_retrieve, multiple_records, different_sessions_isolated, empty_session_returns_empty, timestamp_is_set
- TestGetTotalCosts: aggregates_by_session, respects_limit, empty_returns_empty, ordered_by_most_recent
- TestCostsTableCreation: costs_table_exists, costs_indexes_exist

### tests/agent/test_runtime_streaming.py (6 tests)
- TestRunStream: yields_chunks, falls_back_on_error, falls_back_to_run_with_tools
- TestRateLimitIntegration: rate_limiter_created, rate_limiter_called_before_completion
- TestCostPersistence: record_cost_passes_session_id

### tests/cli/test_cost_recover.py (9 tests)
- TestCostCommand: shows_budget, unlimited_budget, with_session, with_session_no_data, help_exits_zero
- TestRecoverCommand: no_checkpoints, shows_checkpoints, abandon_all, help_exits_zero
