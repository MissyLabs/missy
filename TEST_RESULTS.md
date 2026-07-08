# TEST_RESULTS

- Timestamp: 2026-07-08 13:29:18 EDT

## Focused OpenAI Provider Tests

```text
python3 -m pytest tests/providers/test_openai_provider.py tests/providers/test_openai.py -q
42 passed in 1.02s
```

## Provider Suite

```text
python3 -m pytest tests/providers -q
833 passed in 23.27s
```

## Full Suite

```text
python3 -m pytest -q
20476 passed, 6 skipped, 3 warnings in 402.11s (0:06:42)
```

Warnings were import-time SWIG deprecation warnings unrelated to this OpenAI
provider change.
