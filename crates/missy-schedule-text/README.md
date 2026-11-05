# missy-schedule-text

Validated scheduling data, not a daemon. This crate never creates jobs, opens a
database, executes prompts, or starts background tasks. The automation owner must
persist the spec and atomically dispatch/reschedule occurrences. Parser tests are
not evidence of a functioning runtime scheduler.

API: `ScheduleSpec::parse(expression, timezone)` anchors relative expressions to
`Utc::now()`. Deterministic callers use `parse_at(expression, timezone, now)`.
`next_after(after)` returns a strictly later UTC instant; `next_occurrence_after`
returns `Ok(None)` for an exhausted one-shot. `is_one_shot()` and `timezone()` are
also available. All public methods return typed `ScheduleError` failures.

Accepted case-insensitive English grammar (ASCII, maximum 256 bytes):

* `daily at09:00` or `daily at 09:00`
* `every monday at08:00` (all seven full weekday names)
* `in2 hours` or `in 2 hours`: one-shot, anchored at parse time
* `every5 minutes` or `every 5 minutes`: elapsed interval, first occurrence one
  interval after parse time, maintaining its persisted anchor thereafter
* Five-field cron, e.g. `0 9 * * 1-5`, evaluated by maintained `croner` 4.x.
  Exactly five fields; no aliases, seconds/year fields, or grammar fallback.
  The engine's day-of-month/day-of-week OR semantics apply when both restricted.

Relative units are second(s), minute(s), hour(s), day(s). Values must be positive
and at most 366 days; days in intervals are exactly 86,400 elapsed seconds, not
calendar days. Clocks require two-digit hour/minute, 00:00 through 23:59. Timezone
is an explicit `chrono-tz` IANA identifier, not a guessed abbreviation or machine
local zone. UTC is supported. Timezone input is at most 64 ASCII bytes. Persist
timezone/spec together; never reparse `in` against a later clock after restart.

DST policy for daily, weekly and cron: ambiguous wall times run **only at the
earlier UTC instant**. The later duplicate is never returned, even when querying
between the two instants. Nonexistent wall times are skipped, never shifted to a
different clock time. Daily/weekly jobs move to the next matching day; cron moves
to its next matching wall time. Interval and one-shot jobs are elapsed UTC and
unaffected by DST. Results are strictly after the supplied instant, including
fractional seconds. Search is bounded: calendar 370 days, cron 10,000 candidates
and croner's year cap. Supported timestamps are 1970 through 4998 inclusive.
Unsatisfiable cron dates return an error rather than hanging or falling back.

JSON is a strict outer `{ "timezone": "...", "schedule": { "kind": "...", ... } }`
object. Unknown fields, invalid enum tags, clock fields, intervals, timezones and
timestamps are rejected on deserialization. An already-past one-shot is valid
persisted data but has no future occurrence. Example daily data:
`{"timezone":"America/New_York","schedule":{"kind":"daily","hour":9,"minute":0}}`.
