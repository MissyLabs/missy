use chrono::{DateTime, Duration, Utc};
use missy_schedule_text::{MAX_EXPRESSION_BYTES, ScheduleSpec};

fn t(value: &str) -> DateTime<Utc> {
    value.parse().unwrap()
}
fn spec(expression: &str, timezone: &str) -> ScheduleSpec {
    ScheduleSpec::parse_at(expression, timezone, t("2026-01-01T00:00:00Z")).unwrap()
}

#[test]
fn legacy_grammar_and_strict_after() {
    let daily = spec("daily at09:00", "UTC");
    assert_eq!(
        daily.next_after(t("2026-01-01T08:59:59Z")).unwrap(),
        t("2026-01-01T09:00:00Z")
    );
    assert_eq!(
        daily.next_after(t("2026-01-01T09:00:00Z")).unwrap(),
        t("2026-01-02T09:00:00Z")
    );
    let weekly = spec("every monday at08:00", "America/New_York");
    assert_eq!(
        weekly.next_after(t("2026-01-01T00:00:00Z")).unwrap(),
        t("2026-01-05T13:00:00Z")
    );
    assert_eq!(spec(" DAILY at 09:00 ", "UTC"), daily);
    assert_eq!(daily.timezone(), "UTC");
}

#[test]
fn relative_anchor_and_exhausted_one_shot() {
    let now = t("2026-11-05T15:45:00Z");
    let once = ScheduleSpec::parse_at("in2 hours", "America/New_York", now).unwrap();
    assert!(once.is_one_shot());
    assert_eq!(once.next_after(now).unwrap(), now + Duration::hours(2));
    assert!(
        once.next_occurrence_after(now + Duration::hours(2))
            .unwrap()
            .is_none()
    );
    assert!(once.next_after(now + Duration::days(2)).is_err());
    let restored: ScheduleSpec =
        serde_json::from_str(&serde_json::to_string(&once).unwrap()).unwrap();
    assert_eq!(restored, once);
    assert!(
        restored
            .next_occurrence_after(now + Duration::days(3))
            .unwrap()
            .is_none()
    );
}

#[test]
fn intervals_keep_original_anchor_and_subseconds() {
    let now = t("2026-11-05T15:45:00.500Z");
    let interval = ScheduleSpec::parse_at("every5 minutes", "UTC", now).unwrap();
    assert!(!interval.is_one_shot());
    for (after, expected) in [
        ("2026-11-05T15:45:00.500Z", "2026-11-05T15:50:00.500Z"),
        ("2026-11-05T15:50:00.499Z", "2026-11-05T15:50:00.500Z"),
        ("2026-11-05T15:50:00.500Z", "2026-11-05T15:55:00.500Z"),
        ("2026-11-05T15:55:00.499Z", "2026-11-05T15:55:00.500Z"),
        ("2026-11-05T16:06:00Z", "2026-11-05T16:10:00.500Z"),
    ] {
        assert_eq!(interval.next_after(t(after)).unwrap(), t(expected));
    }
}

#[test]
fn spring_gap_skipped_not_shifted() {
    for expression in ["daily at02:30", "30 2 * * *"] {
        assert_eq!(
            spec(expression, "America/New_York")
                .next_after(t("2026-03-08T06:59:59Z"))
                .unwrap(),
            t("2026-03-09T06:30:00Z")
        );
    }
    assert_eq!(
        spec("every sunday at02:30", "America/New_York")
            .next_after(t("2026-03-08T06:59:59Z"))
            .unwrap(),
        t("2026-03-15T06:30:00Z")
    );
    assert_eq!(
        spec("*/15 * * * *", "America/New_York")
            .next_after(t("2026-03-08T06:59:59Z"))
            .unwrap(),
        t("2026-03-08T07:00:00Z")
    );
}

#[test]
fn fall_fold_first_occurrence_only() {
    for expression in ["daily at01:30", "30 1 * * *"] {
        let s = spec(expression, "America/New_York");
        assert_eq!(
            s.next_after(t("2026-11-01T05:00:00Z")).unwrap(),
            t("2026-11-01T05:30:00Z")
        );
        for after in [
            "2026-11-01T05:30:00Z",
            "2026-11-01T05:45:00Z",
            "2026-11-01T06:00:00Z",
            "2026-11-01T06:29:59Z",
        ] {
            assert_eq!(s.next_after(t(after)).unwrap(), t("2026-11-02T06:30:00Z"));
        }
    }
    assert_eq!(
        spec("every sunday at01:30", "America/New_York")
            .next_after(t("2026-11-01T05:30:00Z"))
            .unwrap(),
        t("2026-11-08T06:30:00Z")
    );
    let frequent = spec("*/15 * * * *", "America/New_York");
    assert_eq!(
        frequent.next_after(t("2026-11-01T05:45:00Z")).unwrap(),
        t("2026-11-01T07:00:00Z")
    );
    assert_eq!(
        frequent.next_after(t("2026-11-01T06:10:00Z")).unwrap(),
        t("2026-11-01T07:00:00Z")
    );
}

#[test]
fn interval_elapsed_across_dst() {
    let s = ScheduleSpec::parse_at("every 1 day", "America/New_York", t("2026-03-07T14:00:00Z"))
        .unwrap();
    assert_eq!(
        s.next_after(t("2026-03-07T14:00:00Z")).unwrap(),
        t("2026-03-08T14:00:00Z")
    );
}

#[test]
fn leap_month_and_day_bounds() {
    assert_eq!(
        spec("0 9 29 2 *", "UTC")
            .next_after(t("2026-03-01T00:00:00Z"))
            .unwrap(),
        t("2028-02-29T09:00:00Z")
    );
    assert_eq!(
        spec("0 0 31 * *", "UTC")
            .next_after(t("2026-04-01T00:00:00Z"))
            .unwrap(),
        t("2026-05-31T00:00:00Z")
    );
    assert_eq!(
        spec("daily at00:00", "UTC")
            .next_after(t("2026-12-31T23:59:59Z"))
            .unwrap(),
        t("2027-01-01T00:00:00Z")
    );
    assert!(
        spec("0 0 30 2 *", "UTC")
            .next_after(t("2026-01-01T00:00:00Z"))
            .is_err()
    );
}

#[test]
fn rejects_invalid_grammar_without_fallback() {
    for expression in [
        "",
        "every0 minutes",
        "in 0 seconds",
        "every 367 days",
        "every 9999999999 days",
        "in -1 hours",
        "in 1.5 hours",
        "daily at24:00",
        "daily at09:60",
        "daily at9:00",
        "every someday at08:00",
        "tomorrow at 09:00",
        "@daily",
        "0 0 * *",
        "0 0 0 * * *",
        "99 * * * *",
        "* * * * 9",
        "0 0 * 13 *",
        "0 0 * * * garbage",
        "daily at09:00 trailing",
        "every5minutes",
        "0 0 L * *",
        "0 0 * * MON#2",
    ] {
        assert!(
            ScheduleSpec::parse_at(expression, "UTC", t("2026-01-01T00:00:00Z")).is_err(),
            "accepted {expression:?}"
        );
    }
    assert!(ScheduleSpec::parse(&" ".repeat(MAX_EXPRESSION_BYTES + 1), "UTC").is_err());
    assert!(ScheduleSpec::parse("daily at09:00", "not/a/timezone").is_err());
    assert!(ScheduleSpec::parse("daily at09:00", "UTC+5").is_err());
    assert!(ScheduleSpec::parse_at("daily at09:00", "UTC", t("1969-01-01T00:00:00Z")).is_err());
}

#[test]
fn strict_validating_serde() {
    for json in [
        r#"{"timezone":"UTC","schedule":{"kind":"daily","hour":9,"minute":0},"extra":true}"#,
        r#"{"timezone":"UTC","schedule":{"kind":"daily","hour":9,"minute":0,"extra":true}}"#,
        r#"{"timezone":"UTC","schedule":{"kind":"daily","hour":24,"minute":0}}"#,
        r#"{"timezone":"UTC","schedule":{"kind":"weekly","weekday":7,"hour":9,"minute":0}}"#,
        r#"{"timezone":"UTC","schedule":{"kind":"interval","seconds":0,"anchor":"2026-01-01T00:00:00Z"}}"#,
        r#"{"timezone":"UTC","schedule":{"kind":"interval","seconds":31622401,"anchor":"2026-01-01T00:00:00Z"}}"#,
        r#"{"timezone":"bad","schedule":{"kind":"once","at":"2026-01-01T00:00:00Z"}}"#,
        r#"{"timezone":"UTC","schedule":{"kind":"cron","expression":"@daily"}}"#,
        r#"{"timezone":"UTC","schedule":{"kind":"cron","expression":"99 * * * *"}}"#,
        r#"{"timezone":"UTC","schedule":{"kind":"once","at":"1900-01-01T00:00:00Z"}}"#,
        r#"{"timezone":"UTC","schedule":{"kind":"made_up"}}"#,
    ] {
        assert!(
            serde_json::from_str::<ScheduleSpec>(json).is_err(),
            "accepted {json}"
        );
    }
    for expression in [
        "daily at09:00",
        "every monday at08:00",
        "in2 hours",
        "every5 minutes",
        "0 9 * * MON-FRI",
    ] {
        let s = spec(expression, "America/New_York");
        assert_eq!(
            serde_json::from_str::<ScheduleSpec>(&serde_json::to_string(&s).unwrap()).unwrap(),
            s
        );
    }
}
