//! Data-only scheduler expressions. No worker, database, task dispatch, or side effects.
//! See the package README for grammar, boundedness, and DST policy.

use chrono::{DateTime, Datelike, Duration, LocalResult, NaiveDateTime, TimeZone, Utc};
use chrono_tz::Tz;
use croner::{Cron, parser::CronParser};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use thiserror::Error;

pub const MAX_EXPRESSION_BYTES: usize = 256;
pub const MAX_INTERVAL_SECONDS: u64 = 366 * 24 * 60 * 60;
const MAX_CANDIDATES: usize = 10_000;

#[derive(Debug, Error)]
pub enum ScheduleError {
    #[error("invalid schedule expression")]
    InvalidExpression,
    #[error("invalid IANA timezone")]
    InvalidTimezone,
    #[error("schedule time is outside supported years 1970 through 4998")]
    TimeOutOfRange,
    #[error("no future occurrence exists within the bounded search")]
    NoFutureOccurrence,
}
pub type Result<T> = std::result::Result<T, ScheduleError>;

/// Validated, serializable schedule, safe to persist as JSON. All deserialization
/// paths revalidate; callers cannot construct an invalid inner variant.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "RawSpec", into = "RawSpec")]
pub struct ScheduleSpec {
    raw: RawSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RawSpec {
    timezone: String,
    schedule: Schedule,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Schedule {
    Daily {
        hour: u8,
        minute: u8,
    },
    /// Monday = 0, Sunday = 6.
    Weekly {
        weekday: u8,
        hour: u8,
        minute: u8,
    },
    Interval {
        seconds: u64,
        anchor: DateTime<Utc>,
    },
    Once {
        at: DateTime<Utc>,
    },
    Cron {
        expression: String,
    },
}

impl From<ScheduleSpec> for RawSpec {
    fn from(spec: ScheduleSpec) -> Self {
        spec.raw
    }
}

impl TryFrom<RawSpec> for ScheduleSpec {
    type Error = ScheduleError;
    fn try_from(raw: RawSpec) -> Result<Self> {
        timezone(&raw.timezone)?;
        match &raw.schedule {
            Schedule::Daily { hour, minute } => clock(*hour, *minute)?,
            Schedule::Weekly {
                weekday,
                hour,
                minute,
            } => {
                if *weekday > 6 {
                    return Err(ScheduleError::InvalidExpression);
                }
                clock(*hour, *minute)?;
            }
            Schedule::Interval { seconds, anchor } => {
                if *seconds == 0 || *seconds > MAX_INTERVAL_SECONDS {
                    return Err(ScheduleError::InvalidExpression);
                }
                bounded_time(*anchor)?;
            }
            Schedule::Once { at } => bounded_time(*at)?,
            Schedule::Cron { expression } => {
                cron(expression)?;
            }
        }
        Ok(Self { raw })
    }
}

fn bounded_time(time: DateTime<Utc>) -> Result<()> {
    if !(1970..=4998).contains(&time.year()) {
        return Err(ScheduleError::TimeOutOfRange);
    }
    Ok(())
}

fn timezone(name: &str) -> Result<Tz> {
    if name.is_empty() || name.len() > 64 || !name.is_ascii() {
        return Err(ScheduleError::InvalidTimezone);
    }
    name.parse().map_err(|_| ScheduleError::InvalidTimezone)
}

fn clock(hour: u8, minute: u8) -> Result<()> {
    if hour > 23 || minute > 59 {
        return Err(ScheduleError::InvalidExpression);
    }
    Ok(())
}

fn cron(expression: &str) -> Result<Cron> {
    if expression.len() > MAX_EXPRESSION_BYTES
        || !expression.is_ascii()
        || expression.split_whitespace().count() != 5
        || !expression
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c.is_ascii_whitespace() || b"*,-/".contains(&c))
    {
        return Err(ScheduleError::InvalidExpression);
    }
    // Only ordinary month/day names are alphabetic in the supported five-field
    // dialect. Do not accidentally accept an engine-specific `L`/`W` modifier.
    for (index, field) in expression.split_whitespace().enumerate() {
        for name in field
            .split(|c: char| !c.is_ascii_alphabetic())
            .filter(|s| !s.is_empty())
        {
            let name = name.to_ascii_lowercase();
            let accepted = match index {
                3 => [
                    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                    "dec",
                ]
                .contains(&name.as_str()),
                4 => ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].contains(&name.as_str()),
                _ => false,
            };
            if !accepted {
                return Err(ScheduleError::InvalidExpression);
            }
        }
    }
    CronParser::new()
        .parse(expression)
        .map_err(|_| ScheduleError::InvalidExpression)
}

static DAILY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^daily\s+at\s*([0-9]{2}):([0-9]{2})$").unwrap());
static WEEKLY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s*([0-9]{2}):([0-9]{2})$").unwrap()
});
static RELATIVE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(in|every)\s*([0-9]{1,10})\s+(second|minute|hour|day)s?$").unwrap()
});

impl ScheduleSpec {
    /// Relative expressions are anchored to the clock at this call. Prefer
    /// `parse_at` when the caller already has a trusted current timestamp.
    pub fn parse(expression: &str, tz: &str) -> Result<Self> {
        Self::parse_at(expression, tz, Utc::now())
    }

    pub fn parse_at(expression: &str, tz: &str, now: DateTime<Utc>) -> Result<Self> {
        bounded_time(now)?;
        timezone(tz)?;
        if expression.len() > MAX_EXPRESSION_BYTES || !expression.is_ascii() {
            return Err(ScheduleError::InvalidExpression);
        }
        let expression = expression.trim().to_ascii_lowercase();
        let schedule = if let Some(c) = DAILY.captures(&expression) {
            Schedule::Daily {
                hour: c[1].parse().unwrap(),
                minute: c[2].parse().unwrap(),
            }
        } else if let Some(c) = WEEKLY.captures(&expression) {
            let weekday = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
            .iter()
            .position(|d| *d == &c[1])
            .unwrap() as u8;
            Schedule::Weekly {
                weekday,
                hour: c[2].parse().unwrap(),
                minute: c[3].parse().unwrap(),
            }
        } else if let Some(c) = RELATIVE.captures(&expression) {
            let amount: u64 = c[2].parse().map_err(|_| ScheduleError::InvalidExpression)?;
            let scale = match &c[3] {
                "second" => 1,
                "minute" => 60,
                "hour" => 3600,
                "day" => 86400,
                _ => unreachable!(),
            };
            let seconds = amount
                .checked_mul(scale)
                .filter(|n| *n > 0 && *n <= MAX_INTERVAL_SECONDS)
                .ok_or(ScheduleError::InvalidExpression)?;
            let at = now
                .checked_add_signed(Duration::seconds(seconds as i64))
                .ok_or(ScheduleError::TimeOutOfRange)?;
            bounded_time(at)?;
            if &c[1] == "in" {
                Schedule::Once { at }
            } else {
                Schedule::Interval {
                    seconds,
                    anchor: at,
                }
            }
        } else {
            cron(&expression)?;
            Schedule::Cron { expression }
        };
        RawSpec {
            timezone: tz.to_owned(),
            schedule,
        }
        .try_into()
    }

    pub fn timezone(&self) -> &str {
        &self.raw.timezone
    }

    pub fn is_one_shot(&self) -> bool {
        matches!(self.raw.schedule, Schedule::Once { .. })
    }

    /// Strictly after `after`. Exhausted one-shots return an explicit error.
    pub fn next_after(&self, after: DateTime<Utc>) -> Result<DateTime<Utc>> {
        self.next_occurrence_after(after)?
            .ok_or(ScheduleError::NoFutureOccurrence)
    }

    /// Strictly after `after`. An exhausted one-shot returns `Ok(None)`;
    /// invalid/out-of-range dates and unfulfillable recurring schedules error.
    pub fn next_occurrence_after(&self, after: DateTime<Utc>) -> Result<Option<DateTime<Utc>>> {
        bounded_time(after)?;
        let next = match &self.raw.schedule {
            Schedule::Once { at } => return Ok((*at > after).then_some(*at)),
            Schedule::Interval { seconds, anchor } => {
                if after < *anchor {
                    *anchor
                } else {
                    // Whole-second division preserves the anchor's fractional
                    // part and never truncates a candidate into the past.
                    let steps =
                        (after.signed_duration_since(*anchor).num_seconds() as u64 / seconds) + 1;
                    let delta = steps
                        .checked_mul(*seconds)
                        .and_then(|s| i64::try_from(s).ok())
                        .ok_or(ScheduleError::TimeOutOfRange)?;
                    anchor
                        .checked_add_signed(Duration::seconds(delta))
                        .ok_or(ScheduleError::TimeOutOfRange)?
                }
            }
            Schedule::Daily { hour, minute } => self.calendar_next(after, None, *hour, *minute)?,
            Schedule::Weekly {
                weekday,
                hour,
                minute,
            } => self.calendar_next(after, Some(*weekday), *hour, *minute)?,
            Schedule::Cron { expression } => self.cron_next(after, expression)?,
        };
        bounded_time(next)?;
        Ok(Some(next))
    }

    fn calendar_next(
        &self,
        after: DateTime<Utc>,
        weekday: Option<u8>,
        hour: u8,
        minute: u8,
    ) -> Result<DateTime<Utc>> {
        let tz = timezone(&self.raw.timezone)?;
        let mut date = after.with_timezone(&tz).date_naive();
        for _ in 0..370 {
            if weekday.is_none_or(|w| date.weekday().num_days_from_monday() == u32::from(w)) {
                let wall = date
                    .and_hms_opt(u32::from(hour), u32::from(minute), 0)
                    .ok_or(ScheduleError::InvalidExpression)?;
                if let Some(candidate) = first_wall_time(tz, wall).filter(|t| *t > after) {
                    return Ok(candidate);
                }
            }
            date = date.succ_opt().ok_or(ScheduleError::TimeOutOfRange)?;
        }
        Err(ScheduleError::NoFutureOccurrence)
    }

    fn cron_next(&self, after: DateTime<Utc>, expression: &str) -> Result<DateTime<Utc>> {
        let tz = timezone(&self.raw.timezone)?;
        let cron = cron(expression)?;
        // Evaluate in a synthetic UTC wall-clock domain, then explicitly resolve
        // each candidate. This prevents the cron engine's interval-dependent
        // fold/gap behavior from silently overriding our uniform DST policy.
        let mut cursor = after.with_timezone(&tz).naive_local().and_utc();
        for _ in 0..MAX_CANDIDATES {
            cursor = cron
                .find_next_occurrence(&cursor, false)
                .map_err(|_| ScheduleError::NoFutureOccurrence)?;
            bounded_time(cursor)?;
            if let Some(candidate) = first_wall_time(tz, cursor.naive_utc()).filter(|t| *t > after)
            {
                return Ok(candidate);
            }
        }
        Err(ScheduleError::NoFutureOccurrence)
    }
}

fn first_wall_time(tz: Tz, wall: NaiveDateTime) -> Option<DateTime<Utc>> {
    match tz.from_local_datetime(&wall) {
        LocalResult::Single(time) => Some(time.with_timezone(&Utc)),
        LocalResult::Ambiguous(a, b) => Some(a.min(b).with_timezone(&Utc)),
        LocalResult::None => None,
    }
}
