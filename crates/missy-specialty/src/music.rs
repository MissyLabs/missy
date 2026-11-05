use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Waveform {
    #[default]
    Sine,
    Triangle,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Note {
    pub pitch: u8,
    pub start_tick: u32,
    pub duration_ticks: u32,
    pub velocity: u8,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub bpm: u16,
    pub ticks_per_beat: u16,
    pub notes: Vec<Note>,
    #[serde(default)]
    pub waveform: Waveform,
}
impl Score {
    pub fn duration_seconds(&self) -> f64 {
        let end = self
            .notes
            .iter()
            .map(|n| n.start_tick as u64 + n.duration_ticks as u64)
            .max()
            .unwrap_or(0);
        end as f64 * 60.0 / self.bpm as f64 / self.ticks_per_beat as f64
    }
    pub fn validate(&self, max_seconds: u32) -> Result<()> {
        ensure!(
            (20..=300).contains(&self.bpm) && (24..=960).contains(&self.ticks_per_beat),
            "score tempo/grid out of range"
        );
        ensure!(
            !self.notes.is_empty() && self.notes.len() <= 1024,
            "score requires 1..1024 notes"
        );
        for n in &self.notes {
            ensure!(
                (21..=108).contains(&n.pitch)
                    && (1..=127).contains(&n.velocity)
                    && n.duration_ticks > 0,
                "note outside pitch/velocity/duration bounds"
            );
            ensure!(
                n.start_tick.checked_add(n.duration_ticks).is_some(),
                "tick overflow"
            );
        }
        ensure!(
            self.duration_seconds() > 0.0 && self.duration_seconds() <= max_seconds.min(120) as f64,
            "score exceeds duration cap"
        );
        // Both synth and single-channel MIDI agree on overlapping pitch behavior.
        let mut intervals = self.notes.clone();
        intervals.sort_by_key(|n| (n.pitch, n.start_tick));
        for pair in intervals.windows(2) {
            ensure!(
                pair[0].pitch != pair[1].pitch
                    || pair[0].start_tick + pair[0].duration_ticks <= pair[1].start_tick,
                "overlapping identical pitches denied"
            );
        }
        let mut events = Vec::new();
        for n in &self.notes {
            events.push((n.start_tick, 1i32));
            events.push((n.start_tick + n.duration_ticks, -1));
        }
        events.sort();
        let mut voices = 0;
        for (_, delta) in events {
            voices += delta;
            ensure!(voices <= 32, "polyphony exceeds 32 voices");
        }
        Ok(())
    }
}
pub fn render(score: &Score) -> Result<(Vec<u8>, Vec<u8>)> {
    score.validate(120)?;
    const RATE: u32 = 22050;
    let seconds_per_tick = 60.0 / score.bpm as f64 / score.ticks_per_beat as f64;
    let count = (score.duration_seconds() * RATE as f64).ceil() as usize;
    let mut samples = vec![0.0f64; count];
    for n in &score.notes {
        let start = (n.start_tick as f64 * seconds_per_tick * RATE as f64).round() as usize;
        let end = ((n.start_tick + n.duration_ticks) as f64 * seconds_per_tick * RATE as f64)
            .round() as usize;
        let len = end.saturating_sub(start);
        ensure!(len > 1, "note too short for sample rate");
        let freq = 440.0 * 2.0f64.powf((n.pitch as f64 - 69.0) / 12.0);
        for (i, sample) in samples.iter_mut().enumerate().take(end).skip(start) {
            let pos = i - start;
            let phase = freq * pos as f64 / RATE as f64;
            let wave = match score.waveform {
                Waveform::Sine => (std::f64::consts::TAU * phase).sin(),
                Waveform::Triangle => {
                    (2.0 / std::f64::consts::PI) * (std::f64::consts::TAU * phase).sin().asin()
                }
            };
            let attack = (pos as f64 / (RATE as f64 * 0.008)).min(1.0);
            let release = ((len - pos) as f64 / (RATE as f64 * 0.04)).min(1.0);
            *sample += wave * attack * release * n.velocity as f64 / 127.0;
        }
    }
    let peak = samples.iter().fold(0.0f64, |a, s| a.max(s.abs()));
    let gain = 0.9 / peak.max(1.0);
    let mut wav = Vec::with_capacity(44 + count * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + count as u32 * 2).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&RATE.to_le_bytes());
    wav.extend_from_slice(&(RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(count as u32 * 2).to_le_bytes());
    for s in samples {
        wav.extend_from_slice(&((s * gain * 32767.0).round() as i16).to_le_bytes());
    }
    let mut events = Vec::new();
    for n in &score.notes {
        events.push((n.start_tick, 1u8, n.pitch, n.velocity));
        events.push((n.start_tick + n.duration_ticks, 0, n.pitch, 0));
    }
    events.sort();
    let tempo = 60_000_000u32 / score.bpm as u32;
    let mut track = vec![
        0,
        0xff,
        0x51,
        3,
        (tempo >> 16) as u8,
        (tempo >> 8) as u8,
        tempo as u8,
    ];
    let mut prev = 0;
    for (tick, on, pitch, velocity) in events {
        vlq(tick - prev, &mut track);
        track.extend_from_slice(&[if on == 1 { 0x90 } else { 0x80 }, pitch, velocity]);
        prev = tick;
    }
    track.extend_from_slice(&[0, 0xff, 0x2f, 0]);
    let mut midi = b"MThd".to_vec();
    midi.extend_from_slice(&6u32.to_be_bytes());
    midi.extend_from_slice(&0u16.to_be_bytes());
    midi.extend_from_slice(&1u16.to_be_bytes());
    midi.extend_from_slice(&score.ticks_per_beat.to_be_bytes());
    midi.extend_from_slice(b"MTrk");
    midi.extend_from_slice(&(track.len() as u32).to_be_bytes());
    midi.extend_from_slice(&track);
    Ok((wav, midi))
}
fn vlq(mut n: u32, out: &mut Vec<u8>) {
    let mut buf = [0u8; 5];
    let mut i = 4;
    buf[i] = (n & 127) as u8;
    loop {
        n >>= 7;
        if n == 0 {
            break;
        }
        i -= 1;
        buf[i] = (n & 127) as u8 | 128;
    }
    out.extend_from_slice(&buf[i..]);
}
pub fn schema() -> Value {
    super::schema(
        json!({"bpm":{"type":"integer","minimum":20,"maximum":300},"ticks_per_beat":{"type":"integer","minimum":24,"maximum":960},"waveform":{"enum":["sine","triangle"]},"notes":{"type":"array","minItems":1,"maxItems":1024,"items":super::schema(json!({"pitch":{"type":"integer","minimum":21,"maximum":108},"velocity":{"type":"integer","minimum":1,"maximum":127},"start_tick":{"type":"integer","minimum":0},"duration_ticks":{"type":"integer","minimum":1}}),&["pitch","velocity","start_tick","duration_ticks"])}}),
        &["bpm", "ticks_per_beat", "notes"],
    )
}
