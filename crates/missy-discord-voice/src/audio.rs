use anyhow::{Result, ensure};
use std::io::Cursor;
/// Energy gate at 100 PCM16 RMS, approximately -50dBFS. Silence never extends
/// an utterance; this is not a noise-suppressing speech classifier.
pub(crate) fn speech_energy(pcm: &[i16]) -> bool {
    if pcm.is_empty() {
        return false;
    }
    let mean = pcm
        .iter()
        .map(|s| {
            let s = *s as i64;
            s * s
        })
        .sum::<i64>()
        / pcm.len() as i64;
    mean >= 10_000
}

/// Discord stereo S16LE -> mono S16LE at 48kHz. STT owns any resampling.
pub(crate) fn mono(pcm: &[i16]) -> Vec<u8> {
    pcm.as_chunks::<2>()
        .0
        .iter()
        .flat_map(|p| (((p[0] as i32 + p[1] as i32) / 2) as i16).to_le_bytes())
        .collect()
}

/// Decode bounded PCM WAV, resample to Discord 48kHz stereo. No ffmpeg or URLs.
pub(crate) fn wav_frames(wav: &[u8], max_seconds: u16) -> Result<Vec<Vec<i16>>> {
    ensure!(
        wav.len() <= max_seconds as usize * 192_000 + 65_536,
        "TTS WAV exceeds bound"
    );
    let mut reader = hound::WavReader::new(Cursor::new(wav))?;
    let spec = reader.spec();
    ensure!(
        spec.sample_format == hound::SampleFormat::Int
            && spec.bits_per_sample == 16
            && (1..=2).contains(&spec.channels)
            && (8_000..=48_000).contains(&spec.sample_rate),
        "TTS requires PCM16 WAV, 8..48kHz mono/stereo"
    );
    ensure!(
        reader.duration() as u64 <= spec.sample_rate as u64 * max_seconds as u64,
        "TTS duration exceeds bound"
    );
    let samples = reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let count = samples.len() / spec.channels as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    let out_count = count * 48_000 / spec.sample_rate as usize;
    let mut out = Vec::with_capacity(out_count.div_ceil(960));
    let mut frame = Vec::with_capacity(1920);
    for i in 0..out_count {
        let position = i as f64 * spec.sample_rate as f64 / 48_000.0;
        let at = position as usize;
        let next = (at + 1).min(count - 1);
        let mix = position.fract();
        for ch in 0..2 {
            let ch = ch.min(spec.channels as usize - 1);
            let a = samples[at * spec.channels as usize + ch] as f64;
            let b = samples[next * spec.channels as usize + ch] as f64;
            frame.push((a + (b - a) * mix) as i16);
        }
        if frame.len() == 1920 {
            out.push(std::mem::replace(&mut frame, Vec::with_capacity(1920)));
        }
    }
    if !frame.is_empty() {
        frame.resize(1920, 0);
        out.push(frame);
    }
    Ok(out)
}
