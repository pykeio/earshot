use std::{fs, hint::black_box, slice};

use criterion::{Criterion, criterion_group, criterion_main};
use earshot::{VoiceActivityDetector, VoiceActivityModel, VoiceActivityProfile};

fn bench_vad_8khz(c: &mut Criterion) {
	let file = fs::read("tests/data/audio_tiny8.raw").unwrap();
	let i16_samples = unsafe { slice::from_raw_parts(file.as_ptr().cast::<i16>(), file.len() / 2) };
	let mut vad = VoiceActivityDetector::new_with_model(VoiceActivityModel::ES_ALPHA, VoiceActivityProfile::VERY_AGGRESSIVE);
	c.bench_function("VAD - 8 KHz (Real world)", |b| {
		b.iter(|| {
			for frame in i16_samples.chunks_exact(240) {
				let _ = black_box(vad.predict_8khz(black_box(frame)));
			}
		})
	});
	c.bench_function("VAD - 8 KHz (Single frame)", |b| {
		let frame = (0..240 as i16).map(|i| i.wrapping_mul(i)).collect::<Vec<_>>();
		b.iter(|| {
			let _ = black_box(vad.predict_8khz(black_box(&frame)));
		})
	});
}

fn bench_vad_16khz(c: &mut Criterion) {
	let file = fs::read("tests/data/audio_tiny16.raw").unwrap();
	let i16_samples = unsafe { slice::from_raw_parts(file.as_ptr().cast::<i16>(), file.len() / 2) };
	let mut vad = VoiceActivityDetector::new_with_model(VoiceActivityModel::ES_ALPHA, VoiceActivityProfile::VERY_AGGRESSIVE);
	c.bench_function("VAD - 16 KHz (Real world)", |b| {
		b.iter(|| {
			for frame in i16_samples.chunks_exact(240) {
				let _ = black_box(vad.predict_16khz(black_box(frame)));
			}
		})
	});
	c.bench_function("VAD - 16 KHz (Single frame)", |b| {
		let frame = (0..480 as i16).map(|i| i.wrapping_mul(i)).collect::<Vec<_>>();
		b.iter(|| {
			let _ = black_box(vad.predict_16khz(black_box(&frame)));
		})
	});
}

fn bench_vad_32khz(c: &mut Criterion) {
	let file = fs::read("tests/data/audio_tiny32.raw").unwrap();
	let i16_samples = unsafe { slice::from_raw_parts(file.as_ptr().cast::<i16>(), file.len() / 2) };
	let mut vad = VoiceActivityDetector::new_with_model(VoiceActivityModel::ES_ALPHA, VoiceActivityProfile::VERY_AGGRESSIVE);
	c.bench_function("VAD - 32 KHz (Real world)", |b| {
		b.iter(|| {
			for frame in i16_samples.chunks_exact(240) {
				let _ = black_box(vad.predict_32khz(black_box(frame)));
			}
		})
	});
	c.bench_function("VAD - 32 KHz (Single frame)", |b| {
		let frame = (0..960 as i16).map(|i| i.wrapping_mul(i)).collect::<Vec<_>>();
		b.iter(|| {
			let _ = black_box(vad.predict_32khz(black_box(&frame)));
		})
	});
}

fn bench_vad_48khz(c: &mut Criterion) {
	let file = fs::read("tests/data/audio_tiny48.raw").unwrap();
	let i16_samples = unsafe { slice::from_raw_parts(file.as_ptr().cast::<i16>(), file.len() / 2) };
	let mut vad = VoiceActivityDetector::new_with_model(VoiceActivityModel::ES_ALPHA, VoiceActivityProfile::VERY_AGGRESSIVE);
	c.bench_function("VAD - 48 KHz (Real world)", |b| {
		b.iter(|| {
			for frame in i16_samples.chunks_exact(240) {
				let _ = black_box(vad.predict_48khz(black_box(frame)));
			}
		})
	});
	c.bench_function("VAD - 48 KHz (Single frame)", |b| {
		let frame = (0..1440 as i16).map(|i| i.wrapping_mul(i)).collect::<Vec<_>>();
		b.iter(|| {
			let _ = black_box(vad.predict_48khz(black_box(&frame)));
		})
	});
}

criterion_group!(vad, bench_vad_8khz, bench_vad_16khz, bench_vad_32khz, bench_vad_48khz);
criterion_main!(vad);
