use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use earshot::Detector;

fn bench_vad(c: &mut Criterion) {
	let mut vad = Detector::default();
	c.bench_function("Single frame - f32", |b| {
		let frame = (0..256 as i16).map(|i| i.wrapping_mul(i) as f32).collect::<Vec<_>>();
		b.iter(|| {
			let _ = black_box(vad.predict_f32(black_box(&frame)));
		})
	});
	c.bench_function("Single frame - i16", |b| {
		let frame = (0..256 as i16).map(|i| i.wrapping_mul(i)).collect::<Vec<_>>();
		b.iter(|| {
			let _ = black_box(vad.predict_i16(black_box(&frame)));
		})
	});
}

criterion_group!(vad, bench_vad);
criterion_main!(vad);
