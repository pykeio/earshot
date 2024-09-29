use std::{fs, slice};

use criterion::{criterion_group, criterion_main, Criterion};
use earshot::__internal_downsampling::downsample_2x;

fn downsample_48khz_24khz(c: &mut Criterion) {
	c.bench_function("Downsample 48KHz -> 24KHz", |b| {
		let file = fs::read("tests/data/audio_tiny48.raw").unwrap();
		let i16_samples = unsafe { slice::from_raw_parts(file.as_ptr().cast::<i16>(), file.len() / 2) };
		let mut out = vec![0; i16_samples.len() / 2];
		let mut filter_state = [0, 0];
		b.iter(|| {
			downsample_2x(&i16_samples, &mut out, &mut filter_state);
		})
	});
}

criterion_group!(downsample, downsample_48khz_24khz);
criterion_main!(downsample);
