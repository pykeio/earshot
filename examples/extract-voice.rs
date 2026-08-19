use std::{env::args, fs::File, io::BufWriter};

use earshot::Detector;
use hound::{SampleFormat, WavSpec, WavWriter};
use itertools::Itertools;
use rubato::{FftFixedOut, Resampler};

struct VadWriter {
	detector: Detector,
	resampler: FftFixedOut<f32>,
	output: WavWriter<BufWriter<File>>,
	buf: Vec<f32>,
	resampled_out_buf: Vec<Vec<f32>>,
	detect_buf: Vec<f32>,
	input_chunk: Vec<f32>
}

impl VadWriter {
	pub fn new(resampler: FftFixedOut<f32>, output: WavWriter<BufWriter<File>>) -> Self {
		Self {
			detector: Detector::default(),
			output,
			buf: vec![],
			resampled_out_buf: resampler.output_buffer_allocate(true),
			resampler,
			detect_buf: vec![],
			input_chunk: vec![0.0; 256]
		}
	}

	pub fn push_chunk(&mut self, iter: impl Iterator<Item = f32>) -> Result<(), Box<dyn std::error::Error>> {
		self.buf.extend(iter);

		while !self.buf.is_empty() {
			let (input_taken, output_written) = self.resampler.process_into_buffer(&[&self.buf], &mut self.resampled_out_buf, None)?;
			self.buf.drain(..input_taken);
			self.detect_buf.extend_from_slice(&self.resampled_out_buf[0][..output_written]);
		}

		while self.detect_buf.len() >= 256 {
			for (i, x) in self.detect_buf.drain(..256).enumerate() {
				self.input_chunk[i] = x;
			}

			if self.detector.predict(&self.input_chunk).is_voice() {
				for sample in self.input_chunk.iter().copied() {
					self.output.write_sample(sample)?;
				}
			}
		}

		Ok(())
	}

	pub fn finalize(self) -> Result<(), Box<dyn std::error::Error>> {
		self.output.finalize()?;
		Ok(())
	}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut args = args().skip(1);
	let Some(input) = args.next() else {
		eprintln!("cargo run --example extract-voice -- [wav] [out]");
		return Ok(());
	};
	let Some(output) = args.next() else {
		eprintln!("cargo run --example extract-voice -- [wav] [out]");
		return Ok(());
	};

	let mut input = hound::WavReader::open(input)?;
	let spec = input.spec();
	let resampler = rubato::FftFixedOut::<f32>::new(spec.sample_rate as _, 16_000, 256, 4, 1)?;

	let output = hound::WavWriter::create(
		output,
		WavSpec {
			sample_rate: 16_000,
			channels: 1,
			sample_format: SampleFormat::Float,
			bits_per_sample: 32
		}
	)?;
	let mut writer = VadWriter::new(resampler, output);

	match (spec.sample_format, spec.bits_per_sample) {
		(SampleFormat::Float, 32) => {
			for samples in input.samples::<f32>().chunks(256).into_iter() {
				if spec.channels == 1 {
					writer.push_chunk(samples.map(|s| s.unwrap()))?;
				} else {
					writer.push_chunk(samples.chunks(spec.channels as _).into_iter().map(|c| {
						let mut mean = 0.0;
						for sample in c {
							mean += sample.unwrap();
						}
						mean / spec.channels as f32
					}))?;
				}
			}
		}
		(SampleFormat::Int, 16) => {
			for samples in input.samples::<i16>().chunks(256).into_iter() {
				if spec.channels == 1 {
					writer.push_chunk(samples.map(|s| s.unwrap() as f32 / 32768.0))?;
				} else {
					writer.push_chunk(samples.chunks(spec.channels as _).into_iter().map(|c| {
						let mut mean = 0.0;
						for sample in c {
							mean += sample.unwrap() as f32 / 32768.0;
						}
						mean / spec.channels as f32
					}))?;
				}
			}
		}
		(format, bits) => unimplemented!("unsupported sample format {format:?} in {bits} bits")
	}
	writer.finalize()?;
	Ok(())
}
