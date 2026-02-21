#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

extern crate alloc;

use alloc::{boxed::Box, vec};
use core::{f32, ptr};

mod default_predictor;
mod fft;
mod util;

pub use self::default_predictor::DefaultPredictor;
use self::util::OnceLock;

/// Used by [`Detector`] to predict the VAD score of a frame based on extracted features.
///
/// # Stability
/// If you wish to implement `Predictor` yourself, note that **the API is unstable and subject to change!**
pub trait Predictor {
	#[doc(hidden)]
	fn reset(&mut self);

	#[doc(hidden)]
	fn normalize(&self, features: &mut [f32]);

	#[doc(hidden)]
	fn predict(&mut self, features: &[f32], buffer: &mut [f32]) -> f32;
}

const FFT_SIZE: usize = 1024;
const WINDOW_SIZE: usize = 768;
const N_MELS: usize = 40;
const N_FEATURES: usize = N_MELS;
const N_CONTEXT_FRAMES: usize = 3;
const N_BINS: usize = FFT_SIZE / 2 + 1;
const PRE_EMPHASIS_COEFF: f32 = 0.97;
const POWER_FAC: f32 = 1. / (32768.0f32 * 32768.0);

struct Filters {
	mel_coeffs: Box<[(usize, Box<[f32]>)]>,
	window: Box<[f32]>
}

impl Filters {
	pub fn new() -> Self {
		let low_mel = 2595. * libm::log10f(1.0f32 + 0.0 / 700.);
		let high_mel = 2595. * libm::log10f(1.0f32 + 8000. / 700.);

		let mut bin_points = [0; 42];
		for i in 0..N_MELS + 2 {
			let mel = i as f32 * (high_mel - low_mel) / (N_MELS as f32 + 1.0) + low_mel;
			let hz = 700.0 * (libm::exp10f(mel / 2595.) - 1.);
			bin_points[i] = ((FFT_SIZE as f32 + 1.) * hz / 16000.) as usize;
		}

		let mut mel_coeffs = Vec::with_capacity(N_MELS);
		for i in 0..N_MELS {
			let mut points = Vec::with_capacity(bin_points[i + 2] - bin_points[i]);
			for j in bin_points[i]..bin_points[i + 1] {
				points.push((j - bin_points[i]) as f32 / (bin_points[i + 1] - bin_points[i]) as f32);
			}

			for j in bin_points[i + 1]..bin_points[i + 2] {
				points.push((bin_points[i + 2] - j) as f32 / (bin_points[i + 2] - bin_points[i + 1]) as f32);
			}

			// Mel filterbank is naturally very sparse. Rather than waste compute & storage on the whole matrix, only store
			// non-zero elements.
			mel_coeffs.push((bin_points[i], points.into_boxed_slice()));
		}

		// hann window
		let mut window = vec![0.0; WINDOW_SIZE].into_boxed_slice();
		let df = f32::consts::PI / WINDOW_SIZE as f32;
		for i in 0..WINDOW_SIZE {
			let x = libm::sinf(df * i as f32);
			window[i] = x * x;
		}

		Self {
			mel_coeffs: mel_coeffs.into_boxed_slice(),
			window
		}
	}
}

static FILTERS: OnceLock<Filters> = OnceLock::new();

pub struct Detector<P = DefaultPredictor> {
	predictor: P,
	prev_signal: f32,
	sample_ring_buffer: Box<[f32]>,
	features: Box<[f32]>,
	buffer: Box<[f32]>
}

impl Default for Detector<DefaultPredictor> {
	fn default() -> Self {
		Self::new(DefaultPredictor::new())
	}
}

impl<P: Predictor> Detector<P> {
	pub fn new(predictor: P) -> Self {
		// create filters now so we don't accidentally make the first `predict` take super long
		FILTERS.get_or_init(Filters::new);

		Self {
			predictor,
			prev_signal: 0.0,
			sample_ring_buffer: vec![0.0; 768].into_boxed_slice(),
			features: vec![0.0; N_FEATURES * N_CONTEXT_FRAMES].into_boxed_slice(),
			buffer: vec![0.0; 1026].into_boxed_slice()
		}
	}

	/// Resets the internal state of the voice activity detector.
	///
	/// The detector should be reset whenever:
	/// - the recording device changes; or
	/// - the detector is being used for a new audio sequence.
	pub fn reset(&mut self) {
		self.predictor.reset();
		self.prev_signal = 0.0;
		self.sample_ring_buffer.fill(0.0);
		self.features.fill(0.0);
	}

	/// Predicts the voice activity score of a single input frame of 16-bit PCM audio.
	///
	/// The frame:
	/// - should be sampled at 16 KHz;
	/// - should be exactly 256 samples (so 16 ms) in length.
	///
	/// The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
	/// can be adjusted according to application-specific needs.
	pub fn predict_i16(&mut self, frame: &[i16]) -> f32 {
		assert_eq!(frame.len(), 256);

		unsafe {
			ptr::copy(self.sample_ring_buffer.as_ptr().add(256), self.sample_ring_buffer.as_mut_ptr(), 512);
		};
		for (emph, sample) in (&mut self.sample_ring_buffer[512..]).iter_mut().zip(frame.iter()) {
			let sample = *sample as f32;
			*emph = sample - PRE_EMPHASIS_COEFF * self.prev_signal;
			self.prev_signal = sample;
		}

		self.predict_inner()
	}

	/// Predicts the voice activity score of a single input frame of 32-bit floating-point PCM audio.
	///
	/// The frame:
	/// - should be sampled at 16 KHz;
	/// - should be exactly 256 samples (so 16 ms) in length;
	/// - should consist only of samples in the range [-1, 1].
	///
	/// The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
	/// can be adjusted according to application-specific needs.
	pub fn predict_f32(&mut self, frame: &[f32]) -> f32 {
		assert_eq!(frame.len(), 256);

		debug_assert!(
			*frame
				.iter()
				.max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap_or(core::cmp::Ordering::Equal))
				.unwrap() <= 1.0,
			"input frame should be in the range [-1, 1]"
		);

		/// We perform the FFT at i16 scale and scale down afterwards; doing the FFT at f32 scale ([-1, 1]) loses a lot
		/// of precision.
		const SCALE: f32 = 32768.0;

		unsafe {
			ptr::copy(self.sample_ring_buffer.as_ptr().add(256), self.sample_ring_buffer.as_mut_ptr(), 512);
		};
		for (emph, sample) in (&mut self.sample_ring_buffer[512..]).iter_mut().zip(frame.iter()) {
			let sample = *sample * SCALE;
			*emph = sample - PRE_EMPHASIS_COEFF * self.prev_signal;
			self.prev_signal = sample;
		}

		self.predict_inner()
	}

	fn predict_inner(&mut self) -> f32 {
		let filters = FILTERS.get_or_init(Filters::new);

		// windowize for FFT
		for i in 0..WINDOW_SIZE {
			self.buffer[i] = self.sample_ring_buffer[i] * filters.window[i];
		}
		// FFT size is 1024 but window size is 768, so fill the rest with zeros (+2 to store nyquist frequency)
		unsafe {
			ptr::write_bytes(self.buffer.as_mut_ptr().add(WINDOW_SIZE), 0, 256 + 2);
		};

		fft::rfft_1024(&mut self.buffer);
		for i in 0..N_BINS {
			let j = i * 2;
			self.buffer[i] = fft::Complex32::new(self.buffer[j], self.buffer[j + 1]).norm_sqr()
				// downscale from i16 scale
				* POWER_FAC;
		}

		unsafe {
			ptr::copy(self.features.as_ptr().add(N_FEATURES), self.features.as_mut_ptr(), N_FEATURES * (N_CONTEXT_FRAMES - 1));
		};
		let cur_frame_features = &mut self.features[(N_FEATURES * (N_CONTEXT_FRAMES - 1))..];
		for i in 0..N_MELS {
			let mut per_band_value = 0.;
			let (start, ref coeffs) = filters.mel_coeffs[i];
			for (offs, coeff) in coeffs.iter().enumerate() {
				per_band_value += self.buffer[start + offs] * *coeff;
			}

			cur_frame_features[i] = libm::logf(per_band_value + 1e-20);
		}
		self.predictor.normalize(cur_frame_features);

		self.predictor.predict(&self.features, &mut self.buffer)
	}
}
