#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

extern crate alloc;

use alloc::{boxed::Box, vec};
use core::{f32, ptr};

mod fft;
mod quantized_predictor;
mod util;

#[cfg(feature = "embed-weights")]
pub use self::quantized_predictor::default_weights as default_quantized_weights;
pub use self::quantized_predictor::{PackedWeights, QuantizedPredictor};
use self::util::OnceLock;

pub trait Predictor {
	fn reset(&mut self);
	fn predict(&mut self, features: &[f32], buffer: &mut [f32]) -> f32;
}

const FFT_SIZE: usize = 1024;
const WINDOW_SIZE: usize = 768;
const N_MELS: usize = 40;
const N_FEATURES: usize = N_MELS + 1;
const N_CONTEXT_FRAMES: usize = 3;
const N_BINS: usize = FFT_SIZE / 2 + 1;
const PRE_EMPHASIS_COEFF: f32 = 0.97;
const POWER_FAC: f32 = 1. / (32768.0f32 * 32768.0);

#[rustfmt::skip]
const FEATURE_MEANS: [f32; 40] = [
    -8.198236465454, -6.265716552734, -5.483818531036, -4.758691310883,
	-4.417088985443, -4.142892837524, -3.912850379944, -3.845927953720,
	-3.657090425491, -3.723418712616, -3.876134157181, -3.843890905380,
    -3.690405130386, -3.756065845490, -3.698696136475, -3.650463104248,
	-3.700468778610, -3.567321300507, -3.498900175095, -3.477807044983,
	-3.458816051483, -3.444923877716, -3.401328563690, -3.306261301041,
    -3.278556823730, -3.233250856400, -3.198616027832, -3.204526424408,
	-3.208798646927, -3.257838010788, -3.381376743317, -3.534021377563,
	-3.640867948532, -3.726858854294, -3.773730993271, -3.804667234421,
    -3.832901000977, -3.871120452881, -3.990592956543, -4.480289459229
];

#[rustfmt::skip]
const FEATURE_STDS: [f32; 40] = [
    5.166063785553, 4.977209568024, 4.698895931244, 4.630621433258,
	4.634347915649, 4.641156196594, 4.640676498413, 4.666367053986,
	4.650534629822, 4.640020847321, 4.637400150299, 4.620099067688,
    4.596316337585, 4.562654972076, 4.554360389709, 4.566910743713,
	4.562489986420, 4.562412738800, 4.585299491882, 4.600179672241,
	4.592845916748, 4.585922718048, 4.583496570587, 4.626092910767,
    4.626957893372, 4.626289367676, 4.637005805969, 4.683015823364,
	4.726813793182, 4.734289646149, 4.753227233887, 4.849722862244,
	4.869434833527, 4.884482860565, 4.921327114105, 4.959212303162,
    4.996619224548, 5.044823646545, 5.072216987610, 5.096439361572
];

struct Filters {
	mel_coeffs: Box<[f32]>,
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

		let mut mel_coeffs = vec![0.0; N_MELS * N_BINS].into_boxed_slice();
		for i in 0..N_MELS {
			for j in bin_points[i]..bin_points[i + 1] {
				mel_coeffs[(i * N_BINS) + j] = (j - bin_points[i]) as f32 / (bin_points[i + 1] - bin_points[i]) as f32;
			}

			for j in bin_points[i + 1]..bin_points[i + 2] {
				mel_coeffs[(i * N_BINS) + j] = (bin_points[i + 2] - j) as f32 / (bin_points[i + 2] - bin_points[i + 1]) as f32;
			}
		}

		// hann window
		let mut window = vec![0.0; WINDOW_SIZE].into_boxed_slice();
		let df = f32::consts::PI / WINDOW_SIZE as f32;
		for i in 0..WINDOW_SIZE {
			let x = libm::sinf(df * i as f32);
			window[i] = x * x;
		}

		Self { mel_coeffs, window }
	}
}

static FILTERS: OnceLock<Filters> = OnceLock::new();

pub struct Detector<P> {
	predictor: P,
	prev_signal: f32,
	sample_ring_buffer: Box<[f32]>,
	features: Box<[f32]>,
	buffer: Box<[f32]>
}

impl<P: Predictor + Default> Default for Detector<P> {
	fn default() -> Self {
		Self::new(P::default())
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
			for j in 0..N_BINS {
				per_band_value += self.buffer[j] * filters.mel_coeffs[(i * N_BINS) + j];
			}

			per_band_value = libm::logf(per_band_value + 1e-20);
			cur_frame_features[i] = (per_band_value - FEATURE_MEANS[i]) / FEATURE_STDS[i];
		}

		self.predictor.predict(&self.features, &mut self.buffer)
	}
}
