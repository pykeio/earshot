//! Earshot is a ridiculously fast & accurate [voice activity detector](https://en.wikipedia.org/wiki/Voice_activity_detection).
//!
//! Earshot operates on 16 millisecond frames of mono audio sampled at 16,000 Hz & supports streaming. Earshot detects
//! voice in any language and is resilient to most kinds of environmental noise with an SNR ≥ 3dB.
//!
//! ## Streaming usage
//! ```
//! use earshot::Detector;
//! # fn get_frame_receiver() -> impl Iterator<Item = &'static [f32]> { core::iter::once([0.0f32; earshot::FRAME_SIZE].as_ref()) }
//!
//! let mut detector = Detector::default();
//!
//! // Get a stream of frames from a microphone or the network or something
//! let mut frame_receiver = get_frame_receiver();
//!
//! while let Some(frame) = frame_receiver.next() {
//! 	// Frames must be exactly `earshot::FRAME_SIZE` in length (256 samples/16ms).
//! 	// They can be given as slices of either i16 or f32.
//! 	// Mono audio is expected by default. You can wrap frames in `earshot::Stereo(frame)` to use interleaved stereo audio.
//!
//! 	let score = detector.predict(frame);
//! 	if score.is_voice() {
//! 		println!("Voice detected! Confidence: {:.1}%", score.raw * 100.);
//! 	}
//! }
//! ```
//!
//! ## Segmentation
//! [`segments`] returns all speech segments in a complete audio buffer, like Silero VAD's `get_speech_timestamps`.
//! Unlike [`Detector`], which only gives you raw per-frame scores, `segments` includes smoothing, minimum segment
//! duration, etc.
//!
//! ```
//! use earshot::{SegmenterOptions, segments};
//!
//! # let file = std::fs::read("testdata/1.wav").unwrap();
//! # let audio = unsafe { std::slice::from_raw_parts(file.as_ptr().add(44).cast::<i16>(), file.len() / 2) };
//! #
//! for segment in segments(audio, &SegmenterOptions::default()) {
//! 	println!(
//! 		"Voice detected from {:.2}s - {:.2}s ({:.2}s)",
//! 		segment.start_secs(),
//! 		segment.end_secs(),
//! 		segment.duration_secs()
//! 	);
//! }
//! ```

#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!("earshot's `libm` feature must be enabled when the `std` feature is disabled");

#[cfg(feature = "alloc")]
extern crate alloc;

use core::{cmp::Ordering, f32, ptr};

#[cfg(feature = "__ffi")]
mod c;
mod default_predictor;
mod fft;
mod filters;
mod frame;
mod segments;
mod util;

use self::util::libm;
pub use self::{
	default_predictor::DefaultPredictor,
	frame::{FRAME_SIZE, Frame, Stereo},
	segments::{ExactChunks, IntoFrames, Segment, SegmenterOptions, segments, segments_with_predictor}
};

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

pub const SAMPLE_RATE: usize = 16_000;
const FFT_SIZE: usize = 1024;
const CONTEXT_FRAMES: usize = 3;
const WINDOW_SIZE: usize = FRAME_SIZE * CONTEXT_FRAMES;
const N_MELS: usize = 40;
const N_FEATURES: usize = N_MELS;
const N_BINS: usize = FFT_SIZE / 2 + 1;
const PRE_EMPHASIS_COEFF: f32 = 0.97;
const POWER_FAC: f32 = 1. / (32768.0f32 * 32768.0);

/// A streaming voice activity detector. Create one per separate audio stream.
///
/// # Stack size
/// `Detector` is a fairly large object, as it allocates its state (about 8 KiB by default) on the stack. If stack space
/// is a concern, create a `Box<Detector>` with [`Detector::default_boxed`]/[`Detector::new_boxed`] instead. (Maps or
/// vectors of `Detector`s shouldn't need to worry about this).
pub struct Detector<P = DefaultPredictor> {
	predictor: P,
	prev_signal: f32,
	sample_ring_buffer: [f32; WINDOW_SIZE],
	features: [f32; N_FEATURES * CONTEXT_FRAMES],
	buffer: [f32; FFT_SIZE + 2]
}

impl Default for Detector<DefaultPredictor> {
	fn default() -> Self {
		Self::new(DefaultPredictor::new())
	}
}

impl Detector<DefaultPredictor> {
	#[inline]
	pub const fn const_default() -> Detector<DefaultPredictor> {
		Self::new(DefaultPredictor::new())
	}

	/// Creates a new `Detector` directly on the heap, without ever allocating the large amount of stack space that
	/// `Detector` normally uses.
	///
	/// This is preferred over `Box::<Detector>::default()`, since that creates the detector on the stack before moving
	/// it to the heap.
	#[cfg(feature = "alloc")]
	#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
	pub fn default_boxed() -> Box<Self> {
		// TODO: use new_zeroed instead, MSRV 1.92
		let mut boxed = alloc::boxed::Box::<Self>::new_uninit();
		let mut detector = unsafe {
			let boxed_ptr = boxed.as_mut_ptr();
			core::ptr::write(&raw mut (*boxed_ptr).predictor, DefaultPredictor::new());
			boxed.assume_init()
		};
		detector.prev_signal = 0.0;
		detector.sample_ring_buffer.fill(0.0);
		detector.features.fill(0.0);
		detector.buffer.fill(0.0);
		detector
	}
}

impl<P: Predictor> Detector<P> {
	/// Creates a new `Detector` on the stack.
	///
	/// To create directly on the heap, see [`Detector::new_boxed`] instead.
	#[inline]
	pub const fn new(predictor: P) -> Self {
		Self {
			predictor,
			prev_signal: 0.0,
			sample_ring_buffer: [0.0; WINDOW_SIZE],
			features: [0.0; N_FEATURES * CONTEXT_FRAMES],
			buffer: [0.0; FFT_SIZE + 2]
		}
	}

	/// Creates a new `Detector` directly on the heap.
	///
	/// This is more efficient than `Box::new(Detector::new(predictor))`, since that creates the detector on the stack
	/// before moving it to the heap.
	#[cfg(feature = "alloc")]
	#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
	#[inline]
	pub fn new_boxed(predictor: P) -> Box<Self> {
		// TODO: use new_zeroed instead, MSRV 1.92
		let mut boxed = alloc::boxed::Box::<Self>::new_uninit();
		let mut detector = unsafe {
			let boxed_ptr = boxed.as_mut_ptr();
			core::ptr::write(&raw mut (*boxed_ptr).predictor, predictor);
			boxed.assume_init()
		};
		detector.prev_signal = 0.0;
		detector.sample_ring_buffer.fill(0.0);
		detector.features.fill(0.0);
		detector.buffer.fill(0.0);
		detector
	}

	/// Resets the internal state of the voice activity detector.
	#[inline]
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
	#[deprecated = "use Detector::predict instead"]
	#[inline]
	#[doc(hidden)]
	pub fn predict_i16(&mut self, frame: &[i16]) -> f32 {
		self.predict(frame).raw
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
	#[deprecated = "use Detector::predict instead"]
	#[inline]
	#[doc(hidden)]
	pub fn predict_f32(&mut self, frame: &[f32]) -> f32 {
		self.predict(frame).raw
	}

	/// Detect voice in a single frame of a 16 KHz PCM audio stream.
	///
	/// The frame must be exactly [`FRAME_SIZE`] samples in length (or `FRAME_SIZE * 2` if [`Stereo`] is used).
	///
	/// ```
	/// let mut detector = earshot::Detector::default();
	/// # let frame = [0.0_f32; earshot::FRAME_SIZE].as_ref();
	/// if detector.predict(frame).is_voice() {
	/// 	// ...
	/// }
	/// ```
	pub fn predict<F: Frame>(&mut self, frame: F) -> Score {
		debug_assert_eq!(frame.len(), FRAME_SIZE, "frame should be exactly {FRAME_SIZE} samples");
		if frame.len() != FRAME_SIZE {
			return Score { raw: 0.0 };
		}

		const OTHER_FRAMES: usize = WINDOW_SIZE - FRAME_SIZE;
		unsafe {
			ptr::copy(self.sample_ring_buffer.as_ptr().add(FRAME_SIZE), self.sample_ring_buffer.as_mut_ptr(), OTHER_FRAMES);
		};
		for (emph, sample) in (&mut self.sample_ring_buffer[OTHER_FRAMES..]).iter_mut().zip(frame.samples()) {
			debug_assert!((-32768.0..=32768.0).contains(&sample), "encountered a bad sample (note f32 inputs must be within [-1, 1])");

			*emph = sample - PRE_EMPHASIS_COEFF * self.prev_signal;
			self.prev_signal = sample;
		}

		self.predict_inner()
	}

	fn predict_inner(&mut self) -> Score {
		// windowize for FFT
		for i in 0..WINDOW_SIZE {
			self.buffer[i] = self.sample_ring_buffer[i] * filters::HANN_WINDOW[i];
		}
		// FFT size is 1024 but window size is 768, so fill the rest with zeros (+2 to store nyquist frequency)
		unsafe {
			ptr::write_bytes(self.buffer.as_mut_ptr().add(WINDOW_SIZE), 0, const { (FFT_SIZE - WINDOW_SIZE) + 2 });
		};

		fft::rfft_1024(&mut self.buffer);
		for i in 0..N_BINS {
			let j = i * 2;
			self.buffer[i] = fft::Complex32::new(self.buffer[j], self.buffer[j + 1]).norm_sqr()
				// downscale from i16 scale
				* POWER_FAC;
		}

		unsafe {
			ptr::copy(self.features.as_ptr().add(N_FEATURES), self.features.as_mut_ptr(), N_FEATURES * (CONTEXT_FRAMES - 1));
		};
		let cur_frame_features = &mut self.features[(N_FEATURES * (CONTEXT_FRAMES - 1))..];
		for i in 0..N_MELS {
			let mut per_band_value = 0.;
			let (start, coeffs) = filters::MEL_COEFFS[i];
			for (offs, coeff) in coeffs.iter().enumerate() {
				per_band_value += self.buffer[start + offs] * *coeff;
			}

			cur_frame_features[i] = libm::logf(per_band_value + 1e-20);
		}
		self.predictor.normalize(cur_frame_features);

		let score = self.predictor.predict(&self.features, &mut self.buffer);
		Score::new(score)
	}
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[non_exhaustive]
pub struct Score {
	/// The raw predicted score, in the range `[0, 1]`.
	pub raw: f32
}

impl Score {
	#[inline]
	pub(crate) const fn new(raw: f32) -> Self {
		Self { raw }
	}

	#[inline]
	pub const fn is_voice(&self) -> bool {
		self.raw >= 0.5
	}

	#[inline]
	pub const fn is_silence(&self) -> bool {
		self.raw < 0.5
	}
}

// score will never be NaN or infinity so we can implement eq & ord
impl Eq for Score {}
impl Ord for Score {
	fn cmp(&self, other: &Self) -> Ordering {
		self.raw.total_cmp(&other.raw)
	}
}
