use core::{num::NonZeroU16, ops::Range, slice::ChunksExact};

use crate::{
	Detector, Predictor,
	default_predictor::DefaultPredictor,
	frame::{FRAME_SIZE, Frame, Stereo}
};

#[doc(hidden)]
pub trait ExactChunks: ExactSizeIterator {
	fn into_remainder(self) -> Self::Item;
}

impl<'a, T> ExactChunks for ChunksExact<'a, T> {
	fn into_remainder(self) -> Self::Item {
		self.remainder()
	}
}

#[doc(hidden)]
pub trait IntoFrames {
	type Frame: Frame;

	fn into_frames(self, chunk_size: usize) -> impl ExactChunks<Item = Self::Frame>;
}

/// A segment of voice detected in an audio buffer.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
	start: usize,
	end: usize
}

const SAMPLES_PER_FRAME: f32 = FRAME_SIZE as f32 / crate::SAMPLE_RATE as f32;
impl Segment {
	pub const fn start_frame(&self) -> usize {
		self.start
	}

	pub const fn end_frame(&self) -> usize {
		self.end
	}

	pub const fn as_frame_range(&self) -> Range<usize> {
		self.start..self.end
	}

	pub const fn duration_frames(&self) -> usize {
		let range = self.as_frame_range();
		range.end - range.start
	}

	pub const fn start_sample(&self) -> usize {
		self.start * FRAME_SIZE
	}

	pub const fn end_sample(&self) -> usize {
		self.end * FRAME_SIZE
	}

	pub const fn as_sample_range(&self) -> Range<usize> {
		(self.start * FRAME_SIZE)..(self.end * FRAME_SIZE)
	}

	pub const fn duration_samples(&self) -> usize {
		let range = self.as_sample_range();
		range.end - range.start
	}

	pub const fn start_secs(&self) -> f32 {
		self.start as f32 * SAMPLES_PER_FRAME
	}

	pub const fn end_secs(&self) -> f32 {
		self.end as f32 * SAMPLES_PER_FRAME
	}

	pub const fn as_secs_range(&self) -> Range<f32> {
		(self.start as f32 * SAMPLES_PER_FRAME)..(self.end as f32 * SAMPLES_PER_FRAME)
	}

	pub const fn duration_secs(&self) -> f32 {
		let range = self.as_secs_range();
		range.end - range.start
	}

	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn duration(&self) -> std::time::Duration {
		let range = self.as_secs_range();
		std::time::Duration::from_secs_f32(range.end - range.start)
	}
}

#[derive(Debug, Clone)]
pub struct SegmenterOptions {
	smoothing_factor: f32,
	attack_threshold: f32,
	decay_threshold: f32,
	preroll_frames: u16,
	min_duration: u16,
	start_delay: NonZeroU16,
	end_delay: NonZeroU16
}

impl Default for SegmenterOptions {
	fn default() -> Self {
		SegmenterOptions::const_default()
	}
}

impl SegmenterOptions {
	pub const fn const_default() -> Self {
		SegmenterOptions {
			smoothing_factor: 0.4,
			attack_threshold: 0.6,
			decay_threshold: 0.5,
			preroll_frames: 5,
			min_duration: 6,
			start_delay: const { NonZeroU16::new(2).unwrap() },
			end_delay: const { NonZeroU16::new(4).unwrap() }
		}
	}

	/// Configure the smoothing factor for raw probabilities from the model, in the range `[0, 1)`. `0` disables
	/// probability smoothing.
	///
	/// **Default: `0.4`**
	pub const fn with_smoothing_factor(mut self, factor: f32) -> Self {
		self.smoothing_factor = factor;
		self
	}

	/// Configures the minimum probability required to begin a voiced segment, in the range `[0, 1)`.
	///
	/// **Default: `0.6`**
	pub const fn with_attack_threshold(mut self, threshold: f32) -> Self {
		self.attack_threshold = threshold;
		self
	}

	/// Configures the minimum probability required to end a voiced segment, in the range `[0, 1)`.
	///
	/// **Default: `0.5`**
	pub const fn with_decay_threshold(mut self, threshold: f32) -> Self {
		self.decay_threshold = threshold;
		self
	}

	/// Configures the number of frames to shift segment starts backwards by, to account for any frames at the start of
	/// a segment that may have been lost to smoothing.
	///
	/// **Default: `5`**
	pub const fn with_preroll_frames(mut self, preroll_frames: u16) -> Self {
		self.preroll_frames = preroll_frames;
		self
	}

	/// Configures the minimum number of voiced frames required to output a segment.
	///
	/// **Default: `6`**
	pub const fn with_min_duration(mut self, min_duration: u16) -> Self {
		self.min_duration = min_duration;
		self
	}

	/// Require at least this number of consecutive voice frames to start a segment.
	///
	/// **Default: `2`**
	pub const fn with_start_delay(mut self, start_delay: NonZeroU16) -> Self {
		self.start_delay = start_delay;
		self
	}

	/// Require at least this number of consecutive silence frames to end a segment.
	///
	/// **Default: `4`**
	pub const fn with_end_delay(mut self, end_delay: NonZeroU16) -> Self {
		self.end_delay = end_delay;
		self
	}
}

enum SegmenterState {
	Silence,
	PendingStart { voice_frames: u16 },
	Voice { start: usize, silence_frames: u16 }
}

struct Segmenter<I, P = DefaultPredictor> {
	detector: Detector<P>,
	frames: I,
	options: SegmenterOptions,
	smoothed_prob: f32,
	n_frames: usize,
	state: SegmenterState
}

impl<I, P> Segmenter<I, P> {
	fn new(detector: Detector<P>, frames: I, options: &SegmenterOptions) -> Self {
		Segmenter {
			detector,
			frames,
			options: options.clone(),
			smoothed_prob: 0.0,
			n_frames: 0,
			state: SegmenterState::Silence
		}
	}
}

impl<F: Frame, I: ExactChunks<Item = F>, P: Predictor> Iterator for Segmenter<I, P> {
	type Item = Segment;

	fn next(&mut self) -> Option<Self::Item> {
		while let Some(frame) = self.frames.next() {
			let raw_score = self.detector.predict(frame).raw;
			self.smoothed_prob = self.options.smoothing_factor * self.smoothed_prob + (1.0 - self.options.smoothing_factor) * raw_score;

			let is_voice = match self.state {
				SegmenterState::Silence | SegmenterState::PendingStart { .. } => self.smoothed_prob >= self.options.attack_threshold,
				SegmenterState::Voice { .. } => self.smoothed_prob >= self.options.decay_threshold
			};
			match &mut self.state {
				SegmenterState::Silence => {
					if is_voice {
						self.state = SegmenterState::PendingStart { voice_frames: 1 };
					}
				}
				SegmenterState::PendingStart { voice_frames } => {
					if is_voice {
						*voice_frames += 1;
						if *voice_frames >= self.options.start_delay.get() {
							self.state = SegmenterState::Voice {
								start: (self.n_frames + *voice_frames as usize).saturating_sub(self.options.preroll_frames as usize),
								silence_frames: 0
							};
						}
					} else {
						self.state = SegmenterState::Silence;
					}
				}
				SegmenterState::Voice { start, silence_frames } => {
					if is_voice {
						*silence_frames = 0;
					} else {
						*silence_frames += 1;
					}

					if *silence_frames >= self.options.end_delay.get() {
						let start = *start;
						self.state = SegmenterState::Silence;
						if self.n_frames - start >= self.options.min_duration as usize {
							return Some(Segment { start, end: self.n_frames });
						}
					}
				}
			}

			self.n_frames += 1;
		}

		if let SegmenterState::Voice { start, silence_frames } = self.state
			&& silence_frames >= self.options.end_delay.get()
			&& self.n_frames - start >= self.options.min_duration as usize
		{
			self.state = SegmenterState::Silence;
			Some(Segment { start, end: self.n_frames })
		} else {
			None
		}
	}
}

/// Returns an iterator over all voiced segments in an audio buffer.
///
/// ```
/// use earshot::{SegmenterOptions, segments};
///
/// # let file = std::fs::read("testdata/1.wav").unwrap();
/// # let audio = unsafe { std::slice::from_raw_parts(file.as_ptr().add(44).cast::<i16>(), file.len() / 2) };
/// #
/// for segment in segments(audio, &SegmenterOptions::default()) {
/// 	println!(
/// 		"Voice detected from {:.2}s - {:.2}s ({:.2}s)",
/// 		segment.start_secs(),
/// 		segment.end_secs(),
/// 		segment.duration_secs()
/// 	);
/// }
/// ```
///
/// You can also provide interleaved stereo audio:
/// ```
/// use earshot::{SegmenterOptions, Stereo, segments};
///
/// # let audio = vec![0i16; 512];
/// for segment in segments(Stereo(audio.as_slice()), &SegmenterOptions::default()) {
/// 	// ...
/// }
/// ```
pub fn segments<T: IntoFrames>(data: T, options: &SegmenterOptions) -> impl Iterator<Item = Segment> + use<T> {
	Segmenter::new(Detector::const_default(), data.into_frames(FRAME_SIZE), options)
}

/// Returns an iterator over all voiced segments in an audio buffer using a custom [`Predictor`].
///
/// See [`segments`] for more information.
///
/// ```
/// use earshot::{SegmenterOptions, segments_with_predictor};
///
/// # let file = std::fs::read("testdata/1.wav").unwrap();
/// # let samples = unsafe { std::slice::from_raw_parts(file.as_ptr().add(44).cast::<i16>(), file.len() / 2) };
/// #
/// for segment in segments_with_predictor(samples, earshot::DefaultPredictor::new(), &SegmenterOptions::default()) {
/// 	println!(
/// 		"Voice detected from {:.2}s - {:.2}s ({:.2}s)",
/// 		segment.start_secs(),
/// 		segment.end_secs(),
/// 		segment.duration_secs()
/// 	);
/// }
/// ```
pub fn segments_with_predictor<T: IntoFrames, P: Predictor>(data: T, predictor: P, options: &SegmenterOptions) -> impl Iterator<Item = Segment> + use<T, P> {
	Segmenter::new(Detector::new(predictor), data.into_frames(FRAME_SIZE), options)
}

impl<'a> IntoFrames for &'a [i16] {
	type Frame = &'a [i16];

	fn into_frames(self, chunk_size: usize) -> impl ExactChunks<Item = Self::Frame> {
		self.chunks_exact(chunk_size)
	}
}

impl<'a> IntoFrames for &'a [f32] {
	type Frame = &'a [f32];

	fn into_frames(self, chunk_size: usize) -> impl ExactChunks<Item = Self::Frame> {
		self.chunks_exact(chunk_size)
	}
}

struct Map<I, F> {
	iter: I,
	f: F
}

impl<T, I: Iterator, F: FnMut(I::Item) -> T> Iterator for Map<I, F> {
	type Item = T;

	fn next(&mut self) -> Option<Self::Item> {
		self.iter.next().map(&mut self.f)
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		self.iter.size_hint()
	}
}

impl<T, I: ExactSizeIterator, F: FnMut(I::Item) -> T> ExactSizeIterator for Map<I, F> {
	fn len(&self) -> usize {
		self.iter.len()
	}
}

impl<'a, T, U: 'a, F: FnMut(&'a [T]) -> U> ExactChunks for Map<ChunksExact<'a, T>, F> {
	fn into_remainder(mut self) -> Self::Item {
		(self.f)(self.iter.remainder())
	}
}

impl<'a> IntoFrames for Stereo<&'a [i16]> {
	type Frame = Stereo<&'a [i16]>;

	fn into_frames(self, chunk_size: usize) -> impl ExactChunks<Item = Self::Frame> {
		Map {
			iter: self.0.chunks_exact(chunk_size * 2),
			f: Stereo
		}
	}
}

impl<'a> IntoFrames for Stereo<&'a [f32]> {
	type Frame = Stereo<&'a [f32]>;

	fn into_frames(self, chunk_size: usize) -> impl ExactChunks<Item = Self::Frame> {
		Map {
			iter: self.0.chunks_exact(chunk_size * 2),
			f: Stereo
		}
	}
}
