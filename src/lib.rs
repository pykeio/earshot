#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

//! Earshot is a fast voice activity detection library.
//!
//! For more details, see [`VoiceActivityDetector`].
//!
//! ```
//! use earshot::{VoiceActivityDetector, VoiceActivityProfile};
//!
//! let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
//!
//! # let mut stream = std::iter::once(vec![0; 320]);
//! while let Some(frame) = stream.next() {
//! 	let is_speech_detected = vad.predict_16khz(&frame).unwrap();
//! 	# assert_eq!(is_speech_detected, false);
//! }
//! ```

use core::{
	fmt,
	ops::{Deref, DerefMut}
};

#[cfg(feature = "alloc")]
extern crate alloc;
extern crate core;

pub(crate) mod energy;
pub(crate) mod filterbank;
pub(crate) mod gmm;
pub(crate) mod resample;
pub(crate) mod sp;
pub(crate) mod util;

#[doc(hidden)]
pub mod __internal_downsampling {
	pub use crate::resample::resample_48khz_to_8khz;
	pub use crate::sp::downsample_2x;
}

use self::{
	filterbank::calculate_features,
	gmm::gaussian_probability,
	resample::resample_48khz_to_8khz,
	sp::{downsample_2x, find_minimum},
	util::{div_i32_i16, norm_i32, weighted_average}
};

const SPECTRUM_WEIGHT: [i16; 6] = [6, 8, 10, 12, 14, 16];
const NOISE_UPDATE: i16 = 655; // Q15
const SPEECH_UPDATE: i16 = 6554; // Q15
const BACK_ETA: i16 = 154; // Q8
const MINIMUM_DIFFERENCE: [i16; 6] = [544, 544, 576, 576, 576, 576]; // Q5
const MAXIMUM_SPEECH: [i16; 6] = [11392, 11392, 11520, 11520, 11520, 11520]; // Q7
const MAX_SPEECH_FRAMES: u8 = 6;
const MIN_STD: i16 = 384;

#[doc(hidden)]
pub const NUM_GAUSSIANS: usize = 2;

#[derive(Debug, Clone)]
pub struct VoiceActivityModel {
	pub noise_weights: [[i16; 6]; NUM_GAUSSIANS],
	pub speech_weights: [[i16; 6]; NUM_GAUSSIANS],
	pub noise_means: [[i16; 6]; NUM_GAUSSIANS],
	pub speech_means: [[i16; 6]; NUM_GAUSSIANS],
	pub noise_stds: [[i16; 6]; NUM_GAUSSIANS],
	pub speech_stds: [[i16; 6]; NUM_GAUSSIANS],
	pub minimum_mean: [i16; NUM_GAUSSIANS],
	pub maximum_noise: [i16; 6]
}

impl VoiceActivityModel {
	/// The default VAD model from the original WebRTC source.
	pub const WRTC: VoiceActivityModel = VoiceActivityModel {
		noise_weights: [[34, 62, 72, 66, 53, 25], [94, 66, 56, 62, 75, 103]],
		speech_weights: [[48, 82, 45, 87, 50, 47], [80, 46, 83, 41, 78, 81]],
		noise_means: [[6738, 4892, 7065, 6715, 6771, 3369], [7646, 3863, 7820, 7266, 5020, 4362]],
		speech_means: [[8306, 10085, 10078, 11823, 11843, 6309], [9473, 9571, 10879, 7581, 8180, 7483]],
		noise_stds: [[378, 1064, 493, 582, 688, 593], [474, 697, 475, 688, 421, 455]],
		speech_stds: [[555, 505, 567, 524, 585, 1231], [509, 828, 492, 1540, 1079, 850]],
		minimum_mean: [640, 768],
		maximum_noise: [9216, 9088, 8960, 8832, 8704, 8576]
	};

	/// A custom model with slightly better accuracy than the default WebRTC model.
	pub const ES_ALPHA: VoiceActivityModel = VoiceActivityModel {
		noise_weights: [[34, 52, 61, 72, 42, 17], [103, 68, 65, 54, 64, 80]],
		speech_weights: [[43, 53, 18, 90, 30, 46], [56, 24, 76, 36, 52, 66]],
		noise_means: [[6799, 4771, 7070, 6775, 6843, 3225], [7659, 3939, 7626, 7328, 5091, 4424]],
		speech_means: [[8279, 10067, 10053, 11805, 11687, 6224], [9636, 9554, 10973, 7657, 8468, 7466]],
		noise_stds: [[361, 1044, 514, 557, 718, 630], [394, 763, 476, 735, 422, 453]],
		speech_stds: [[599, 451, 564, 483, 545, 1292], [409, 808, 526, 1447, 1089, 722]],
		minimum_mean: [640, 768],
		maximum_noise: [9232, 9101, 8952, 8830, 8653, 8555]
	};
}

impl Default for VoiceActivityModel {
	fn default() -> Self {
		Self::WRTC
	}
}

#[derive(Debug, Clone)]
pub struct VoiceActivityProfile {
	overhang_max_1: [i16; 3],
	overhang_max_2: [i16; 3],
	local_threshold: [i16; 3],
	global_threshold: [i16; 3]
}

impl VoiceActivityProfile {
	/// The least aggressive profile, tuned to preserve as much probable speech as possible.
	pub const QUALITY: VoiceActivityProfile = VoiceActivityProfile::new([8, 4, 3], [14, 7, 5], [24, 21, 24], [57, 48, 57]);
	/// Tuned for low bit rate scenarios.
	pub const LBR: VoiceActivityProfile = VoiceActivityProfile::new([8, 4, 3], [14, 7, 5], [37, 32, 37], [100, 80, 100]);
	/// Aggressive profile, tuned to minimize false positives.
	pub const AGGRESSIVE: VoiceActivityProfile = VoiceActivityProfile::new([6, 3, 2], [9, 5, 3], [82, 78, 82], [285, 260, 285]);
	/// Even more aggressive profile, tuned to provide the least amount of false positives.
	pub const VERY_AGGRESSIVE: VoiceActivityProfile = VoiceActivityProfile::new([6, 3, 2], [9, 5, 3], [94, 94, 94], [1100, 1050, 1100]);

	#[doc(hidden)]
	pub const fn new(overhang_max_1: [i16; 3], overhang_max_2: [i16; 3], local_threshold: [i16; 3], global_threshold: [i16; 3]) -> Self {
		Self {
			overhang_max_1,
			overhang_max_2,
			local_threshold,
			global_threshold
		}
	}
}

impl Default for VoiceActivityProfile {
	fn default() -> Self {
		Self::AGGRESSIVE
	}
}

#[cfg(feature = "alloc")]
#[repr(transparent)]
#[derive(Debug)]
struct MaybeHeapAllocated<T, const N: usize>(alloc::boxed::Box<[T]>);
#[cfg(not(feature = "alloc"))]
#[repr(transparent)]
struct MaybeHeapAllocated<T, const N: usize>([T; N]);

#[cfg(feature = "alloc")]
impl<T: Default + Clone + Copy, const N: usize> MaybeHeapAllocated<T, N> {
	pub fn new() -> Self {
		Self(alloc::vec![T::default(); N].into_boxed_slice())
	}
}
#[cfg(not(feature = "alloc"))]
impl<T: Default + Clone + Copy, const N: usize> MaybeHeapAllocated<T, N> {
	pub fn new() -> Self {
		Self([T::default(); N])
	}
}

impl<T, const N: usize> Deref for MaybeHeapAllocated<T, N> {
	type Target = [T];

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}
impl<T, const N: usize> DerefMut for MaybeHeapAllocated<T, N> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Error {
	/// The size of the audio frame passed to [`VoiceActivityDetector::predict_8khz()`] (or similar methods) is not one
	/// of the valid frame sizes for this sample rate.
	InvalidFrameSize { frame_size: usize, valid_sizes: [u16; 3] }
}

impl Error {
	pub(crate) fn invalid_frame_size_multiplier(self, multiplier: u8) -> Self {
		match self {
			Error::InvalidFrameSize { frame_size, valid_sizes } => Error::InvalidFrameSize {
				frame_size,
				valid_sizes: [valid_sizes[0] * multiplier as u16, valid_sizes[1] * multiplier as u16, valid_sizes[2] * multiplier as u16]
			},
			#[allow(unreachable_patterns)]
			e => e
		}
	}
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::InvalidFrameSize { frame_size, valid_sizes } => f.write_fmt(format_args!(
				"invalid frame size of {frame_size} samples; valid sizes are {} samples (10ms), {} samples (20ms), or {} samples (30ms)",
				valid_sizes[0], valid_sizes[1], valid_sizes[2]
			))
		}
	}
}

// TODO: `core::error::Error` was stabilized in 1.81.
#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub struct VoiceActivityDetector {
	downsampling_filter_states: MaybeHeapAllocated<i32, 4>,
	downsample_48khz_to_8khz_state: MaybeHeapAllocated<i32, 40>,
	downsample_tmp: MaybeHeapAllocated<i32, { 480 + 256 }>,
	noise_means: MaybeHeapAllocated<[i16; 6], NUM_GAUSSIANS>,
	speech_means: MaybeHeapAllocated<[i16; 6], NUM_GAUSSIANS>,
	noise_stds: MaybeHeapAllocated<[i16; 6], NUM_GAUSSIANS>,
	speech_stds: MaybeHeapAllocated<[i16; 6], NUM_GAUSSIANS>,
	frame_counter: usize,
	overhang: i16,
	num_of_speech: i16,
	age_vector: MaybeHeapAllocated<i16, { 16 * 6 }>,
	low_value_vector: MaybeHeapAllocated<i16, { 16 * 6 }>,
	mean_value: MaybeHeapAllocated<i16, 6>,
	split1_data: MaybeHeapAllocated<i16, 240>,
	split2_data: MaybeHeapAllocated<i16, 120>,
	upper_state: MaybeHeapAllocated<i16, 5>,
	lower_state: MaybeHeapAllocated<i16, 5>,
	hp_filter_state: MaybeHeapAllocated<i16, 4>,
	profile: VoiceActivityProfile,
	feature_vector: MaybeHeapAllocated<i16, 6>,
	total_power: i16,
	#[doc(hidden)]
	pub model: VoiceActivityModel
}

impl VoiceActivityDetector {
	/// Creates a new [`VoiceActivityDetector`] with the default model ([`VoiceActivityModel::WRTC`]) and given profile.
	///
	/// ```
	/// # use earshot::{VoiceActivityDetector, VoiceActivityProfile};
	/// let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
	/// ```
	pub fn new(profile: VoiceActivityProfile) -> Self {
		Self::new_with_model(VoiceActivityModel::default(), profile)
	}

	/// Creates a new [`VoiceActivityDetector`] with the given model and profile.
	///
	/// ```
	/// # use earshot::{VoiceActivityDetector, VoiceActivityModel, VoiceActivityProfile};
	/// let mut vad =
	/// 	VoiceActivityDetector::new_with_model(VoiceActivityModel::ES_ALPHA, VoiceActivityProfile::VERY_AGGRESSIVE);
	/// ```
	pub fn new_with_model(model: VoiceActivityModel, profile: VoiceActivityProfile) -> Self {
		let mut vad = Self {
			downsampling_filter_states: MaybeHeapAllocated::new(),
			downsample_48khz_to_8khz_state: MaybeHeapAllocated::new(),
			downsample_tmp: MaybeHeapAllocated::new(),
			noise_means: MaybeHeapAllocated::new(),
			speech_means: MaybeHeapAllocated::new(),
			noise_stds: MaybeHeapAllocated::new(),
			speech_stds: MaybeHeapAllocated::new(),
			frame_counter: 0,
			overhang: 0,
			num_of_speech: 0,
			age_vector: MaybeHeapAllocated::new(),
			low_value_vector: MaybeHeapAllocated::new(),
			mean_value: MaybeHeapAllocated::new(),
			split1_data: MaybeHeapAllocated::new(),
			split2_data: MaybeHeapAllocated::new(),
			upper_state: MaybeHeapAllocated::new(),
			lower_state: MaybeHeapAllocated::new(),
			hp_filter_state: MaybeHeapAllocated::new(),
			profile,
			feature_vector: MaybeHeapAllocated::new(),
			total_power: 0,
			model
		};
		vad.reset();
		vad
	}

	/// Resets the internal state of the VAD.
	///
	/// Ideally, this should be called whenever a new audio stream begins.
	///
	/// ```
	/// # use earshot::{VoiceActivityDetector, VoiceActivityProfile};
	/// let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
	///
	/// # let streams: [&[i16]; 0] = [];
	/// for stream in streams {
	/// 	let mut speech_frames = 0;
	/// 	for frame in stream.chunks_exact(240) {
	/// 		if let Ok(true) = vad.predict_8khz(frame) {
	/// 			speech_frames += 1;
	/// 		}
	/// 	}
	///
	/// 	vad.reset();
	/// }
	/// ```
	pub fn reset(&mut self) {
		self.frame_counter = 0;
		self.overhang = 0;
		self.num_of_speech = 0;

		self.downsampling_filter_states.fill(0);
		self.downsample_48khz_to_8khz_state.fill(0);

		self.noise_means.copy_from_slice(&self.model.noise_means);
		self.speech_means.copy_from_slice(&self.model.speech_means);
		self.noise_stds.copy_from_slice(&self.model.noise_stds);
		self.speech_stds.copy_from_slice(&self.model.speech_stds);

		self.low_value_vector.fill(10000);
		self.age_vector.fill(0);

		self.upper_state.fill(0);
		self.lower_state.fill(0);

		self.hp_filter_state.fill(0);

		self.mean_value.fill(1600);
	}

	fn gmm(&mut self, frame_len: usize) -> Result<bool, Error> {
		const MIN_ENERGY: i16 = 10;

		let (overhead1, overhead2, individual_test, total_test) = match frame_len {
			80 => (self.profile.overhang_max_1[0], self.profile.overhang_max_2[0], self.profile.local_threshold[0], self.profile.global_threshold[0]),
			160 => (self.profile.overhang_max_1[1], self.profile.overhang_max_2[1], self.profile.local_threshold[1], self.profile.global_threshold[1]),
			240 => (self.profile.overhang_max_1[2], self.profile.overhang_max_2[2], self.profile.local_threshold[2], self.profile.global_threshold[2]),
			_ => {
				return Err(Error::InvalidFrameSize {
					frame_size: frame_len,
					valid_sizes: [80, 160, 240]
				});
			}
		};

		let mut delta_noise = [[0; 6]; NUM_GAUSSIANS];
		let mut delta_speech = [[0; 6]; NUM_GAUSSIANS];
		let mut noise_probability = [0; NUM_GAUSSIANS];
		let mut speech_probability = [0; NUM_GAUSSIANS];
		let mut sum_log_likelihood_ratios = 0;
		let mut ngprvec = [[0; 6]; NUM_GAUSSIANS];
		let mut sgprvec = [[0; 6]; NUM_GAUSSIANS];
		let mut vadflag = false;

		if self.total_power > MIN_ENERGY {
			// The signal power of current frame is large enough for processing. The
			// processing consists of two parts:
			// 1) Calculating the likelihood of speech and thereby a VAD decision.
			// 2) Updating the underlying model, w.r.t., the decision made.

			// The detection scheme is an LRT with hypothesis
			// H0: Noise
			// H1: Speech
			//
			// We combine a global LRT with local tests, for each frequency sub-band,
			// here defined as |channel|.
			for channel in 0..6 {
				// For each channel we model the probability with a GMM consisting of
				// |kNumGaussians|, with different means and standard deviations depending
				// on H0 or H1.
				let mut h0_test = 0;
				let mut h1_test = 0;
				for gaussian in 0..NUM_GAUSSIANS {
					// Probability under H0, that is, probability of frame being noise.
					// Value given in Q27 = Q7 * Q20.
					let (tmp1_s32, delta) =
						gaussian_probability(self.feature_vector[channel], self.noise_means[gaussian][channel], self.noise_stds[gaussian][channel]);
					delta_noise[gaussian][channel] = delta;
					noise_probability[gaussian] = self.model.noise_weights[gaussian][channel] as i32 * tmp1_s32;
					h0_test += noise_probability[gaussian]; // Q27

					// Probability under H1, that is, probability of frame being speech.
					// Value given in Q27 = Q7 * Q20.
					let (tmp1_s32, delta) =
						gaussian_probability(self.feature_vector[channel], self.speech_means[gaussian][channel], self.speech_stds[gaussian][channel]);
					delta_speech[gaussian][channel] = delta;
					speech_probability[gaussian] = self.model.speech_weights[gaussian][channel] as i32 * tmp1_s32;
					h1_test += speech_probability[gaussian]; // Q27
				}

				// Calculate the log likelihood ratio: log2(Pr{X|H1} / Pr{X|H1}).
				// Approximation:
				// log2(Pr{X|H1} / Pr{X|H1}) = log2(Pr{X|H1}*2^Q) - log2(Pr{X|H1}*2^Q)
				//                           = log2(h1_test) - log2(h0_test)
				//                           = log2(2^(31-shifts_h1)*(1+b1))
				//                             - log2(2^(31-shifts_h0)*(1+b0))
				//                           = shifts_h0 - shifts_h1
				//                             + log2(1+b1) - log2(1+b0)
				//                          ~= shifts_h0 - shifts_h1
				//
				// Note that b0 and b1 are values less than 1, hence, 0 <= log2(1+b0) < 1.
				// Further, b0 and b1 are independent and on the average the two terms
				// cancel.
				let (mut shifts_h0, mut shifts_h1) = (norm_i32(h0_test), norm_i32(h1_test));
				if h0_test == 0 {
					shifts_h0 = 31;
				}
				if h1_test == 0 {
					shifts_h1 = 31;
				}
				let log_likelihood_ratio = shifts_h0 - shifts_h1;

				// Update |sum_log_likelihood_ratios| with spectrum weighting. This is
				// used for the global VAD decision.
				sum_log_likelihood_ratios += log_likelihood_ratio as i32 * SPECTRUM_WEIGHT[channel] as i32;

				// Local VAD decision.
				if (log_likelihood_ratio as i16 * 4) > individual_test {
					vadflag = true;
				}

				// Calculate local noise probabilities used later when updating the GMM.
				let h0 = (h0_test >> 12) as i16;
				if h0 > 0 {
					// High probability of noise. Assign conditional probabilities for each
					// Gaussian in the GMM.
					let tmp1_s32 = (noise_probability[0] & 0xFFFFF000u32 as i32) << 2; // Q29
					ngprvec[0][channel] = div_i32_i16(tmp1_s32, h0) as i16; // Q14
					for gaussian in 1..NUM_GAUSSIANS {
						ngprvec[gaussian][channel] = 16384 - ngprvec[0][channel];
					}
				} else {
					// Low noise probability. Assign conditional probability 1 to the first
					// Gaussian and 0 to the rest (which is already set at initialization).
					ngprvec[0][channel] = 16384;
				}

				// Calculate local speech probabilities used later when updating the GMM.
				let h1 = (h1_test >> 12) as i16;
				if h1 > 0 {
					// High probability of speech. Assign conditional probabilities for each
					// Gaussian in the GMM. Otherwise use the initialized values, i.e., 0.
					let tmp1_s32 = (speech_probability[0] & 0xFFFFF000u32 as i32) << 2; // Q29
					sgprvec[0][channel] = div_i32_i16(tmp1_s32, h1) as i16; // Q14
					for gaussian in 1..NUM_GAUSSIANS {
						sgprvec[gaussian][channel] = 16384 - sgprvec[0][channel];
					}
				}
			}

			// Make a global VAD decision.
			vadflag |= sum_log_likelihood_ratios >= total_test as i32;

			// Update the model parameters.
			let mut maxspe = 12800;
			for channel in 0..6 {
				// Get minimum value in past which is used for long term correction in Q4.
				let feature_minimum = find_minimum(
					&mut self.age_vector,
					&mut self.low_value_vector,
					self.frame_counter,
					&mut self.mean_value,
					self.feature_vector[channel],
					channel
				);

				// Compute the "global" mean, that is the sum of the two means weighted.
				let noise_global_mean = weighted_average(&mut self.noise_means, channel, 0, &self.model.noise_weights);
				let tmp1_s16 = (noise_global_mean >> 6) as i16; // Q8
				for gaussian in 0..NUM_GAUSSIANS {
					let nmk = self.noise_means[gaussian][channel];
					let smk = self.speech_means[gaussian][channel];
					let nsk = self.noise_stds[gaussian][channel];
					let ssk = self.speech_stds[gaussian][channel];

					// Update noise mean vector if the frame consists of noise only.
					let mut nmk2 = nmk;
					if !vadflag {
						// (Q14 * Q11 >> 11) = Q14.
						let delt = ((ngprvec[gaussian][channel] as i32 * delta_noise[gaussian][channel] as i32) >> 11) as i16;
						// Q7 + (Q14 * Q15 >> 22) = Q7.
						nmk2 = nmk + ((delt as i32 * NOISE_UPDATE as i32) >> 22) as i16;
					}

					// Long term correction of the noise mean.
					// Q8 - Q8 = Q8.
					let ndelt = (feature_minimum << 4) - tmp1_s16;
					// Q7 + (Q8 * Q8) >> 9 = Q7.
					let nmk3 = nmk2 + ((ndelt as i32 * BACK_ETA as i32) >> 9) as i16;

					// Control that the noise mean does not drift too much.
					self.noise_means[gaussian][channel] = nmk3.clamp((gaussian + 5 << 7) as i16, ((72 + gaussian - channel) << 7) as i16);

					if vadflag {
						// Update speech mean vector:
						// |deltaS| = (x-mu)/sigma^2
						// sgprvec[k] = |speech_probability[k]| /
						//   (|speech_probability[0]| + |speech_probability[1]|)

						// (Q14 * Q11) >> 11 = Q14.
						let delt = ((sgprvec[gaussian][channel] as i32 * delta_speech[gaussian][channel] as i32) >> 11) as i16;
						// Q14 * Q15 >> 21 = Q8
						let tmp_s16 = ((delt as i32 * SPEECH_UPDATE as i32) >> 21) as i16;
						// Q7 + (Q8 >> 1) = Q7
						let smk2 = smk + ((tmp_s16 + 1) >> 1);

						// Control that the max speech mean does not drift too much.
						self.speech_means[gaussian][channel] = smk2.clamp(self.model.minimum_mean[gaussian], maxspe + 640);

						// (Q7 >> 3) = Q4
						let mut tmp_s16 = (smk + 4) >> 3;

						tmp_s16 = self.feature_vector[channel] - tmp_s16;
						// (Q11 * Q4 >> 3) = Q12
						let tmp1_s32 = (delta_speech[gaussian][channel] as i32 * tmp_s16 as i32) >> 3;
						let tmp2_s32 = tmp1_s32 - 4096;
						tmp_s16 = (sgprvec[gaussian][channel] >> 2) as i16;
						// (Q14 >> 2) * Q12 = Q24.
						let tmp1_s32 = tmp_s16 as i32 * tmp2_s32;

						let tmp2_s32 = tmp1_s32 >> 4; // Q20

						// 0.1 * Q20 / Q7 = Q13
						let mut tmp_s16 = if tmp2_s32 > 0 { div_i32_i16(tmp2_s32, ssk * 10) } else { -div_i32_i16(-tmp2_s32, ssk * 10) } as i16;

						// Divide by 4 giving an update factor of 0.025 (= 0.1 / 4).
						// Note that division by 4 equals shift by 2, hence,
						// (Q13 >> 8) = (Q13 >> 6) / 4 = Q7.
						tmp_s16 += 128;
						let ssk = (ssk + (tmp_s16 >> 8)).min(MIN_STD);
						self.speech_stds[gaussian][channel] = ssk;
					} else {
						// Update GMM variance vectors.
						// deltaN * (features[channel] - nmk) - 1
						// Q4 - (Q7 >> 3) = Q4.
						let tmp_s16 = self.feature_vector[channel] - (nmk >> 3);
						// (Q11 * Q4 >> 3) = Q12.
						let mut tmp1_s32 = (delta_noise[gaussian][channel] as i32 * tmp_s16 as i32) >> 3;
						tmp1_s32 -= 4096;

						// (Q14 >> 2) * Q12 = Q24.
						let tmp_s16 = (ngprvec[gaussian][channel] + 2) >> 2;
						let tmp2_s32 = (tmp_s16 as i32).saturating_mul(tmp1_s32);
						// Q20 * approx 0.001 (2^-10=0.0009766), hence,
						// (Q24 >> 14) = (Q24 >> 4) / 2^10 = Q20.
						tmp1_s32 = tmp2_s32 >> 14;

						// Q20 / Q7 = Q13.
						let mut tmp_s16 = if tmp1_s32 > 0 { div_i32_i16(tmp1_s32, nsk) } else { -div_i32_i16(-tmp1_s32, nsk) } as i16;
						tmp_s16 += 32; // Rounding
						let nsk = (nsk + (tmp_s16 >> 6)).min(MIN_STD);
						self.noise_stds[gaussian][channel] = nsk;
					}
				}

				// Separate models if they are too close.
				// |noise_global_mean| in Q14 (= Q7 * Q7).
				let mut noise_global_mean = weighted_average(&mut self.noise_means, channel, 0, &self.model.noise_weights);
				// |speech_global_mean| in Q14 (= Q7 * Q7).
				let mut speech_global_mean = weighted_average(&mut self.speech_means, channel, 0, &self.model.speech_weights);

				// |diff| = "global" speech mean - "global" noise mean.
				// (Q14 >> 9) - (Q14 >> 9) = Q5.
				let diff = (speech_global_mean >> 9) as i16 - (noise_global_mean >> 9) as i16;
				if diff < MINIMUM_DIFFERENCE[channel] {
					let tmp_s16 = MINIMUM_DIFFERENCE[channel] - diff;

					// |tmp1_s16| = ~0.8 * (kMinimumDifference - diff) in Q7.
					// |tmp2_s16| = ~0.2 * (kMinimumDifference - diff) in Q7.
					let tmp1_s16 = (13 * tmp_s16) >> 2;
					let tmp2_s16 = (3 * tmp_s16) >> 2;

					// Move Gaussian means for speech model by |tmp1_s16| and update
					// |speech_global_mean|. Note that |self->speech_means[channel]| is
					// changed after the call.
					speech_global_mean = weighted_average(&mut self.speech_means, channel, tmp1_s16, &self.model.speech_weights);

					// Move Gaussian means for noise model by -|tmp2_s16| and update
					// |noise_global_mean|. Note that |self->noise_means[channel]| is
					// changed after the call.
					noise_global_mean = weighted_average(&mut self.noise_means, channel, -tmp2_s16, &self.model.noise_weights);
				}

				// Control that the speech & noise means do not drift to much.
				maxspe = MAXIMUM_SPEECH[channel];
				let mut tmp2_s16 = (speech_global_mean >> 7) as i16;
				if tmp2_s16 > maxspe {
					// Upper limit of speech model.
					tmp2_s16 -= maxspe;

					for gaussian in 0..NUM_GAUSSIANS {
						self.speech_means[gaussian][channel] -= tmp2_s16;
					}
				}

				tmp2_s16 = (noise_global_mean >> 7) as i16;
				if tmp2_s16 > self.model.maximum_noise[channel] {
					tmp2_s16 -= self.model.maximum_noise[channel];
					for gaussian in 0..NUM_GAUSSIANS {
						self.noise_means[gaussian][channel] -= tmp2_s16;
					}
				}
			}

			self.frame_counter += 1;
		}

		if !vadflag {
			if self.overhang > 0 {
				vadflag = true;
				self.overhang -= 1;
			}
			self.num_of_speech = 0;
		} else {
			self.num_of_speech += 1;
			if self.num_of_speech > MAX_SPEECH_FRAMES as i16 {
				self.num_of_speech = MAX_SPEECH_FRAMES as i16;
				self.overhang = overhead2;
			} else {
				self.overhang = overhead1;
			}
		}

		Ok(vadflag)
	}

	/// Run VAD prediction on a single frame of 48 KHz signed 16-bit mono PCM audio. Returns `Ok(true)` if the model
	/// predicts that this frame contains speech.
	///
	/// The frame must be 10ms (480 samples), 20ms (960 samples), or 30ms (1440 samples) in length. An `Err` is returned
	/// if the frame size is invalid.
	///
	/// ```
	/// # use earshot::{VoiceActivityDetector, VoiceActivityProfile};
	/// let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
	///
	/// # let mut stream = std::iter::once(vec![0; 960]);
	/// while let Some(frame) = stream.next() {
	/// 	let is_speech_detected = vad.predict_48khz(&frame).unwrap();
	/// 	# assert_eq!(is_speech_detected, false);
	/// }
	/// ```
	pub fn predict_48khz(&mut self, frame: &[i16]) -> Result<bool, Error> {
		let mut out_frame = [0; 240];
		for (i, subframe) in frame.chunks_exact(480).enumerate() {
			let out_chunk_size = 80 * i;
			resample_48khz_to_8khz(
				subframe,
				&mut out_frame[out_chunk_size..out_chunk_size + 80],
				&mut self.downsample_48khz_to_8khz_state,
				&mut self.downsample_tmp
			);
		}
		self.predict_8khz(&out_frame).map_err(|e| e.invalid_frame_size_multiplier(6))
	}

	/// Run VAD prediction on a single frame of 32 KHz signed 16-bit mono PCM audio. Returns `Ok(true)` if the model
	/// predicts that this frame contains speech.
	///
	/// The frame must be 10ms (320 samples), 20ms (640 samples), or 30ms (960 samples) in length. An `Err` is returned
	/// if the frame size is invalid.
	///
	/// ```
	/// # use earshot::{VoiceActivityDetector, VoiceActivityProfile};
	/// let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
	///
	/// # let mut stream = std::iter::once(vec![0; 640]);
	/// while let Some(frame) = stream.next() {
	/// 	let is_speech_detected = vad.predict_32khz(&frame).unwrap();
	/// 	# assert_eq!(is_speech_detected, false);
	/// }
	/// ```
	pub fn predict_32khz(&mut self, frame: &[i16]) -> Result<bool, Error> {
		let mut out_frame = [0; 480];
		downsample_2x(frame, &mut out_frame, &mut self.downsampling_filter_states[2..]);
		self.predict_16khz(&out_frame).map_err(|e| e.invalid_frame_size_multiplier(2))
	}

	/// Run VAD prediction on a single frame of 16 KHz signed 16-bit mono PCM audio. Returns `Ok(true)` if the model
	/// predicts that this frame contains speech.
	///
	/// The frame must be 10ms (160 samples), 20ms (320 samples), or 30ms (480 samples) in length. An `Err` is returned
	/// if the frame size is invalid.
	///
	/// ```
	/// # use earshot::{VoiceActivityDetector, VoiceActivityProfile};
	/// let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
	///
	/// # let mut stream = std::iter::once(vec![0; 320]);
	/// while let Some(frame) = stream.next() {
	/// 	let is_speech_detected = vad.predict_16khz(&frame).unwrap();
	/// 	# assert_eq!(is_speech_detected, false);
	/// }
	/// ```
	pub fn predict_16khz(&mut self, frame: &[i16]) -> Result<bool, Error> {
		let mut out_frame = [0; 240];
		downsample_2x(frame, &mut out_frame, &mut self.downsampling_filter_states[0..2]);
		self.predict_8khz(&out_frame).map_err(|e| e.invalid_frame_size_multiplier(2))
	}

	/// Run VAD prediction on a single frame of 8 KHz signed 16-bit mono PCM audio. Returns `Ok(true)` if the model
	/// predicts that this frame contains speech.
	///
	/// The frame must be 10ms (80 samples), 20ms (160 samples), or 30ms (240 samples) in length. An `Err` is returned
	/// if the frame size is invalid.
	///
	/// ```
	/// # use earshot::{VoiceActivityDetector, VoiceActivityProfile};
	/// let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
	///
	/// # let mut stream = std::iter::once(vec![0; 160]);
	/// while let Some(frame) = stream.next() {
	/// 	let is_speech_detected = vad.predict_8khz(&frame).unwrap();
	/// 	# assert_eq!(is_speech_detected, false);
	/// }
	/// ```
	pub fn predict_8khz(&mut self, frame: &[i16]) -> Result<bool, Error> {
		self.total_power = calculate_features(
			frame,
			&mut self.feature_vector,
			&mut self.split1_data,
			&mut self.split2_data,
			&mut self.upper_state,
			&mut self.lower_state,
			&mut self.hp_filter_state
		);
		self.gmm(frame.len())
	}
}

#[cfg(test)]
mod tests {
	use core::slice;
	use std::fs;

	use crate::{Error, VoiceActivityDetector, VoiceActivityProfile};

	#[test]
	fn test_vad_synthetic() -> Result<(), Error> {
		for profile in [
			VoiceActivityProfile::QUALITY,
			VoiceActivityProfile::LBR,
			VoiceActivityProfile::AGGRESSIVE,
			VoiceActivityProfile::VERY_AGGRESSIVE
		] {
			let mut vad = VoiceActivityDetector::new(profile.clone());
			for frame_size in [80, 160, 240] {
				let speech = vec![0; frame_size];
				assert_eq!(false, vad.predict_8khz(&speech)?);
			}
		}
		for profile in [
			VoiceActivityProfile::QUALITY,
			VoiceActivityProfile::LBR,
			VoiceActivityProfile::AGGRESSIVE,
			VoiceActivityProfile::VERY_AGGRESSIVE
		] {
			let mut vad = VoiceActivityDetector::new(profile.clone());
			for frame_size in [80, 160, 240] {
				let speech = (0..frame_size as i16).map(|i| i.wrapping_mul(i)).collect::<Vec<_>>();
				assert_eq!(true, vad.predict_8khz(&speech)?);
			}
		}
		Ok(())
	}

	#[test]
	fn test_real_8khz() -> Result<(), Error> {
		let file = fs::read("tests/data/audio_tiny8.raw").unwrap();
		let samples = unsafe { slice::from_raw_parts(file.as_ptr().cast::<i16>(), file.len() / 2) };
		let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
		let mut voice_frames = 0;
		for frame in samples.chunks_exact(240) {
			if vad.predict_8khz(frame)? {
				voice_frames += 1;
			}
		}
		assert!((162..169).contains(&voice_frames));
		Ok(())
	}

	#[test]
	fn test_real_16khz() -> Result<(), Error> {
		let file = fs::read("tests/data/audio_tiny16.raw").unwrap();
		let samples = unsafe { slice::from_raw_parts(file.as_ptr().cast::<i16>(), file.len() / 2) };
		let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
		let mut voice_frames = 0;
		for frame in samples.chunks_exact(480) {
			if vad.predict_16khz(frame)? {
				voice_frames += 1;
			}
		}
		assert!((162..169).contains(&voice_frames));
		Ok(())
	}

	#[test]
	fn test_real_32khz() -> Result<(), Error> {
		let file = fs::read("tests/data/audio_tiny32.raw").unwrap();
		let samples = unsafe { slice::from_raw_parts(file.as_ptr().cast::<i16>(), file.len() / 2) };
		let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
		let mut voice_frames = 0;
		for frame in samples.chunks_exact(960) {
			if vad.predict_32khz(frame)? {
				voice_frames += 1;
			}
		}
		assert!((162..169).contains(&voice_frames));
		Ok(())
	}

	#[test]
	fn test_real_48khz() -> Result<(), Error> {
		let file = fs::read("tests/data/audio_tiny48.raw").unwrap();
		let samples = unsafe { slice::from_raw_parts(file.as_ptr().cast::<i16>(), file.len() / 2) };
		let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);
		let mut voice_frames = 0;
		for frame in samples.chunks_exact(1440) {
			if vad.predict_48khz(frame)? {
				voice_frames += 1;
			}
		}
		assert!((162..169).contains(&voice_frames));
		Ok(())
	}
}
