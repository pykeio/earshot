use crate::{
	energy::{energy, EnergyResult},
	util::norm_u32
};

// High pass filtering, with a cut-off frequency at 80 Hz, if the |data_in| is
// sampled at 500 Hz.
//
// - data_in      [i]   : Input audio data sampled at 500 Hz.
// - data_length  [i]   : Length of input and output data.
// - filter_state [i/o] : State of the filter.
// - data_out     [o]   : Output audio data in the frequency interval 80 - 250 Hz.
fn highpass(inv: &[i16], out: &mut [i16], filter_state: &mut [i16]) {
	const COEFFS_ZERO: [i32; 3] = [6631, -13262, 6631];
	const COEFFS_POLE: [i32; 2] = [-7756, 5620];
	assert_eq!(inv.len(), out.len());
	// The sum of the absolute values of the impulse response:
	// The zero/pole-filter has a max amplification of a single sample of: 1.4546
	// Impulse response: 0.4047 -0.6179 -0.0266  0.1993  0.1035  -0.0194
	// The all-zero section has a max amplification of a single sample of: 1.6189
	// Impulse response: 0.4047 -0.8094  0.4047  0       0        0
	// The all-pole section has a max amplification of a single sample of: 1.9931
	// Impulse response: 1.0000  0.4734 -0.1189 -0.2187 -0.0627   0.04532
	for i in 0..inv.len() {
		// All-zero section (filter coefficients in Q14).
		let mut t = COEFFS_ZERO[0] * inv[i] as i32;
		t += COEFFS_ZERO[1] * filter_state[0] as i32;
		t += COEFFS_ZERO[2] * filter_state[1] as i32;
		filter_state[1] = filter_state[0];
		filter_state[0] = inv[i];

		// All-pole section (filter coefficients in Q14).
		t -= COEFFS_POLE[0] * filter_state[2] as i32;
		t -= COEFFS_POLE[1] * filter_state[3] as i32;
		filter_state[3] = filter_state[2];
		filter_state[2] = (t >> 14) as i16;
		out[i] = filter_state[2];
	}
}

// All pass filtering of |data_in|, used before splitting the signal into two
// frequency bands (low pass vs high pass).
// Note that |data_in| and |data_out| can NOT correspond to the same address.
//
// - inv                [i]   : Input audio signal given in Q0.
// - out                [o]   : Output audio signal given in Q(-1).
// - filter_state       [i/o] : State of the filter given in Q(-1).
// - filter_coefficient [i]   : Given in Q15.
fn allpass(inv: &[i16], out: &mut [i16], filter_state: &mut i16, filter_coefficient: i16) {
	// The filter can only cause overflow (in the i16 output variable)
	// if more than 4 consecutive input numbers are of maximum value and
	// has the the same sign as the impulse responses first taps.
	// First 6 taps of the impulse response:
	// 0.6399 0.5905 -0.3779 0.2418 -0.1547 0.0990
	let mut state32 = (*filter_state as i32) * (1 << 16);
	let mut j = 0;
	for i in 0..out.len() {
		let tmp32 = state32 + filter_coefficient as i32 * inv[j] as i32;
		let tmp16 = (tmp32 >> 16) as i16;
		out[i] = tmp16;
		state32 = (inv[j] as i32 * (1 << 14)) - filter_coefficient as i32 * tmp16 as i32;
		state32 *= 2;
		j += 2;
	}
	*filter_state = (state32 >> 16) as i16;
}

// Splits |data_in| into |hp_data_out| and |lp_data_out| corresponding to
// an upper (high pass) part and a lower (low pass) part respectively.
//
// - inv          [i]   : Input audio data to be split into two frequency bands.
// - highpass_out [o]   : Output audio data of the upper half of the spectrum. The length is |data_length| / 2.
// - upper_state  [i/o] : State of the upper filter, given in Q(-1).
// - lowpass_out  [o]   : Output audio data of the lower half of the spectrum. The length is |data_length| / 2.
// - lower_state  [i/o] : State of the lower filter, given in Q(-1).
fn split(inv: &[i16], highpass_out: &mut [i16], upper_state: &mut i16, lowpass_out: &mut [i16], lower_state: &mut i16) {
	const HIGHPASS_COEFFICIENT: i16 = 20972; // 0.64 in Q15
	const LOWPASS_COEFFICIENT: i16 = 5571; // 0.17 in Q15
	assert_eq!(inv.len() / 2, highpass_out.len());
	assert_eq!(inv.len() / 2, lowpass_out.len());
	allpass(inv, highpass_out, upper_state, HIGHPASS_COEFFICIENT);
	allpass(&inv[1..], lowpass_out, lower_state, LOWPASS_COEFFICIENT);
	for i in 0..(inv.len() >> 1) {
		let tmp = highpass_out[i];
		// TODO: do we want this to be wrapping?
		highpass_out[i] = highpass_out[i].wrapping_sub(lowpass_out[i]);
		lowpass_out[i] = lowpass_out[i].wrapping_add(tmp);
	}
}

// Calculates the energy of |data_in| in dB, and also updates an overall
// |total_energy| if necessary.
//
// - data_in      [i]   : Input audio data for energy calculation.
// - data_length  [i]   : Length of input data.
// - offset       [i]   : Offset value added to |log_energy|.
// - total_energy [i/o] : An external energy updated with the energy of |data_in|. NOTE: |total_energy| is only updated
//   if |total_energy| <= |kMinEnergy|.
// - log_energy   [o]   : 10 * log10("energy of |data_in|") given in Q4.
// returns log_energy
fn log_of_energy(inv: &[i16], offset: i16, total_energy: &mut i16) -> i16 {
	const LOG_CONST: i16 = 24660; // 160*log10(2) in Q9
	let EnergyResult { energy, scaling_factor } = energy(inv);
	let mut energy = energy as u32;
	let mut scaling_factor = scaling_factor as i8;
	let mut log_energy = 0;
	if energy != 0 {
		// By construction, normalizing to 15 bits is equivalent with 17 leading
		// zeros of an unsigned 32 bit value.
		let normalizing_rshifts = 17 - norm_u32(energy) as i8;
		// In a 15 bit representation the leading bit is 2^14. log2(2^14) in Q10 is
		// (14 << 10), which is what we initialize |log2_energy| with. For a more
		// detailed derivations, see below.
		let mut log2_energy = 14336; // 14 in Q10

		scaling_factor += normalizing_rshifts;
		// Normalize |energy| to 15 bits.
		// |tot_rshifts| is now the total number of right shifts performed on
		// |energy| after normalization. This means that |energy| is in
		// Q(-tot_rshifts).
		if normalizing_rshifts < 0 {
			energy <<= -normalizing_rshifts;
		} else {
			energy >>= normalizing_rshifts;
		}

		// Calculate the energy of |data_in| in dB, in Q4.
		//
		// 10 * log10("true energy") in Q4 = 2^4 * 10 * log10("true energy") =
		// 160 * log10(|energy| * 2^|tot_rshifts|) =
		// 160 * log10(2) * log2(|energy| * 2^|tot_rshifts|) =
		// 160 * log10(2) * (log2(|energy|) + log2(2^|tot_rshifts|)) =
		// (160 * log10(2)) * (log2(|energy|) + |tot_rshifts|) =
		// |kLogConst| * (|log2_energy| + |tot_rshifts|)
		//
		// We know by construction that |energy| is normalized to 15 bits. Hence,
		// |energy| = 2^14 + frac_Q15, where frac_Q15 is a fractional part in Q15.
		// Further, we'd like |log2_energy| in Q10
		// log2(|energy|) in Q10 = 2^10 * log2(2^14 + frac_Q15) =
		// 2^10 * log2(2^14 * (1 + frac_Q15 * 2^-14)) =
		// 2^10 * (14 + log2(1 + frac_Q15 * 2^-14)) ~=
		// (14 << 10) + 2^10 * (frac_Q15 * 2^-14) =
		// (14 << 10) + (frac_Q15 * 2^-4) = (14 << 10) + (frac_Q15 >> 4)
		//
		// Note that frac_Q15 = (|energy| & 0x00003FFF)

		// Calculate and add the fractional part to |log2_energy|.
		log2_energy += ((energy & 0x00003FFF) >> 4) as i16;

		// |kLogConst| is in Q9, |log2_energy| in Q10 and |tot_rshifts| in Q0.
		// Note that we in our derivation above have accounted for an output in Q4.
		log_energy += (((LOG_CONST as i32 * log2_energy as i32) >> 19) + ((scaling_factor as i32 * LOG_CONST as i32) >> 9)) as i16;
		if log_energy < 0 {
			log_energy = 0;
		}
	} else {
		log_energy = offset;
		return log_energy;
	}

	log_energy += offset;

	// Update the approximate |total_energy| with the energy of |data_in|, if
	// |total_energy| has not exceeded |kMinEnergy|. |total_energy| is used as an
	// energy indicator in WebRtcVad_GmmProbability() in vad_core.c.
	if *total_energy <= 10 {
		if scaling_factor >= 0 {
			// We know by construction that the |energy| > |kMinEnergy| in Q0, so add
			// an arbitrary value such that |total_energy| exceeds |kMinEnergy|.
			*total_energy += 10 + 1;
		} else {
			// By construction |energy| is represented by 15 bits, hence any number of
			// right shifted |energy| will fit in an int16_t. In addition, adding the
			// value to |total_energy| is wrap around safe as long as
			// |kMinEnergy| < 8192.
			*total_energy += (energy >> -scaling_factor) as i16;
		}
	}

	log_energy
}

const FEATURES_OFFSET_VECTOR: [i16; 6] = [368, 368, 272, 176, 176, 176];

pub fn calculate_features(
	inv: &[i16],
	features: &mut [i16],
	split1_data: &mut [i16],
	split2_data: &mut [i16],
	upper_state: &mut [i16],
	lower_state: &mut [i16],
	hp_filter_state: &mut [i16]
) -> i16 {
	let split1_size = inv.len() >> 1;
	let split2_size = split1_size >> 1;
	assert!(split1_data.len() >= split1_size * 2);
	assert!(split2_data.len() >= split2_size * 2);
	let (hp_120, lp_120) = split1_data[..split1_size << 1].split_at_mut(split1_size);
	let (hp_60, lp_60) = split2_data[..split2_size << 1].split_at_mut(split2_size);
	let mut total_energy = 0;

	// Split at 2000 Hz and downsample.
	split(inv, hp_120, &mut upper_state[0], lp_120, &mut lower_state[0]);

	// For the upper band (2000 Hz - 4000 Hz), split at 3000 Hz and downsample.
	split(hp_120, hp_60, &mut upper_state[1], lp_60, &mut lower_state[1]);

	// Energy in 3000 Hz - 4000 Hz.
	features[5] = log_of_energy(&hp_60, FEATURES_OFFSET_VECTOR[5], &mut total_energy);
	// Energy in 2000 Hz - 3000 Hz.
	features[4] = log_of_energy(&lp_60, FEATURES_OFFSET_VECTOR[4], &mut total_energy);

	// For the lower band (0 Hz - 2000 Hz), split at 1000 Hz and downsample.
	split(lp_120, hp_60, &mut upper_state[2], lp_60, &mut lower_state[2]);

	// Energy in 1000 Hz - 2000 Hz.
	features[3] = log_of_energy(&hp_60, FEATURES_OFFSET_VECTOR[3], &mut total_energy);

	// For the lower band (0 Hz - 1000 Hz), split at 500 Hz and downsample.
	let hp_30 = &mut hp_120[..split2_size >> 1];
	let lp_30 = &mut lp_120[..split2_size >> 1];
	split(lp_60, hp_30, &mut upper_state[3], lp_30, &mut lower_state[3]);

	// Energy in 500 Hz - 1000 Hz.
	features[2] = log_of_energy(&hp_30, FEATURES_OFFSET_VECTOR[2], &mut total_energy);

	// For the lower band (0 Hz - 500 Hz) split at 250 Hz and downsample.
	let hp_15 = &mut hp_60[..split2_size >> 2];
	let lp_15 = &mut lp_60[..split2_size >> 2];
	split(lp_30, hp_15, &mut upper_state[4], lp_15, &mut lower_state[4]);

	// Energy in 250 Hz - 500 Hz.
	features[1] = log_of_energy(&hp_15, FEATURES_OFFSET_VECTOR[1], &mut total_energy);

	// Remove 0 Hz - 80 Hz by high pass filtering the lower band.
	highpass(&lp_15, hp_15, hp_filter_state);

	// Energy in 80 Hz - 250 Hz.
	features[0] = log_of_energy(&hp_15, FEATURES_OFFSET_VECTOR[0], &mut total_energy);

	total_energy
}

#[cfg(test)]
mod tests {
	use super::{calculate_features, FEATURES_OFFSET_VECTOR};

	#[test]
	fn test_calculate_features_reference() {
		const ENERGIES: [i16; 3] = [48, 11, 11];
		const REFERENCES: [[i16; 6]; 3] = [[1213, 759, 587, 462, 434, 272], [1479, 1385, 1291, 1200, 1103, 1099], [1732, 1692, 1681, 1629, 1436, 1436]];

		let mut features = [0; 6];
		let mut upper_state = [0; 5];
		let mut lower_state = [0; 5];
		let mut hp_filter_state = [0; 4];
		for (i, frame_length) in [80, 160, 240].into_iter().enumerate() {
			let speech = (0..frame_length).map(|i| (i as i16).wrapping_mul(i as i16)).collect::<Vec<_>>();
			let mut split1_data = vec![0; frame_length];
			let mut split2_data = vec![0; frame_length / 2];

			let total_energy =
				calculate_features(&speech, &mut features, &mut split1_data, &mut split2_data, &mut upper_state, &mut lower_state, &mut hp_filter_state);
			assert_eq!(total_energy, ENERGIES[i]);
			assert_eq!(features, REFERENCES[i]);
		}
	}

	#[test]
	fn test_calculate_features_zeros() {
		let mut features = [0; 6];
		let mut upper_state = [0; 5];
		let mut lower_state = [0; 5];
		let mut hp_filter_state = [0; 4];
		for frame_length in [80, 160, 240] {
			let speech = vec![0; frame_length];
			let mut split1_data = vec![0; frame_length];
			let mut split2_data = vec![0; frame_length / 2];

			let total_energy =
				calculate_features(&speech, &mut features, &mut split1_data, &mut split2_data, &mut upper_state, &mut lower_state, &mut hp_filter_state);
			assert_eq!(total_energy, 0);
			assert_eq!(features, FEATURES_OFFSET_VECTOR);
		}
	}

	#[test]
	fn test_calculate_features_ones() {
		for frame_length in [80, 160, 240] {
			let speech = vec![1; frame_length];
			let mut features = [0; 6];
			let mut upper_state = [0; 5];
			let mut lower_state = [0; 5];
			let mut hp_filter_state = [0; 4];
			let mut split1_data = vec![0; frame_length];
			let mut split2_data = vec![0; frame_length / 2];

			let total_energy =
				calculate_features(&speech, &mut features, &mut split1_data, &mut split2_data, &mut upper_state, &mut lower_state, &mut hp_filter_state);
			assert_eq!(total_energy, 0);
			assert_eq!(features, FEATURES_OFFSET_VECTOR);
		}
	}
}
