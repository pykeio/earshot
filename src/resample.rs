use core::{ptr, slice};

//   Resampling ratio: 2/3
// input:  i32 (normalized, not saturated) :: size (3 * K) + 6
// output: i32 (shifted 15 positions to the left, + offset 16384) :: size 2 * K
//      K: number of blocks
fn resample_48khz_to_32khz(inv: &[i32], out: &mut [i32]) {
	const COEFFICIENTS: [i32; 8] = [778, -2050, 1087, 23285, 12903, -3783, 441, 222];

	let block_size = out.len() / 2;
	assert!(inv.len() >= (block_size * 3) + 6);

	let mut ip = 0;
	let mut op = 0;
	for _ in 0..block_size {
		let mut tmp = 1 << 14;
		tmp += COEFFICIENTS[0] * inv[ip];
		tmp += COEFFICIENTS[1] * inv[ip + 1];
		tmp += COEFFICIENTS[2] * inv[ip + 2];
		tmp += COEFFICIENTS[3] * inv[ip + 3];
		tmp += COEFFICIENTS[4] * inv[ip + 4];
		tmp += COEFFICIENTS[5] * inv[ip + 5];
		tmp += COEFFICIENTS[6] * inv[ip + 6];
		tmp += COEFFICIENTS[7] * inv[ip + 7];
		out[op] = tmp;

		tmp = 1 << 14;
		tmp += COEFFICIENTS[7] * inv[ip + 1];
		tmp += COEFFICIENTS[6] * inv[ip + 2];
		tmp += COEFFICIENTS[5] * inv[ip + 3];
		tmp += COEFFICIENTS[4] * inv[ip + 4];
		tmp += COEFFICIENTS[3] * inv[ip + 5];
		tmp += COEFFICIENTS[2] * inv[ip + 6];
		tmp += COEFFICIENTS[1] * inv[ip + 7];
		tmp += COEFFICIENTS[0] * inv[ip + 8];
		out[op + 1] = tmp;

		ip += 3;
		op += 2;
	}
}

const RESAMPLE_ALLPASS_COEFFS: [i32; 6] = [3050, 9368, 15063, 821, 6110, 12382];

fn down_x2_i32_i16(inv: &mut [i32], out: &mut [i16], state: &mut [i32]) {
	let len = inv.len() >> 1;
	// lower allpass filter (operates on even input samples)
	for i in 0..len {
		let tmp0 = inv[i << 1];
		let mut diff = tmp0 - state[1];

		// scale down and round
		diff = (diff + (1 << 13)) >> 14;
		let tmp1 = state[0] + diff * RESAMPLE_ALLPASS_COEFFS[0];
		state[0] = tmp0;
		diff = tmp1 - state[2];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		let tmp0 = state[1] + diff * RESAMPLE_ALLPASS_COEFFS[1];
		state[1] = tmp1;
		diff = tmp0 - state[3];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		state[3] = state[2] + diff * RESAMPLE_ALLPASS_COEFFS[2];
		state[2] = tmp0;

		// divide by two and store temporarily
		inv[i << 1] = state[3] >> 1;
	}

	// upper allpass filter (operates on odd input samples)
	let inv2 = &mut inv[1..];
	for i in 0..len {
		let tmp0 = inv2[i << 1];
		let mut diff = tmp0 - state[5];
		// scale down and round
		diff = (diff + (1 << 13)) >> 14;
		let tmp1 = state[4] + diff * RESAMPLE_ALLPASS_COEFFS[3];
		state[4] = tmp0;
		diff = tmp1 - state[6];

		// scale down and round
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		let tmp0 = state[5] + diff * RESAMPLE_ALLPASS_COEFFS[4];
		state[5] = tmp1;
		diff = tmp0 - state[7];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		state[7] = state[6] + diff * RESAMPLE_ALLPASS_COEFFS[5];
		state[6] = tmp0;

		// divide by two and store temporarily
		inv2[i << 1] = state[7] >> 1;
	}

	// combine allpass outputs
	for i in (0..len).step_by(2) {
		// divide by two, add both allpass outputs and round
		let tmp0 = (inv[i << 1] + inv[(i << 1) + 1]) >> 15;
		let tmp1 = (inv[(i << 1) + 2] + inv[(i << 1) + 3]) >> 15;
		out[i] = tmp0.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
		out[i + 1] = tmp1.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
	}
}

fn down_x2_i16_i32(inv: &[i16], out: &mut [i32], state: &mut [i32]) {
	let len = inv.len() >> 1;

	// lower allpass filter (operates on even input samples)
	for i in 0..len {
		let tmp0 = ((inv[1 << 1] as i32) << 15) + (1 << 14);
		let mut diff = tmp0 - state[1];
		// scale down and round
		diff = (diff + (1 << 13)) >> 14;
		let tmp1 = state[0] + diff * RESAMPLE_ALLPASS_COEFFS[0];
		state[0] = tmp0;
		diff = tmp1 - state[2];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		let tmp0 = state[1] + diff * RESAMPLE_ALLPASS_COEFFS[1];
		state[1] = tmp1;
		diff = tmp0 - state[3];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		state[3] = state[2] + diff * RESAMPLE_ALLPASS_COEFFS[2];
		state[2] = tmp0;

		out[i] = state[3] >> 1;
	}

	// upper allpass filter (operates on odd input samples)
	let inv2 = &inv[1..];
	for i in 0..len {
		let tmp0 = ((inv2[i << 1] as i32) << 15) + (1 << 14);
		let mut diff = tmp0 - state[5];
		// scale down and round
		diff = (diff + (1 << 13)) >> 14;
		let tmp1 = state[4] + diff * RESAMPLE_ALLPASS_COEFFS[3];
		state[4] = tmp0;
		diff = tmp1 - state[6];

		// scale down and round
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		let tmp0 = state[5] + diff * RESAMPLE_ALLPASS_COEFFS[4];
		state[5] = tmp1;
		diff = tmp0 - state[7];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		state[7] = state[6] + diff * RESAMPLE_ALLPASS_COEFFS[5];
		state[6] = tmp0;

		out[i] += state[7] >> 1;
	}
}

fn lowpass_2x_i32_i32(inv: &[i32], out: &mut [i32], state: &mut [i32]) {
	let len = inv.len() >> 1;

	let inv1 = &inv[1..];
	let mut v0 = state[12];
	for i in 0..len {
		let mut diff = v0 - state[1];
		// scale down and round
		diff = (diff + (1 << 13)) >> 14;
		let tmp1 = state[0] + diff * RESAMPLE_ALLPASS_COEFFS[0];
		state[0] = v0;
		diff = tmp1 - state[2];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		let tmp0 = state[1] + diff * RESAMPLE_ALLPASS_COEFFS[1];
		state[1] = tmp1;
		diff = tmp0 - state[3];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		state[3] = state[2] + diff * RESAMPLE_ALLPASS_COEFFS[2];
		state[2] = tmp0;

		// scale down, round and store
		out[i << 1] = state[3] >> 1;
		v0 = inv1[i << 1];
	}

	// upper allpass filter: even input -> even output samples
	for i in 0..len {
		let tmp0 = inv[i << 1];
		let mut diff = tmp0 - state[5];
		// scale down and round
		diff = (diff + (1 << 13)) >> 14;
		let tmp1 = state[4] + diff * RESAMPLE_ALLPASS_COEFFS[3];
		state[4] = tmp0;
		diff = tmp1 - state[6];

		// scale down and round
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		let tmp0 = state[5] + diff * RESAMPLE_ALLPASS_COEFFS[4];
		state[5] = tmp1;
		diff = tmp0 - state[7];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		state[7] = state[6] + diff * RESAMPLE_ALLPASS_COEFFS[5];
		state[6] = tmp0;

		// average the two allpass outputs, scale down and store
		out[i << 1] = (out[i << 1] + (state[7] >> 1)) >> 15;
	}

	// lower allpass filter: even input -> odd output samples
	let out = &mut out[1..];
	for i in 0..len {
		let tmp0 = inv[i << 1];
		let mut diff = tmp0 - state[9];
		// scale down and round
		diff = (diff + (1 << 13)) >> 14;
		let tmp1 = state[8] + diff * RESAMPLE_ALLPASS_COEFFS[0];
		state[8] = tmp0;
		diff = tmp1 - state[10];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		let tmp0 = state[9] + diff * RESAMPLE_ALLPASS_COEFFS[1];
		state[9] = tmp1;
		diff = tmp0 - state[11];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		state[11] = state[10] + diff * RESAMPLE_ALLPASS_COEFFS[2];
		state[10] = tmp0;

		// scale down, round and store
		out[i << 1] = state[11] >> 1;
	}

	// upper allpass filter: odd input -> odd output samples
	let inv = &inv[1..];
	for i in 0..len {
		let tmp0 = inv[i << 1];
		let mut diff = tmp0 - state[13];
		// scale down and round
		diff = (diff + (1 << 13)) >> 14;
		let tmp1 = state[12] + diff * RESAMPLE_ALLPASS_COEFFS[3];
		state[12] = tmp0;
		diff = tmp1 - state[14];

		// scale down and round
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		let tmp0 = state[13] + diff * RESAMPLE_ALLPASS_COEFFS[4];
		state[13] = tmp1;
		diff = tmp0 - state[15];

		// scale down and truncate
		diff >>= 14;
		if diff < 0 {
			diff += 1;
		}
		state[15] = state[14] + diff * RESAMPLE_ALLPASS_COEFFS[5];
		state[14] = tmp0;

		// average the two allpass outputs, scale down and store
		out[i << 1] = (out[i << 1] + (state[15] >> 1)) >> 15;
	}
}

pub fn resample_48khz_to_8khz(inv: &[i16], out: &mut [i16], states: &mut [i32], tmp: &mut [i32]) {
	assert_eq!(tmp.len(), 480 + 256);
	assert_eq!(inv.len(), 480);

	down_x2_i16_i32(inv, &mut tmp[256..], &mut states[0..8]);
	lowpass_2x_i32_i32(&tmp[256..256 + 240], unsafe { slice::from_raw_parts_mut(tmp.as_ptr().cast_mut().add(16), 240) }, &mut states[8..24]);
	// uses states[24..32] as a buffer to store the last 8 samples to put at the beginning of the next frame, because
	// `resample_48khz_to_32khz` wants to read a couple samples beyond the frame
	unsafe {
		ptr::copy_nonoverlapping(states.as_ptr().add(24), tmp.as_mut_ptr().add(8), 8);
		ptr::copy_nonoverlapping(tmp.as_ptr().add(248), states.as_mut_ptr().add(24), 8);
	};
	// obviously horrible in many ways
	resample_48khz_to_32khz(&tmp[8..256], unsafe { slice::from_raw_parts_mut(tmp.as_ptr().cast_mut(), 160) });
	down_x2_i32_i16(&mut tmp[..160], out, &mut states[32..40]);
}

#[cfg(test)]
mod tests {
	use super::resample_48khz_to_32khz;

	#[test]
	fn test_resample48_saturated() {
		#[rustfmt::skip]
		let vector_saturated = [
			-32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
			-32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
			-32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
			32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
			32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
			32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
			32767, 32767, 32767, 32767, 32767, 32767
		];

		let ref1 = -1077493760;
		let ref2 = 1077493645;

		let mut out_vector = [0; 2 * 16];

		resample_48khz_to_32khz(&vector_saturated, &mut out_vector);

		// values at position 12-15 are skipped to account for the filter lag.
		for i in 0..12 {
			assert_eq!(out_vector[i], ref1);
		}
		for i in 16..32 {
			assert_eq!(out_vector[i], ref2);
		}
	}
}
