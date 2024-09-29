// Downsampling filter based on splitting filter and allpass functions.
pub fn downsample_2x(inv: &[i16], out: &mut [i16], filter_state: &mut [i32]) {
	const UPPER_COEFFICIENT: i32 = 5243; // 0.64 in Q13
	const LOWER_COEFFICIENT: i32 = 1392; // 0.17 in Q13

	let half_len = inv.len() >> 1;
	let mut tmpd1 = filter_state[0];
	let mut tmpd2 = filter_state[1];
	// Filter coefficients in Q13, filter state in Q0.
	for n in 0..half_len {
		// All-pass filtering upper branch.
		let tmpw1 = ((tmpd1 >> 1) + ((UPPER_COEFFICIENT * inv[n * 2] as i32) >> 14)) as i16;
		out[n] = tmpw1;
		tmpd1 = (inv[n * 2] as i32) - ((UPPER_COEFFICIENT * tmpw1 as i32) >> 12);

		// All-pass filtering lower branch.
		let tmpw2 = ((tmpd2 >> 1) + ((LOWER_COEFFICIENT * inv[n * 2 + 1] as i32) >> 14)) as i16;
		out[n] = out[n].saturating_add(tmpw2); // C source originally wrapped but that introduced popping
		tmpd2 = (inv[n * 2 + 1] as i32) - ((LOWER_COEFFICIENT * tmpw2 as i32) >> 12);
	}
	filter_state[0] = tmpd1;
	filter_state[1] = tmpd2;
}

// Inserts |feature_value| into |low_value_vector|, if it is one of the 16
// smallest values the last 100 frames. Then calculates and returns the median
// of the five smallest values.
pub fn find_minimum(age: &mut [i16], low_value: &mut [i16], frame_counter: usize, mean_values: &mut [i16], feature_value: i16, channel: usize) -> i16 {
	let offset = channel << 4;
	let age = &mut age[offset..offset + 16];
	let smallest_values = &mut low_value[offset..offset + 16];

	// Each value in |smallest_values| is getting 1 loop older. Update |age|, and
	// remove old values.
	for i in 0..16 {
		if age[i] != 100 {
			age[i] += 1;
		} else {
			// Too old value. Remove from memory and shift larger values downwards.
			for j in i..15 {
				smallest_values[j] = smallest_values[j + 1];
				age[j] = age[j + 1];
			}
			age[15] = 101;
			smallest_values[15] = 10000;
		}
	}

	// Check if |feature_value| is smaller than any of the values in
	// |smallest_values|. If so, find the |position| where to insert the new value
	// (|feature_value|).
	let position = if feature_value < smallest_values[7] {
		if feature_value < smallest_values[3] {
			if feature_value < smallest_values[1] {
				if feature_value < smallest_values[0] { 0 } else { 1 }
			} else if feature_value < smallest_values[2] {
				2
			} else {
				3
			}
		} else if feature_value < smallest_values[5] {
			if feature_value < smallest_values[4] { 4 } else { 5 }
		} else if feature_value < smallest_values[6] {
			6
		} else {
			7
		}
	} else if feature_value < smallest_values[15] {
		if feature_value < smallest_values[11] {
			if feature_value < smallest_values[9] {
				if feature_value < smallest_values[8] { 8 } else { 9 }
			} else if feature_value < smallest_values[10] {
				10
			} else {
				11
			}
		} else if feature_value < smallest_values[13] {
			if feature_value < smallest_values[12] { 12 } else { 13 }
		} else if feature_value < smallest_values[14] {
			14
		} else {
			15
		}
	} else {
		-1
	};

	// If we have detected a new small value, insert it at the correct position
	// and shift larger values up.
	if position > -1 {
		let position = position as usize;
		let mut i = 15;
		while i > position {
			smallest_values[i] = smallest_values[i - 1];
			age[i] = age[i - 1];
			i -= 1;
		}
		smallest_values[position] = feature_value;
		age[position] = 1;
	}

	let current_median = if frame_counter > 2 {
		smallest_values[2]
	} else if frame_counter > 0 {
		smallest_values[0]
	} else {
		1600
	};

	// Smooth the median value.
	const SMOOTHING_DOWN: i16 = 6553; // 0.2 in Q15
	const SMOOTHING_UP: i16 = 32439; // 0.99 in Q15
	let alpha = if frame_counter > 0 {
		if current_median < mean_values[channel] { SMOOTHING_DOWN } else { SMOOTHING_UP }
	} else {
		0
	};

	let smoothed = ((alpha + 1) as i32 * mean_values[channel] as i32) + ((i16::MAX - alpha) as i32 * current_median as i32) + 16384;
	mean_values[channel] = (smoothed >> 15) as i16;
	mean_values[channel]
}

#[cfg(test)]
mod tests {
	use super::downsample_2x;
	use crate::sp::find_minimum;

	#[test]
	fn test_downsample() {
		let zeros = vec![0; 960];
		let mut out = vec![0; 960 / 2];
		let mut filter_state = [0, 0];
		downsample_2x(&zeros, &mut out, &mut filter_state);
		assert_eq!(filter_state[0], 0);
		assert_eq!(filter_state[1], 0);

		let inv = (0..960i16).map(|c| c.wrapping_mul(c)).collect::<Vec<_>>();
		downsample_2x(&inv, &mut out, &mut filter_state);
		assert_eq!(filter_state[0], 207);
		assert_eq!(filter_state[1], 2270);
	}

	#[test]
	fn test_min() {
		let reference = [
			1600, 720, 509, 512, 532, 552, 570, 588, 606, 624, 642, 659, 675, 691, 707, 723, 1600, 544, 502, 522, 542, 561, 579, 597, 615, 633, 651, 667, 683,
			699, 715, 731
		];
		let mut age = vec![0; 6 * 16].into_boxed_slice();
		let mut low_value = vec![10000; 6 * 16].into_boxed_slice();
		let mut mean_values = vec![1600; 6].into_boxed_slice();
		let mut frame_counter = 0;
		for i in 0..16 {
			let value = 500 * (i as i16 + 1);
			for j in 0..6 {
				assert_eq!(reference[i], find_minimum(&mut age, &mut low_value, frame_counter, &mut mean_values, value, j));
				assert_eq!(reference[i + 16], find_minimum(&mut age, &mut low_value, frame_counter, &mut mean_values, 12000, j));
			}
			frame_counter += 1;
		}
	}
}
