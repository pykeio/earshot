use crate::util::div_i32_i16;

// For a normal distribution, the probability of |input| is calculated and
// returned (in Q20). The formula for normal distributed probability is
//
// 1 / s * exp(-(x - m)^2 / (2 * s^2))
//
// where the parameters are given in the following Q domains:
// m = |mean| (Q7)
// s = |std| (Q7)
// x = |input| (Q4)
// in addition to the probability we output |delta| (in Q11) used when updating
// the noise/speech model.
//
// Returns (probability, delta)
pub fn gaussian_probability(input: i16, mean: i16, std: i16) -> (i32, i16) {
	const COMP_VAR: i32 = 22005;
	const LOG2_EXP: i32 = 5909; // log2(exp(1)) in Q12.

	// Calculate |inv_std| = 1 / s, in Q10.
	// 131072 = 1 in Q17, and (|std| >> 1) is for rounding instead of truncation.
	// Q-domain: Q17 / Q7 = Q10.
	let tmp32 = 131072 + (std >> 1) as i32;
	let inv_std = div_i32_i16(tmp32, std) as i16;

	// Calculate |inv_std2| = 1 / s^2, in Q14.
	let tmp16 = inv_std >> 2; // Q10 -> Q8.
	// Q-domain: (Q8 * Q8) >> 2 = Q14.
	let inv_std2 = ((tmp16 as i32 * tmp16 as i32) >> 2) as i16;

	let tmp16 = input << 3; // Q4 -> Q7
	let tmp16 = tmp16 - mean; // Q7 - Q7 = Q7

	// To be used later, when updating noise/speech model.
	// |delta| = (x - m) / s^2, in Q11.
	// Q-domain: (Q14 * Q7) >> 10 = Q11.
	let delta = ((inv_std2 as i32 * tmp16 as i32) >> 10) as i16;

	// Calculate the exponent |tmp32| = (x - m)^2 / (2 * s^2), in Q10. Replacing
	// division by two with one shift.
	// Q-domain: (Q11 * Q7) >> 8 = Q10.
	let tmp32 = (delta as i32 * tmp16 as i32) >> 9;

	// If the exponent is small enough to give a non-zero probability we calculate
	// |exp_value| ~= exp(-(x - m)^2 / (2 * s^2))
	//             ~= exp2(-log2(exp(1)) * |tmp32|).
	let exp_value = if tmp32 < COMP_VAR {
		// Calculate |tmp16| = log2(exp(1)) * |tmp32|, in Q10.
		// Q-domain: (Q12 * Q10) >> 12 = Q10.
		let mut tmp16 = -((LOG2_EXP * tmp32) >> 12) as i16;
		let exp_value = 0x0400 | (tmp16 & 0x03FF);
		tmp16 ^= 0xFFFFu16 as i16;
		tmp16 >>= 10;
		tmp16 += 1;
		// Get |exp_value| = exp(-|timp32|) in Q10.
		exp_value.wrapping_shr(tmp16 as _)
	} else {
		0
	};

	// Calculate and return (1 / s) * exp(-(x - m)^2 / (2 * s^2)), in Q20.
	// Q-domain: Q10 * Q10 = Q20.
	(inv_std as i32 * exp_value as i32, delta)
}

#[cfg(test)]
mod tests {
	use crate::gmm::gaussian_probability;

	#[test]
	fn test_gaussian_probability() {
		assert_eq!((1048576, 0), gaussian_probability(0, 0, 128));
		assert_eq!((1048576, 0), gaussian_probability(16, 128, 128));
		assert_eq!((1048576, 0), gaussian_probability(-16, -128, 128));

		assert_eq!((1024, 7552), gaussian_probability(59, 0, 128));
		assert_eq!((1024, 7552), gaussian_probability(75, 128, 128));
		assert_eq!((1024, -7552), gaussian_probability(-75, -128, 128));

		assert_eq!((0, 13440), gaussian_probability(105, 0, 128));
	}
}
