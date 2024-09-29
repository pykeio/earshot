#[inline]
pub fn div_i32_i16(a: i32, b: i16) -> i32 {
	a.checked_div(b as i32).unwrap_or(i32::MAX)
}

#[inline]
pub fn size_in_bits(n: i32) -> u8 {
	32 - n.leading_zeros() as u8
}

/// Return the number of steps `a` can be left-shifted without overflow, or `0` if `a == 0`.
#[inline]
pub fn norm_i32(a: i32) -> i8 {
	if a != 0 {
		(if a < 0 { !a } else { a }).leading_zeros() as i8 - 1 // sub 1 for the sign bit
	} else {
		0
	}
}

/// Return the number of steps `a` can be left-shifted without overflow, or `0` if `a == 0`.
#[inline]
pub fn norm_u32(a: u32) -> u8 {
	if a != 0 { a.leading_zeros() as u8 } else { 0 }
}

pub fn weighted_average(data: &mut [[i16; 6]], channel: usize, offset: i16, weights: &[[i16; 6]]) -> i32 {
	let mut weighted_average = 0;
	for gaussian in 0..data.len() {
		data[gaussian][channel] += offset;
		weighted_average += data[gaussian][channel] as i32 * weights[gaussian][channel] as i32;
	}
	weighted_average
}

#[cfg(test)]
mod tests {
	use crate::util::{norm_u32, size_in_bits};

	#[test]
	fn test_norm() {
		assert_eq!(17, size_in_bits(111121));
		assert_eq!(0, norm_u32(0));
		assert_eq!(0, norm_u32(u32::MAX));
		assert_eq!(15, norm_u32(111121));
	}
}
