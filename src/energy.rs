use crate::util::{norm_i32, size_in_bits};

fn get_scaling_square(inv: &[i16], times: usize) -> u8 {
	let n_bits = size_in_bits(times as i32);
	let smax = inv.iter().map(|c| c.abs()).max().unwrap() as i32;
	let t = norm_i32(smax * smax) as u8;
	if smax != 0 {
		if t > n_bits { 0 } else { n_bits - t }
	} else {
		0 // Since norm(0) returns 0
	}
}

pub struct EnergyResult {
	pub energy: i32,
	pub scaling_factor: u8
}

pub fn energy(inv: &[i16]) -> EnergyResult {
	let scaling_factor = get_scaling_square(inv, inv.len());
	let energy = inv.iter().map(|x| (*x as i32 * *x as i32) >> scaling_factor).sum();
	EnergyResult { energy, scaling_factor }
}

#[cfg(test)]
mod tests {
	use super::EnergyResult;
	use crate::energy::energy;

	#[test]
	fn test_energy() {
		let inv = [1, 2, 33, 100];
		let EnergyResult { energy, scaling_factor } = energy(&inv);
		assert_eq!(energy, 11094);
		assert_eq!(scaling_factor, 0);
	}
}
