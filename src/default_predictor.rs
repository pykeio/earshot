use core::slice;

use crate::util::libm;

const _WEIGHTS_LEN: usize = include_bytes!("weights.bin").len();
static WEIGHTS: &[u8; _WEIGHTS_LEN] = {
	#[repr(C, align(4))]
	struct AlignedData<T: ?Sized>(T);

	const __DATA: &'static AlignedData<[u8; _WEIGHTS_LEN]> = &AlignedData(*include_bytes!("weights.bin"));
	&__DATA.0
};

const fn weight<'a, const SIZE: usize>(offset: usize) -> &'a [f32; SIZE] {
	unsafe { &*(WEIGHTS.as_ptr().cast::<f32>().add(offset) as *const [_; SIZE]) }
}

const fn quantized_weight<'a, const SIZE: usize>(offset: usize) -> &'a [i16; SIZE] {
	unsafe { &*(WEIGHTS.as_ptr().cast::<i16>().add(offset * 2) as *const [_; SIZE]) }
}

static NORM_WEIGHT: &[f32; 40] = weight(0);
static LAYER1_KERNEL: &[f32; 9] = weight(40);
static LAYER1_WEIGHT: &[f32; 16] = weight(49);
static LAYER1_BIAS: &[f32; 16] = weight(65);
static LAYER2_KERNEL: &[f32; 48] = weight(81);
static LAYER2_WEIGHT: &[f32; 256] = weight(129);
static LAYER2_BIAS: &[f32; 16] = weight(385);
static LAYER3_KERNEL: &[f32; 48] = weight(401);
static LAYER3_WEIGHT: &[f32; 256] = weight(449);
static RNN1_WEIGHT: &[i16; 10240] = quantized_weight(705);
static RNN2_WEIGHT: &[i16; 8192] = quantized_weight(5825);
static OUTPUT_WEIGHT: &[i16; 128] = quantized_weight(9921);

pub struct DefaultPredictor {
	state: [i16; 128]
}

impl DefaultPredictor {
	pub const fn new() -> Self {
		Self { state: [0; 128] }
	}
}

impl crate::Predictor for DefaultPredictor {
	fn reset(&mut self) {
		self.state.fill(0);
	}

	fn normalize(&self, features: &mut [f32]) {
		let i_rms = libm::rsqrtf(features.iter().map(|x| x * x).sum::<f32>() / features.len() as f32);
		for (i, v) in features.iter_mut().enumerate() {
			*v = NORM_WEIGHT[i] * *v * i_rms;
		}
	}

	fn predict(&mut self, features: &[f32], buffer: &mut [f32]) -> f32 {
		let (buffer1, buffer2) = buffer.split_at_mut(288);
		let (state1, state2) = self.state.split_at_mut(64);

		input_layer1(&features, buffer1);
		input_layer2(&buffer1[..288], &mut buffer2[..144]);
		// SAFETY: &[i32] is aligned to 4 bytes, &[i16] requires alignment to 2, so this works
		let layer3_out = unsafe { slice::from_raw_parts_mut(buffer1.as_mut_ptr().cast::<i16>(), 80) };
		input_layer3(&buffer2[..144], layer3_out);

		// SAFETY: align_of::<i32>() == align_of::<f32>()
		let (buffer1, buffer2) = unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<i32>(), buffer.len()) }.split_at_mut(288);
		mingru::<80>(layer3_out, state1, RNN1_WEIGHT, &mut buffer2[..128]);
		mingru::<64>(state1, state2, RNN2_WEIGHT, &mut buffer1[..128]);

		output(&self.state)
	}
}

fn input_layer1(features: &[f32], output: &mut [f32]) {
	const NUM_FRAMES: usize = 3;
	const NUM_FEATURES: usize = 40;

	const KERNEL_SIZE: usize = 3;
	const {
		assert!((NUM_FRAMES - KERNEL_SIZE) / 1 + 1 == 1);
	};
	const DEPTHWISE_NUM_FEATURES: usize = (NUM_FEATURES - KERNEL_SIZE) / 1 + 1;
	const OUT_CHANNELS: usize = 16;

	const POOL_KERNEL_SIZE: usize = 3;
	const POOL_STRIDE: usize = 2;
	const POOLED_COLS: usize = (DEPTHWISE_NUM_FEATURES - POOL_KERNEL_SIZE) / POOL_STRIDE + 1;

	output.fill(0.0);

	let mut row = [0.0_f32; DEPTHWISE_NUM_FEATURES];
	for ox in 0..DEPTHWISE_NUM_FEATURES {
		// depthwise conv
		let mut sum = 0.0;
		for kh in 0..KERNEL_SIZE {
			for kw in 0..KERNEL_SIZE {
				let w = ox + kw;
				let input_idx = (kh * NUM_FEATURES) + w;
				sum += features[input_idx] * LAYER1_KERNEL[(kh * KERNEL_SIZE) + kw];
			}
		}

		row[ox] = sum;
	}

	for c in 0..OUT_CHANNELS {
		let mut new_row = [0.0; DEPTHWISE_NUM_FEATURES];
		for ox in 0..DEPTHWISE_NUM_FEATURES {
			// pointwise conv
			new_row[ox] = (row[ox] * LAYER1_WEIGHT[c]) + LAYER1_BIAS[c];
		}

		// max pool over row
		let out_row_offs = POOLED_COLS * c;
		for q in 0..POOLED_COLS {
			for x in 0..POOL_KERNEL_SIZE {
				let out_q = &mut output[out_row_offs + q];
				// `out` is zeroed, so this also acts as ReLU
				*out_q = (*out_q).max(new_row[(q * POOL_STRIDE) + x]);
			}
		}
	}
}

fn input_layer2(features: &[f32], output: &mut [f32]) {
	const IN_FEATURES: usize = 18;
	const OUT_FEATURES: usize = 9;
	const HORIZONTAL_KERNEL_SIZE: usize = 3;
	const STRIDE: usize = 2;
	const CHANNELS: usize = 16;

	for ox in 0..OUT_FEATURES {
		let mut dw = [0.0; CHANNELS];
		for c in 0..CHANNELS {
			// depthwise conv
			let mut sum = 0.0;
			for kw in 0..HORIZONTAL_KERNEL_SIZE {
				let ix = (ox * STRIDE + kw) as isize - 1;
				if ix < 0 || ix >= IN_FEATURES as isize {
					continue;
				}
				sum += features[(c * IN_FEATURES) + ix as usize] * LAYER2_KERNEL[(c * HORIZONTAL_KERNEL_SIZE) + kw];
			}

			dw[c] = sum;
		}

		// pointwise conv
		for oc in 0..CHANNELS {
			let mut ic = 0.0;
			for c in 0..CHANNELS {
				let sum = dw[c];
				ic += sum * LAYER2_WEIGHT[(oc * CHANNELS) + c];
			}

			output[(oc * OUT_FEATURES) + ox] = (ic + LAYER2_BIAS[oc]).max(0.0);
		}
	}
}

fn input_layer3(features: &[f32], output: &mut [i16]) {
	const IN_FEATURES: usize = 9;
	const OUT_FEATURES: usize = 5;
	const HORIZONTAL_KERNEL_SIZE: usize = 3;
	const STRIDE: usize = 2;
	const CHANNELS: usize = 16;

	for ox in 0..OUT_FEATURES {
		let mut dw = [0.0; CHANNELS];
		for c in 0..CHANNELS {
			// depthwise conv
			let mut sum = 0.0;
			for kw in 0..HORIZONTAL_KERNEL_SIZE {
				let ix = (ox * STRIDE + kw) as isize - 1;
				if ix < 0 || ix >= IN_FEATURES as isize {
					continue;
				}
				sum += features[(c * IN_FEATURES) + ix as usize] * LAYER3_KERNEL[(c * HORIZONTAL_KERNEL_SIZE) + kw];
			}

			dw[c] = sum;
		}

		// pointwise conv
		for oc in 0..CHANNELS {
			let mut ic = 0.0;
			for c in 0..CHANNELS {
				let sum = dw[c];
				ic += sum * LAYER3_WEIGHT[(oc * CHANNELS) + c];
			}

			// note: transposed, no ReLU or bias
			output[(ox * CHANNELS) + oc] = libm::roundf(ic * 32768.0).clamp(-32768.0, 32767.0) as i16;
		}
	}
}

fn mingru<const IN_DIM: usize>(features: &[i16], h: &mut [i16], weight: &[i16], temp: &mut [i32]) {
	for d in 0..128 {
		let mut o = 0;
		let ri = d * IN_DIM;

		for f in 0..IN_DIM {
			o += features[f] as i32 * weight[ri + f] as i32; // Q15 * Q9 = Q24
		}

		temp[d] = o;
	}

	for i in 0..64 {
		// gate is multiplied by 4 then clamped within [0, 1] in Q24
		let g = (temp[64 + i] << 2).clamp(0, 16777215) as i64;
		let y = (16777215 - g) * ((h[i] as i64) << 9) as i64 + g * temp[i] as i64; // Q24 * Q24 = Q48
		h[i] = (y >> 33).clamp(-32768, 32767) as i16; // back down to Q15
	}
}

#[inline]
fn sigmoid(x: f32) -> f32 {
	1. / (1. + libm::expf(-x))
}

fn output(state: &[i16]) -> f32 {
	let mut out = 0i32;
	for f in 0..128 {
		out += (state[f] as i32 * OUTPUT_WEIGHT[f] as i32) as i32; // Q15 * Q9 = Q24
	}
	sigmoid(out as f32 / const { 16777216 as f32 })
}
