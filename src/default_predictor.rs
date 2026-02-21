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

static NORM_WEIGHT: &[f32; 40] = weight(0);
static LAYER1_KERNEL: &[f32; 9] = weight(40);
static LAYER1_WEIGHT: &[f32; 16] = weight(49);
static LAYER1_BIAS: &[f32; 16] = weight(65);
static LAYER2_KERNEL: &[f32; 48] = weight(81);
static LAYER2_WEIGHT: &[f32; 256] = weight(129);
static LAYER2_BIAS: &[f32; 16] = weight(385);
static LAYER3_KERNEL: &[f32; 48] = weight(401);
static LAYER3_WEIGHT: &[f32; 256] = weight(449);
static LAYER3_BIAS: &[f32; 16] = weight(705);
static RNN1_WEIGHT: &[f32; 10240] = weight(721);
static RNN2_WEIGHT: &[f32; 8192] = weight(10961);
static OUTPUT_WEIGHT: &[f32; 128] = weight(19153);

pub struct DefaultPredictor {
	state: Vec<f32>
}

impl DefaultPredictor {
	pub fn new() -> Self {
		Self { state: vec![0.0; 128] }
	}
}

impl crate::Predictor for DefaultPredictor {
	fn reset(&mut self) {
		self.state.fill(0.0);
	}

	fn normalize(&self, features: &mut [f32]) {
		let i_rms = 1. / (features.iter().map(|x| x * x).sum::<f32>() / features.len() as f32).sqrt();
		for (i, v) in features.iter_mut().enumerate() {
			*v = NORM_WEIGHT[i] * *v * i_rms;
		}
	}

	fn predict(&mut self, features: &[f32], buffer: &mut [f32]) -> f32 {
		let (buffer1, buffer2) = buffer.split_at_mut(288);
		input_layer1(&features, buffer1);
		input_layer2_3::<18, 9, false>(&buffer1[..288], LAYER2_KERNEL, LAYER2_WEIGHT, LAYER2_BIAS, &mut buffer2[..144]);
		input_layer2_3::<9, 5, true>(&buffer2[..144], LAYER3_KERNEL, LAYER3_WEIGHT, LAYER3_BIAS, &mut buffer1[..80]);
		mingru::<80>(&buffer1[..80], &self.state[..64], RNN1_WEIGHT, &mut buffer2[..128]);
		self.state[..64].copy_from_slice(&buffer2[..64]);
		mingru::<64>(&buffer2[..64], &self.state[64..128], RNN2_WEIGHT, &mut buffer1[..128]);
		self.state[64..128].copy_from_slice(&buffer1[..64]);
		output(&buffer2[..64], &buffer1[..64])
	}
}

#[inline(never)]
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

#[inline(never)]
fn input_layer2_3<const IN_FEATURES: usize, const OUT_FEATURES: usize, const LAYER3: bool>(
	features: &[f32],
	kernel: &[f32; 48],
	weight: &[f32; 256],
	bias: &[f32; 16],
	output: &mut [f32]
) {
	const HORIZONTAL_KERNEL_SIZE: usize = 3;
	const STRIDE: usize = 2;
	const CHANNELS: usize = 16;

	output.fill(0.0);

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
				sum += features[(c * IN_FEATURES) + ix as usize] * kernel[(c * HORIZONTAL_KERNEL_SIZE) + kw];
			}

			dw[c] = sum;
		}

		// pointwise conv
		for oc in 0..CHANNELS {
			let mut ic = 0.0;
			for c in 0..CHANNELS {
				let sum = dw[c];
				ic += sum * weight[(oc * CHANNELS) + c];
			}

			let ptr = if !LAYER3 { &mut output[(oc * OUT_FEATURES) + ox] } else { &mut output[(ox * CHANNELS) + oc] };
			*ptr = (ic + bias[oc]).max(0.0);
		}
	}
}

#[inline(never)]
fn mingru<const IN_DIM: usize>(features: &[f32], h: &[f32], weight: &[f32], out: &mut [f32]) {
	for d in 0..128 {
		let mut o = 0.0;
		let ri = d * IN_DIM;

		for f in 0..IN_DIM {
			o += features[f] * weight[ri + f];
		}

		out[d] = o;
	}

	for i in 0..64 {
		let g = (out[64 + i] * 0.25).clamp(0.0, 1.0);
		let v = &mut out[i];
		*v = (1. - g) * h[i] + g * *v;
	}
}

#[inline]
fn sigmoid(x: f32) -> f32 {
	1. / (1. + (-x).exp())
}

#[inline(never)]
fn output(out_1: &[f32], out_2: &[f32]) -> f32 {
	let mut out = 0.0;
	for f in 0..64 {
		out += out_1[f] * OUTPUT_WEIGHT[f];
	}
	for f in 0..64 {
		out += out_2[f] * OUTPUT_WEIGHT[64 + f];
	}
	sigmoid(out)
}
