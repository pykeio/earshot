use alloc::{boxed::Box, vec};
use core::{mem, slice};

use super::{Predictor, util::OnceLock};

struct BitBufferReader<'d> {
	pub buf: &'d [u8],
	idx: usize,
	bit_buffer: u32,
	n_bits: u32
}

impl<'d> BitBufferReader<'d> {
	pub fn new(buffer: &'d [u8]) -> Self {
		Self {
			buf: buffer,
			idx: 0,
			bit_buffer: 0,
			n_bits: 0
		}
	}

	pub fn read(&mut self, len: u32) -> i32 {
		while self.n_bits < len {
			let byte = self.buf[self.idx];
			self.idx += 1;

			self.bit_buffer |= (byte as u32) << self.n_bits;
			self.n_bits += 8;
		}

		let bits = self.bit_buffer & ((1 << len) - 1);
		self.bit_buffer >>= len;
		self.n_bits -= len;

		let sign = (bits & (1 << (len - 1))) != 0;
		(if sign { -1 << len } else { 0 }) | bits as i32
	}

	pub fn read_array<T: FromI32>(&mut self, bit_len: u32, cnt: usize) -> Box<[T]> {
		(0..cnt).map(|_| T::from_i32(self.read(bit_len))).collect()
	}
}

trait FromI32 {
	fn from_i32(x: i32) -> Self;
}

impl FromI32 for i8 {
	fn from_i32(x: i32) -> Self {
		x as i8
	}
}
impl FromI32 for i16 {
	fn from_i32(x: i32) -> Self {
		x as i16
	}
}
impl FromI32 for i32 {
	fn from_i32(x: i32) -> Self {
		x
	}
}

pub struct PackedWeights {
	layer1_kernel: Box<[i16]>,
	layer1_weight: Box<[i16]>,
	layer1_bias: Box<[i16]>,
	layer2_kernel: Box<[i16]>,
	layer2_weight: Box<[i16]>,
	layer2_bias: Box<[i16]>,
	layer3_kernel: Box<[i16]>,
	layer3_weight: Box<[i16]>,
	layer3_bias: Box<[i16]>,
	lstm1_ih: Box<[i16]>,
	lstm1_hh: Box<[i16]>,
	lstm1_bias: Box<[i16]>,
	lstm2_ih: Box<[i16]>,
	lstm2_hh: Box<[i16]>,
	lstm2_bias: Box<[i16]>,
	out1_weight: Box<[i16]>,
	out1_bias: Box<[i16]>,
	out2_weight: Box<[i16]>,
	out2_bias: i8
}

impl PackedWeights {
	pub fn new(bytes: &[u8]) -> Self {
		assert_eq!(bytes.len(), 135783, "invalid length for packed QuantizedPredictor weights");
		let mut reader = BitBufferReader::new(bytes);
		Self {
			layer1_kernel: reader.read_array(14, 9),
			layer1_weight: reader.read_array(14, 16),
			layer1_bias: reader.read_array(12, 16),
			layer2_kernel: reader.read_array(15, 48),
			layer2_weight: reader.read_array(16, 256),
			layer2_bias: reader.read_array(14, 16),
			layer3_kernel: reader.read_array(14, 48),
			layer3_weight: reader.read_array(15, 256),
			layer3_bias: reader.read_array(12, 16),
			lstm1_ih: reader.read_array(15, 20480),
			lstm1_hh: reader.read_array(14, 16384),
			lstm1_bias: reader.read_array(12, 256),
			lstm2_ih: reader.read_array(15, 16384),
			lstm2_hh: reader.read_array(14, 16384),
			lstm2_bias: reader.read_array(12, 256),
			out1_weight: reader.read_array(14, 4096),
			out1_bias: reader.read_array(11, 32),
			out2_weight: reader.read_array(13, 32),
			out2_bias: reader.read(4) as i8
		}
	}
}

#[cfg(feature = "embed-weights")]
static DEFAULT_WEIGHT_BYTES: &[u8] = include_bytes!("quantized-model.bin");
#[cfg(feature = "embed-weights")]
static DEFAULT_WEIGHTS: OnceLock<PackedWeights> = OnceLock::new();

#[cfg(feature = "embed-weights")]
pub fn default_weights() -> &'static PackedWeights {
	DEFAULT_WEIGHTS.get_or_init(|| PackedWeights::new(DEFAULT_WEIGHT_BYTES))
}

pub struct ActivationTables {
	sigmoid: Box<[i32]>,
	tanh: Box<[i32]>
}

impl ActivationTables {
	const Q11_SCALE: i32 = 2048; // 2 ** 11
	const Q11_SCALE_FLOAT: f32 = 2048.;
	const SIGMOID_MAX: i32 = Self::Q11_SCALE * 6; // sigmoid goes asymptotic < -6 or > 6, so limit computation to between these values
	const TANH_MAX: i32 = Self::Q11_SCALE * 4; // ^ 4 for tanh
	pub const OUT_SCALE: f32 = 65536.; // 2 ** 16, outputs in Q16

	pub fn new() -> Self {
		let sigmoid_len = Self::SIGMOID_MAX * 2 + 1;
		let mut sigmoid_table = vec![0; sigmoid_len as usize].into_boxed_slice();
		for i in 0..sigmoid_len {
			let v = Self::_real_sigmoid((i - (Self::SIGMOID_MAX)) as f32 / Self::Q11_SCALE_FLOAT);
			sigmoid_table[i as usize] = libm::roundevenf(v * Self::OUT_SCALE) as i32;
		}
		let tanh_len = Self::TANH_MAX * 2 + 1;
		let mut tanh_table = vec![0; tanh_len as usize].into_boxed_slice();
		for i in 0..tanh_len {
			let v = libm::tanhf((i - (Self::TANH_MAX)) as f32 / Self::Q11_SCALE_FLOAT);
			tanh_table[i as usize] = libm::roundevenf(v * Self::OUT_SCALE) as i32;
		}

		Self {
			sigmoid: sigmoid_table,
			tanh: tanh_table
		}
	}

	#[inline]
	fn _real_sigmoid(x: f32) -> f32 {
		1. / (1. + libm::expf(-x))
	}

	#[inline]
	pub fn sigmoid(&self, x: i32) -> i32 {
		unsafe {
			*self
				.sigmoid
				.get_unchecked((x + Self::SIGMOID_MAX).clamp(0, Self::SIGMOID_MAX * 2) as usize)
		}
	}
	#[inline]
	pub fn tanh(&self, x: i32) -> i32 {
		unsafe { *self.tanh.get_unchecked((x + Self::TANH_MAX).clamp(0, Self::TANH_MAX * 2) as usize) }
	}
}

static ACTIVATION_TABLES: OnceLock<ActivationTables> = OnceLock::new();

pub struct QuantizedPredictor<'w> {
	weights: &'w PackedWeights,
	state: Box<[i32]>
}

impl<'w> QuantizedPredictor<'w> {
	pub fn new(weights: &'w PackedWeights) -> Self {
		Self {
			weights,
			state: vec![0; 256].into_boxed_slice()
		}
	}
}

#[cfg(feature = "embed-weights")]
impl Default for QuantizedPredictor<'static> {
	fn default() -> Self {
		Self::new(default_weights())
	}
}

impl Predictor for QuantizedPredictor<'_> {
	fn reset(&mut self) {
		self.state.fill(0);
	}

	fn predict(&mut self, features: &[f32], buffer: &mut [f32]) -> f32 {
		assert_eq!(features.len(), 41 * 3);
		assert!(buffer.len() > 464);

		let buffer = unsafe { mem::transmute::<&mut [f32], &mut [i32]>(buffer) };

		let buffer_ptr = buffer.as_mut_ptr();
		input_layer1(features, &self.weights.layer1_kernel, &self.weights.layer1_weight, &self.weights.layer1_bias, &mut buffer[..304]);
		input_layer2(&buffer[..304], &self.weights.layer2_kernel, &self.weights.layer2_weight, &self.weights.layer2_bias, unsafe {
			slice::from_raw_parts_mut(buffer_ptr.add(304), 160)
		});
		input_layer3(&buffer[304..], &self.weights.layer3_kernel, &self.weights.layer3_weight, &self.weights.layer3_bias, unsafe {
			slice::from_raw_parts_mut(buffer_ptr, 80)
		});
		lstm::<80, { 80 * 256 }>(
			&buffer[..80],
			&self.state[..64],
			&self.state[64..128],
			&self.weights.lstm1_ih,
			&self.weights.lstm1_hh,
			&self.weights.lstm1_bias,
			unsafe { slice::from_raw_parts_mut(buffer_ptr.add(80), 256) }
		);
		self.state[..128].copy_from_slice(&buffer[80..208]);
		lstm::<64, { 64 * 256 }>(
			&self.state[..128],
			&self.state[128..192],
			&self.state[192..],
			&self.weights.lstm2_ih,
			&self.weights.lstm2_hh,
			&self.weights.lstm2_bias,
			&mut buffer[..256]
		);
		self.state[128..].copy_from_slice(&buffer[..128]);
		output(
			&self.state[..128],
			&self.state[128..],
			&self.weights.out1_weight,
			&self.weights.out1_bias,
			&self.weights.out2_weight,
			self.weights.out2_bias
		)
	}
}

#[inline(never)]
fn input_layer1(features: &[f32], kernel: &[i16], weight: &[i16], bias: &[i16], output: &mut [i32]) {
	const NUM_FRAMES: usize = 3;
	const NUM_FEATURES: usize = 41;
	const FEATURES_INPUT: usize = const { NUM_FRAMES * NUM_FEATURES };

	const KERNEL_SIZE: usize = 3;
	const {
		assert!((NUM_FRAMES - KERNEL_SIZE) / 1 + 1 == 1);
	};
	const DEPTHWISE_NUM_FEATURES: usize = (NUM_FEATURES - KERNEL_SIZE) / 1 + 1;
	const OUT_CHANNELS: usize = 16;

	const POOL_KERNEL_SIZE: usize = 3;
	const POOL_STRIDE: usize = 2;
	const SCALE_FACTOR: f32 = (1 << 16) as f32;
	const POOLED_COLS: usize = (DEPTHWISE_NUM_FEATURES - POOL_KERNEL_SIZE) / POOL_STRIDE + 1;

	output.fill(0);

	assert_eq!(features.len(), FEATURES_INPUT);

	let mut tmp = [0i32; FEATURES_INPUT];
	// doing this conversion in the convolution loop kills performance
	for i in 0..FEATURES_INPUT {
		unsafe {
			// convert to Q16
			*tmp.get_unchecked_mut(i) = libm::floorf(*features.get_unchecked(i) * SCALE_FACTOR) as i32;
		};
	}

	let mut row = [0; DEPTHWISE_NUM_FEATURES];
	for c in 0..OUT_CHANNELS {
		for ox in 0..DEPTHWISE_NUM_FEATURES {
			// depthwise conv
			let mut sum = 0;
			for kh in 0..KERNEL_SIZE {
				for kw in 0..KERNEL_SIZE {
					let w = ox + kw;
					let input_idx = (kh * NUM_FEATURES) + w;
					unsafe {
						// Q16 * Q13 = Q29
						sum += *tmp.get_unchecked(input_idx) as i64 * *kernel.get_unchecked((kh * KERNEL_SIZE) + kw) as i64;
					}
				}
			}

			// pointwise conv
			unsafe {
				// Q29 * Q13 = Q42. bias is Q12 so shift left by 42-12=30
				let x = (sum * *weight.get_unchecked(c) as i64) + ((*bias.get_unchecked(c) as i64) << 30);
				// shift down to Q16
				*row.get_unchecked_mut(ox) = (x >> 26) as i32;
			}
		}

		// max pool over row
		let out_row_offs = POOLED_COLS * c;
		for q in 0..POOLED_COLS {
			for x in 0..POOL_KERNEL_SIZE {
				let out_q = unsafe { output.as_mut_ptr().add(out_row_offs + q) };
				// `output` is initially zeroed, so this also acts as ReLU
				unsafe { *out_q = (*out_q).max(*row.get_unchecked((q * POOL_STRIDE) + x)) };
			}
		}
	}
}

#[inline(never)]
fn input_layer2(features: &[i32], kernel: &[i16], weight: &[i16], bias: &[i16], output: &mut [i32]) {
	const HORIZONTAL_KERNEL_SIZE: usize = 3;
	const STRIDE: usize = 2;
	const CHANNELS: usize = 16;

	const IN_FEATURES: usize = 19;
	const OUT_FEATURES: usize = 10;

	output.fill(0);

	for ox in 0..OUT_FEATURES {
		let mut row = [0; CHANNELS];
		for c in 0..CHANNELS {
			// depthwise conv
			let mut sum = 0;
			for kw in 0..HORIZONTAL_KERNEL_SIZE {
				let ix = (ox * STRIDE + kw) as isize - 1;
				if ix < 0 || ix >= IN_FEATURES as isize {
					continue;
				}

				// Q16 * Q13 = Q29
				unsafe {
					sum += *features.get_unchecked((c * IN_FEATURES) + ix as usize) as i64 * *kernel.get_unchecked((c * HORIZONTAL_KERNEL_SIZE) + kw) as i64;
				}
			}

			// pointwise conv
			for oc in 0..CHANNELS {
				unsafe {
					// Q29 * Q13 = Q42
					let r = sum * *weight.get_unchecked((oc * CHANNELS) + c) as i64;
					*row.get_unchecked_mut(oc) += r;
				}
			}
		}

		// apply pointwise conv bias + relu
		for oc in 0..CHANNELS {
			unsafe {
				// bias is Q12 so shift left by 42-12=30
				let br = *row.get_unchecked(oc) + ((*bias.get_unchecked(oc) as i64) << 30);
				// shift down to Q16
				*output.get_unchecked_mut((oc * OUT_FEATURES) + ox) = ((br >> 26) as i32).max(0);
			}
		}
	}
}

#[inline(never)]
fn input_layer3(features: &[i32], kernel: &[i16], weight: &[i16], bias: &[i16], output: &mut [i32]) {
	const HORIZONTAL_KERNEL_SIZE: usize = 3;
	const STRIDE: usize = 2;
	const CHANNELS: usize = 16;

	const IN_FEATURES: usize = 10;
	const OUT_FEATURES: usize = 5;

	output.fill(0);

	for ox in 0..OUT_FEATURES {
		let mut row = [0; CHANNELS];
		for c in 0..CHANNELS {
			// depthwise conv
			let mut sum = 0i64;
			for kw in 0..HORIZONTAL_KERNEL_SIZE {
				let ix = ox * STRIDE + kw; // layer 3 does not use left padding
				if ix >= IN_FEATURES {
					continue;
				}
				unsafe {
					// Q16 * Q13 = Q29
					sum += *features.get_unchecked((c * IN_FEATURES) + ix as usize) as i64 * *kernel.get_unchecked((c * HORIZONTAL_KERNEL_SIZE) + kw) as i64;
				}
			}

			// pointwise conv
			for oc in 0..CHANNELS {
				unsafe {
					// Q29 * Q13 = Q42
					*row.get_unchecked_mut(oc) += sum * *weight.get_unchecked((oc * CHANNELS) + c) as i64;
				}
			}
		}

		// apply pointwise conv bias + relu
		for oc in 0..CHANNELS {
			unsafe {
				// bias is Q12 so shift left by 42-12=30
				let r = *row.get_unchecked_mut(oc) + ((*bias.get_unchecked(oc) as i64) << 30);
				let ptr = output.get_unchecked_mut((ox * CHANNELS) + oc);
				// shift down to Q16
				*ptr = (r >> 26).max(0) as i32;
			}
		}
	}
}

#[inline(never)]
fn lstm<const IN_DIM: usize, const IH_DIM: usize>(features: &[i32], h: &[i32], c: &[i32], weight_ih: &[i16], weight_hh: &[i16], bias: &[i16], out: &mut [i32]) {
	for d in 0..256 {
		// init with Q10 bias, shifted left by 18 to get 10+18=Q28
		let mut o = (unsafe { *bias.get_unchecked(d) } as i64) << 18;
		let (ri, rh) = (d * IN_DIM, d * 64);

		for f in 0..IN_DIM {
			unsafe {
				// Q16 * Q12 = Q28
				o += *features.get_unchecked(f) as i64 * *weight_ih.get_unchecked(ri + f) as i64;
			};
		}

		for g in 0..64 {
			unsafe {
				// Q16 * Q12 = Q28
				o += *h.get_unchecked(g) as i64 * *weight_hh.get_unchecked(rh + g) as i64;
			}
		}

		unsafe {
			// shift down from Q28 to Q11
			*out.get_unchecked_mut(d) = (o >> 17) as i32;
		};
	}

	let act = ACTIVATION_TABLES.get_or_init(ActivationTables::new);
	for i in 0..64 {
		unsafe {
			// layout is [input, output, forget, cell]
			let ix = act.sigmoid(*out.get_unchecked(i)) as i64;
			let fx = act.sigmoid(*out.get_unchecked(128 + i)) as i64;
			let cx = act.tanh(*out.get_unchecked(192 + i)) as i64;
			// Q16 * Q16 = Q32; Q16 * Q16 = Q32
			let x = (fx * *c.get_unchecked(i) as i64) + (ix * cx);
			let xt = act.tanh((x >> 21) as i32) as i64; // Q11 in, Q16 out
			// arrange outputs as [hidden, cell]
			let o = act.sigmoid(mem::replace(out.get_unchecked_mut(64 + i), (x >> 16) as i32)) as i64;
			// Q16 * Q16 = Q32, shift down to Q16
			*out.get_unchecked_mut(i) = ((o * xt) >> 16) as i32;
		};
	}
}

#[inline(never)]
fn output(out_1: &[i32], out_2: &[i32], weight_1: &[i16], bias_1: &[i16], weight_2: &[i16], bias_2: i8) -> f32 {
	let mut temp = [0; 32];
	for h in 0..64 {
		for f in 0..32 {
			unsafe {
				// Q16 * Q12 = Q28, shift down to Q19
				let mut o = *out_2.get_unchecked(h) as i64 * *weight_1.get_unchecked(h * 32 + f) as i64;
				o += *out_1.get_unchecked(h) as i64 * *weight_1.get_unchecked((h + 64) * 32 + f) as i64;
				*temp.get_unchecked_mut(f) += (o >> 9) as i32;
			}
		}
	}

	let mut out = 0;
	for f in 0..32 {
		unsafe {
			// bias is Q10 so shift left by 19-10=9
			let q = *temp.get_unchecked(f) as i64 + ((*bias_1.get_unchecked(f) as i64) << 9);
			// Q19 * Q13 = Q32
			out += q.max(0) * *weight_2.get_unchecked(f) as i64;
		}
	}
	// bias is Q9 so shift left by 32-9=23
	out += (bias_2 as i64) << 23;
	// shift down to Q11
	out >>= 21;
	ACTIVATION_TABLES.get_or_init(ActivationTables::new).sigmoid(out as i32) as f32 / ActivationTables::OUT_SCALE
}
