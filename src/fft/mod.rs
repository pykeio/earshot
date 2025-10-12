//! From https://gitlab.com/teskje/microfft-rs
//! Copyright (c) 2020-2024 Jan Teske, MIT license
//!
//! This is actually 2.5x slower than realfft/rustfft, but it's much simpler and doesn't require `std`.
//! Still, it only takes ~1.8us - only slightly slower than the first NN layer.

use core::{
	ops::{Add, Mul, Sub},
	slice
};

mod tables;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Complex32 {
	pub re: f32,
	pub im: f32
}

impl Complex32 {
	pub const fn new(re: f32, im: f32) -> Self {
		Self { re, im }
	}

	pub const fn norm_sqr(&self) -> f32 {
		self.re * self.re + self.im * self.im
	}
}

impl Add<Complex32> for Complex32 {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Complex32) -> Self::Output {
		Complex32::new(self.re + rhs.re, self.im + rhs.im)
	}
}
impl Sub<Complex32> for Complex32 {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Complex32) -> Self::Output {
		Complex32::new(self.re - rhs.re, self.im - rhs.im)
	}
}
impl Mul<Complex32> for Complex32 {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: Complex32) -> Self::Output {
		let re = self.re * rhs.re - self.im * rhs.im;
		let im = self.re * rhs.im + self.im * rhs.re;
		Complex32::new(re, im)
	}
}
impl Mul<f32> for Complex32 {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: f32) -> Self::Output {
		Complex32::new(self.re * rhs, self.im * rhs)
	}
}

pub(crate) trait CFft {
	type Half: CFft;

	const N: usize;
	const LOG2_N: usize = Self::N.ilog2() as usize;

	const BITREV_TABLE: &'static [u16] = tables::BITREV[Self::LOG2_N];

	#[inline]
	fn transform(x: &mut [Complex32]) -> &mut [Complex32] {
		debug_assert_eq!(x.len(), Self::N);

		Self::bit_reverse_reorder(x);
		Self::compute_butterflies(x);
		x
	}

	#[inline]
	fn bit_reverse_reorder(x: &mut [Complex32]) {
		debug_assert_eq!(x.len(), Self::N);

		for i in 0..Self::N {
			let j = Self::BITREV_TABLE[i] as usize;
			if i != j {
				x.swap(i, j);
			}
		}
	}

	#[inline]
	fn compute_butterflies(x: &mut [Complex32]) {
		debug_assert_eq!(x.len(), Self::N);

		let m = Self::N / 2;
		let u = m / 2;

		let table_len = tables::SINE.len();
		let table_stride = (table_len + 1) * 4 / Self::N;

		Self::Half::compute_butterflies(&mut x[..m]);
		Self::Half::compute_butterflies(&mut x[m..]);

		// [k = 0] twiddle factor: `1 + 0i`
		let (x_0, x_m) = (x[0], x[m]);
		x[0] = x_0 + x_m;
		x[m] = x_0 - x_m;

		// [k in [1, m/2)] twiddle factor:
		//   - re from SINE table backwards and negative
		//   - im from SINE table directly
		for k in 1..u {
			let s = k * table_stride;
			let re = tables::SINE[table_len - s] * -1.;
			let im = tables::SINE[s - 1];
			let twiddle = Complex32::new(re, im);

			let (x_k, x_km) = (x[k], x[k + m]);
			let y = twiddle * x_km;
			x[k] = x_k + y;
			x[k + m] = x_k - y;
		}

		// [k = m/2] twiddle factor: `0 - 1i`
		let (x_u, x_um) = (x[u], x[u + m]);
		let y = x_um * Complex32::new(0., -1.);
		x[u] = x_u + y;
		x[u + m] = x_u - y;

		// [k in (m/2, m)] twiddle factor:
		//   - re from SINE table directly
		//   - im from SINE table backwards
		for k in (u + 1)..m {
			let s = (k - u) * table_stride;
			let re = tables::SINE[s - 1];
			let im = tables::SINE[table_len - s];
			let twiddle = Complex32::new(re, im);

			let (x_k, x_km) = (x[k], x[k + m]);
			let y = twiddle * x_km;
			x[k] = x_k + y;
			x[k + m] = x_k - y;
		}
	}
}

pub(crate) struct CFftN<const N: usize>;

impl CFft for CFftN<1> {
	type Half = Self;

	const N: usize = 1;

	#[inline]
	fn bit_reverse_reorder(x: &mut [Complex32]) {
		debug_assert_eq!(x.len(), 1);
	}

	#[inline]
	fn compute_butterflies(x: &mut [Complex32]) {
		debug_assert_eq!(x.len(), 1);
	}
}

impl CFft for CFftN<2> {
	type Half = CFftN<1>;

	const N: usize = 2;

	#[inline]
	fn compute_butterflies(x: &mut [Complex32]) {
		debug_assert_eq!(x.len(), 2);

		let (x_0, x_1) = (x[0], x[1]);
		x[0] = x_0 + x_1;
		x[1] = x_0 - x_1;
	}
}

macro_rules! cfft_impls {
    ($($N:expr),*) => {
        $(
            impl CFft for CFftN<$N> {
                type Half = CFftN<{$N / 2}>;

                const N: usize = $N;
            }
        )*
    };
}

cfft_impls! { 4, 8, 16, 32, 64, 128, 256, 512, 1024 }

pub(crate) trait RFft {
	type CFft: CFft;

	const N: usize = Self::CFft::N * 2;

	#[inline]
	fn transform(x: &mut [f32]) -> &mut [Complex32] {
		debug_assert_eq!(x.len(), Self::N);

		let x = Self::pack_complex(x);

		Self::CFft::transform(x);
		Self::recombine(x);
		x
	}

	#[inline]
	fn pack_complex(x: &mut [f32]) -> &mut [Complex32] {
		assert_eq!(x.len(), Self::N);

		let len = Self::N / 2;
		let data = x.as_mut_ptr().cast::<Complex32>();
		unsafe { slice::from_raw_parts_mut(data, len) }
	}

	#[inline]
	fn recombine(x: &mut [Complex32]) {
		let m = Self::CFft::N;
		debug_assert_eq!(x.len(), m);

		let table_len = tables::SINE.len();
		let table_stride = (table_len + 1) * 4 / Self::N;

		// The real part of the first element is the DC value.
		// Additionally, the real-valued coefficient at the Nyquist frequency
		// is stored in the imaginary part.
		let x0 = x[0];
		x[0] = Complex32::new(x0.re + x0.im, x0.re - x0.im);

		let u = m / 2;
		for k in 1..u {
			let s = k * table_stride;
			let twiddle_re = tables::SINE[table_len - s] * -1.;
			let twiddle_im = tables::SINE[s - 1];

			let (x_k, x_nk) = (x[k], x[m - k]);
			// 20% speed boost just by replacing / 2 with * 0.5 here!
			let sum = (x_k + x_nk) * 0.5;
			let diff = (x_k - x_nk) * 0.5;

			x[k] = Complex32::new(sum.re + twiddle_re * sum.im + twiddle_im * diff.re, diff.im + twiddle_im * sum.im - twiddle_re * diff.re);
			x[m - k] = Complex32::new(sum.re - twiddle_re * sum.im - twiddle_im * diff.re, -diff.im + twiddle_im * sum.im - twiddle_re * diff.re);
		}

		let xu = x[u];
		x[u] = Complex32::new(xu.re, -xu.im);
	}
}

struct RFftN<const N: usize>;

impl RFft for RFftN<1024> {
	type CFft = CFftN<512>;
}

pub fn rfft_1024(x: &mut [f32]) -> &mut [Complex32] {
	debug_assert_eq!(x.len(), 1026);
	let mut comp = RFftN::<1024>::transform(&mut x[..1024]);
	// microfft packs Nyquist real into DC bin imaginary so the output can fit in the original 1024-wide buffer. we expect
	// 513 values to get the mel spectrogram, so unpack them
	comp = unsafe { slice::from_raw_parts_mut(comp.as_mut_ptr(), comp.len() + 1) };
	comp[comp.len() - 1].re = comp[0].im;
	comp[0].im = 0.0;
	comp
}
