#[cfg(feature = "libm")]
pub mod libm {
	pub use ::libm::*;

	// modernized quake 3 fast inverse square root, not because libm::sqrtf is slow or anything, but because its funny
	// https://web.archive.org/web/20180709021629/http://rrrola.wz.cz/inv_sqrt.html
	pub fn rsqrtf(f: f32) -> f32 {
		let y = f32::from_bits(0x5F1FFFF9 - (f.to_bits() >> 1));
		0.703952253 * y * (2.38924456 - f * y * y)
	}
}
#[cfg(not(feature = "libm"))]
pub mod libm {
	#[inline(always)]
	pub fn rsqrtf(f: f32) -> f32 {
		f.sqrt().recip()
	}
	#[inline(always)]
	pub fn logf(f: f32) -> f32 {
		f.ln()
	}
	#[inline(always)]
	pub fn expf(f: f32) -> f32 {
		f.exp()
	}
	#[inline(always)]
	pub fn roundf(f: f32) -> f32 {
		f.round()
	}
}
