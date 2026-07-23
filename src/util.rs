#[cfg(feature = "libm")]
pub mod libm {
	pub use ::libm::*;

	pub fn rsqrtf(f: f32) -> f32 {
		1.0 / sqrtf(f)
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
