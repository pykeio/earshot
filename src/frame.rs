/// Length of a single frame, in samples at [`SAMPLE_RATE`][crate::SAMPLE_RATE] (16,000 Hz).
pub const FRAME_SIZE: usize = 256;

/// A single frame of audio: [`FRAME_SIZE`] (256) samples of mono PCM sampled at [`SAMPLE_RATE`][crate::SAMPLE_RATE]
/// (16,000 Hz).
pub trait Frame {
	/// The length of the frame in samples; must match [`FRAME_SIZE`].
	fn len(&self) -> usize;

	/// Returns an iterator over this frame's samples.
	///
	/// **The samples are expected to be in the range `[-32768, 32768]`** even though they are represented as `f32`
	/// (which is usually `[-1, 1]`).
	fn samples(self) -> impl ExactSizeIterator<Item = f32>;
}

impl Frame for &[f32] {
	#[inline]
	fn len(&self) -> usize {
		<[f32]>::len(*self)
	}

	#[inline]
	fn samples(self) -> impl ExactSizeIterator<Item = f32> {
		<[f32]>::iter(self).map(|x| x * 32768.0)
	}
}

#[cfg(feature = "alloc")]
impl Frame for &alloc::vec::Vec<f32> {
	#[inline]
	fn len(&self) -> usize {
		alloc::vec::Vec::len(self)
	}

	#[inline]
	fn samples(self) -> impl ExactSizeIterator<Item = f32> {
		<[f32]>::iter(self).map(|x| x * 32768.0)
	}
}

impl Frame for [f32; 256] {
	#[inline]
	fn len(&self) -> usize {
		256
	}

	#[inline]
	fn samples(self) -> impl ExactSizeIterator<Item = f32> {
		IntoIterator::into_iter(self).map(|x| x * 32768.0)
	}
}

impl Frame for &[i16] {
	#[inline]
	fn len(&self) -> usize {
		<[i16]>::len(*self)
	}

	#[inline]
	fn samples(self) -> impl ExactSizeIterator<Item = f32> {
		<[i16]>::iter(self).map(|x| *x as f32)
	}
}

#[cfg(feature = "alloc")]
impl Frame for &alloc::vec::Vec<i16> {
	#[inline]
	fn len(&self) -> usize {
		alloc::vec::Vec::len(self)
	}

	#[inline]
	fn samples(self) -> impl ExactSizeIterator<Item = f32> {
		<[i16]>::iter(self).map(|x| *x as f32)
	}
}

impl Frame for [i16; 256] {
	#[inline]
	fn len(&self) -> usize {
		256
	}

	#[inline]
	fn samples(self) -> impl ExactSizeIterator<Item = f32> {
		IntoIterator::into_iter(self).map(|x| x as f32)
	}
}

/// A [`Frame`] originating from interleaved stereo audio. The inner audio frame will be mixed down to mono so it can be
/// used for VAD.
pub struct Stereo<F>(pub F);

impl<F: Frame> Frame for Stereo<F> {
	fn len(&self) -> usize {
		self.0.len() / 2
	}

	fn samples(self) -> impl ExactSizeIterator<Item = f32> {
		StereoIter { iter: self.0.samples() }
	}
}

impl Frame for Stereo<[f32; 512]> {
	#[inline]
	fn len(&self) -> usize {
		256
	}

	#[inline]
	fn samples(self) -> impl ExactSizeIterator<Item = f32> {
		StereoIter {
			iter: IntoIterator::into_iter(self.0).map(|x| x * 32768.0)
		}
	}
}

impl Frame for Stereo<[i16; 512]> {
	#[inline]
	fn len(&self) -> usize {
		256
	}

	#[inline]
	fn samples(self) -> impl ExactSizeIterator<Item = f32> {
		StereoIter {
			iter: IntoIterator::into_iter(self.0).map(|x| x as f32)
		}
	}
}

struct StereoIter<I> {
	iter: I
}

impl<I: Iterator<Item = f32>> Iterator for StereoIter<I> {
	type Item = f32;

	fn next(&mut self) -> Option<Self::Item> {
		let iter = self.iter.by_ref();
		let a = iter.next()?;
		let b = iter.next()?;
		Some((a + b) * 0.5)
	}
}

impl<I: Iterator<Item = f32> + ExactSizeIterator> ExactSizeIterator for StereoIter<I> {
	fn len(&self) -> usize {
		self.iter.len() / 2
	}
}
