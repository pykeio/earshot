from typing import Sequence

import numpy

__version__: str

class Detector:
	def __init__(self) -> None: ...

	def reset(self) -> None:
		"""
		Resets the internal state of the voice activity detector.

		The detector should be reset whenever:
		- the recording device changes; or
		- the detector is being used for a new audio sequence.
		"""
		...

	def predict_i16(self, frame: Sequence[int] | 'numpy.ndarray') -> float:
		"""
		Predicts the voice activity score of a single input frame of 16-bit PCM audio.

		The frame:
		- should be sampled at 16 KHz;
		- should be exactly 256 samples (so 16 ms) in length.

		The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
		can be adjusted according to application-specific needs.
		"""
		...

	def predict_f32(self, frame: Sequence[float] | 'numpy.ndarray') -> float:
		"""
		Predicts the voice activity score of a single input frame of 32-bit floating-point PCM audio.

		The frame:
		- should be sampled at 16 KHz;
		- should be exactly 256 samples (so 16 ms) in length;
		- should consist only of samples in the range [-1, 1].

		The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
		can be adjusted according to application-specific needs.
		"""
		...
