/***************************************************
 * Earshot VAD - https://github.com/pykeio/earshot *
 *           Copyright (c) 2026 pyke.io            *
 *        Licensed under MIT OR Apache-2.0         *
 **************************************************/

#ifndef __PYKE_EARSHOT_H
#define __PYKE_EARSHOT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ESVoiceActivityDetector ESVoiceActivityDetector;

/** Returns the size of an `ESVoiceActivityDetector` in bytes. */
extern size_t ESVoiceActivityDetectorSize(void);

/**
 * Initializes a pre-allocated `ESVoiceActivityDetector`.
 *
 * The size of the allocation *must* be at least the return value of ESVoiceActivityDetectorSize() in bytes.
 *
 * You are responsible for managing the allocation; **do not use ESVoiceActivityDetectorRelease()**!
 */
extern void ESVoiceActivityDetectorInit(ESVoiceActivityDetector *detector);

/**
 * Creates a new `ESVoiceActivityDetector` on the heap.
 * Release this with ESVoiceActivityDetectorRelease().
 */
extern ESVoiceActivityDetector *ESVoiceActivityDetectorNew(void);

/** Releases a detector created with ESVoiceActivityDetectorNew(). */
extern void ESVoiceActivityDetectorRelease(ESVoiceActivityDetector *detector);

/**
 * Resets the internal state of the voice activity detector.
 *
 * The detector should be reset whenever:
 * - the recording device changes; or
 * - the detector is being used for a new audio sequence.
 */
extern void ESVoiceActivityDetectorReset(ESVoiceActivityDetector *detector);

/**
 * Predicts the voice activity score of a single input frame of 16-bit PCM audio.
 *
 * The frame:
 * - should be sampled at 16 KHz;
 * - should be exactly 256 samples (so 16 ms) in length.
 *
 * The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
 * can be adjusted according to application-specific needs.
 */
extern float ESVoiceActivityDetectorPredictI16(ESVoiceActivityDetector *detector, const int16_t *frame, size_t frameLen);

/**
 * Predicts the voice activity score of a single input frame of 32-bit floating-point PCM audio.
 *
 * The frame:
 * - should be sampled at 16 KHz;
 * - should be exactly 256 samples (so 16 ms) in length;
 * - should consist only of samples in the range [-1, 1].
 *
 * The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
 * can be adjusted according to application-specific needs.
 */
extern float ESVoiceActivityDetectorPredictF32(ESVoiceActivityDetector *detector, const float *frame, size_t frameLen);

#ifdef __cplusplus
}

#include <vector>

namespace earshot {
	class Detector {
	public:
		inline Detector() {
			this->_detector = ESVoiceActivityDetectorNew();
		}
		inline ~Detector() {
			ESVoiceActivityDetectorRelease(this->_detector);
		}

		inline void reset() {
			ESVoiceActivityDetectorReset(this->_detector);
		}

		/**
		 * Predicts the voice activity score of a single input frame of 16-bit PCM audio.
		 *
		 * The frame:
		 * - should be sampled at 16 KHz;
		 * - should be exactly 256 samples (so 16 ms) in length.
		 *
		 * The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
		 * can be adjusted according to application-specific needs.
		 */
		inline float predict(int16_t *frame, size_t frame_len) {
			return ESVoiceActivityDetectorPredictI16(this->_detector, frame, frame_len);
		}

		/**
		 * Predicts the voice activity score of a single input frame of 32-bit floating-point PCM audio.
		 *
		 * The frame:
		 * - should be sampled at 16 KHz;
		 * - should be exactly 256 samples (so 16 ms) in length;
		 * - should consist only of samples in the range [-1, 1].
		 *
		 * The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
		 * can be adjusted according to application-specific needs.
		 */
		inline float predict(float *frame, size_t frame_len) {
			return ESVoiceActivityDetectorPredictF32(this->_detector, frame, frame_len);
		}

		/**
		 * Predicts the voice activity score of a single input frame of 16-bit PCM audio.
		 *
		 * The frame:
		 * - should be sampled at 16 KHz;
		 * - should be exactly 256 samples (so 16 ms) in length.
		 *
		 * The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
		 * can be adjusted according to application-specific needs.
		 */
		inline float predict(std::vector<int16_t> &frame) {
			return ESVoiceActivityDetectorPredictI16(this->_detector, frame.data(), frame.size());
		}

		/**
		 * Predicts the voice activity score of a single input frame of 32-bit floating-point PCM audio.
		 *
		 * The frame:
		 * - should be sampled at 16 KHz;
		 * - should be exactly 256 samples (so 16 ms) in length;
		 * - should consist only of samples in the range [-1, 1].
		 *
		 * The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
		 * can be adjusted according to application-specific needs.
		 */
		inline float predict(std::vector<float> &frame) {
			return ESVoiceActivityDetectorPredictF32(this->_detector, frame.data(), frame.size());
		}

	private:
		ESVoiceActivityDetector *_detector;
	};
}
#endif

#endif
