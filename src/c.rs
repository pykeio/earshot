use core::slice;

use crate::{DefaultPredictor, Detector};

pub enum ESVoiceActivityDetector {}

#[unsafe(no_mangle)]
pub extern "C" fn ESVoiceActivityDetectorSize() -> usize {
	size_of::<Detector>()
}

#[unsafe(no_mangle)]
pub extern "C" fn ESVoiceActivityDetectorInit(detector: *mut ESVoiceActivityDetector) {
	unsafe { *detector.cast() = Detector::new(DefaultPredictor::new()) };
}

#[unsafe(no_mangle)]
pub extern "C" fn ESVoiceActivityDetectorNew() -> *mut ESVoiceActivityDetector {
	(Box::leak(Detector::new_boxed(DefaultPredictor::new())) as *mut Detector).cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn ESVoiceActivityDetectorRelease(detector: *mut ESVoiceActivityDetector) {
	let _ = unsafe { Box::from_raw(detector.cast::<Detector>()) };
}

#[unsafe(no_mangle)]
pub extern "C" fn ESVoiceActivityDetectorReset(detector: *mut ESVoiceActivityDetector) {
	unsafe { &mut *detector.cast::<Detector>() }.reset();
}

#[unsafe(no_mangle)]
pub extern "C" fn ESVoiceActivityDetectorPredictI16(detector: *mut ESVoiceActivityDetector, frame: *const i16, frame_len: usize) -> f32 {
	unsafe { &mut *detector.cast::<Detector>() }.predict_i16(unsafe { slice::from_raw_parts(frame, frame_len) })
}

#[unsafe(no_mangle)]
pub extern "C" fn ESVoiceActivityDetectorPredictF32(detector: *mut ESVoiceActivityDetector, frame: *const f32, frame_len: usize) -> f32 {
	unsafe { &mut *detector.cast::<Detector>() }.predict_f32(unsafe { slice::from_raw_parts(frame, frame_len) })
}
