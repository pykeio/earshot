# Earshot
Ridiculously fast & accurate voice activity detection in pure Rust.

Achieves an RTF of 0.0007 (1,270x real time): **20x faster** than Silero VAD v6 & TEN VAD - and more accurate, too!

> If you find Earshot useful, please consider [sponsoring pyke.io](https://opencollective.com/pyke-osai).

<img src="https://i.pyke.io/earshot-1.0-pr.png"/>

## Usage

```rs
use earshot::Detector;

// Create a new VAD detector using the default NN.
let mut detector = Detector::default();

let mut frame_receiver = ...
while let Some(frame) = frame_receiver.recv() {
	// `frame` is Vec<i16> with length 256. Each frame passed to the detector must be exactly 256 samples.
	// f32 [-1, 1] frames are also supported with `predict_f32`.
	let score = detector.predict_i16(&frame);
	// Score is between 0-1; 0 = no voice, 1 = voice.
	if score >= 0.5 { // 0.5 is a good default threshold, but can be customized.
		println!("Voice detected!");
	}
}
```

## Binary & memory size
Earshot is very embedded-friendly: each instance of `Detector` uses ~8 KiB of memory to store the audio buffer & neural network state. Binary footprint is ~100 KiB; the neural network is 75 KiB of that.

In contrast, Silero's model is 2 MiB, TEN's is 310 KiB, but both require ONNX Runtime, which adds an additional 8 MB to your binary (+ a whole lot more memory).

## `#![no_std]`
Earshot supports `#![no_std]`, but it does require an allocator. The `std` feature is enabled by default, so add `default-features = false` to enable `#![no_std]`:

```toml
[dependencies]
earshot = { version = "1", default-features = false }
```
