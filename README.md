# Earshot
Ridiculously fast & accurate streaming voice activity detection, written in pure Rust and also available for [Python](https://pypi.org/project/earshot/).

Earshot achieves an RTF of 0.0003 (3,600x real time): **40x faster** than Silero VAD v6 & TEN VAD - and more accurate, too!

Earshot operates on 16 millisecond frames of mono audio sampled at 16000 Hz & supports streaming. Earshot detects voice in any language and is resilient to most kinds of environmental noise with an SNR ≥ 3dB.

> If you find Earshot useful, please consider [sponsoring pyke.io](https://opencollective.com/pyke-osai).

<figure>
<img src="https://i.pyke.io/earshot-acc-1.3.png"/>
<figcaption><i>
Earshot, in black, performs markedly better than Silero VAD v6 and TEN VAD in blue and red.
</i></figcaption>
</figure>

## Usage
- **Python**: [`pip install earshot`](https://pypi.org/project/earshot/)

### Rust
> [`cargo add earshot`](https://crates.io/crates/earshot)

```rs
use earshot::Detector;

// Create a new VAD detector using the default NN.
let mut detector = Detector::default();

let mut frame_receiver = ...
while let Some(frame) = frame_receiver.recv() {
	// `frame` is Vec<i16> with length 256.
	// Each frame passed to the detector must be exactly 256 samples (16ms) @ 16 KHz sample rate.
	// f32 [-1, 1] frames are also supported with `predict_f32`.
	let score = detector.predict_i16(&frame);
	// Score is between 0-1; 0 = no voice, 1 = voice.
	if score >= 0.5 { // 0.5 is a good default threshold, but can be customized.
		println!("Voice detected!");
	}
}
```

## Binary & memory size
Earshot is very embedded-friendly: each instance of `Detector` uses ~8 KiB of memory to store the audio buffer & neural network state. Binary footprint is ~95 KiB; the neural network is 40 KiB of that.

In contrast, Silero's model is 2 MiB, TEN's is 310 KiB, but both require ONNX Runtime, which adds an additional 8 MB to your binary (+ a whole lot more memory).

## `#![no_std]`
Earshot supports `#![no_std]`, but it does require the [`libm`](https://crates.io/crates/libm) crate. The `std` feature is enabled by default, so add `default-features = false` and `features = [ "libm" ]` to enable `#![no_std]`:

```toml
[dependencies]
earshot = { version = "1", default-features = false, features = [ "libm" ] }
```
