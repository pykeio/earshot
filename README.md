# Earshot
Ridiculously fast, only slightly bad voice activity detection in pure Rust. Port of the famous [WebRTC VAD](https://webrtc.googlesource.com/).

## Features
- `#![no_std]`, doesn't even require `alloc`
	* Internal buffers can get pretty big when stored on the stack, so the `alloc` feature is enabled by default, which allocates them on the heap instead.
- Stupidly fast; uses only fixed-point arithmetic
	* Achieves an RTF of ~3e-4 with 30 ms 48 KHz frames, ~3e-5 with 30 ms 8 KHz frames.
	* Comparatively, Silero VAD v4 w/ [`ort`](https://ort.pyke.io/) achieves an RTF of ~3e-3 with 60 ms 16 KHz frames.
- Okay accuracy
	* Great at distinguishing between silence and noise, but not between noise and speech.
	* Earshot provides alternative models with slight accuracy gains compared to the base WebRTC model.
