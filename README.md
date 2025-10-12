# Earshot
Ridiculously fast & accurate voice activity detection in pure Rust.

Achieves an RTF of 0.0014; 10x faster than Silero/TEN VAD.

## Performance
Compiling with `RUSTFLAGS="-C target-cpu=native"` in release mode is highly recommended as it can cut processing time in half.
