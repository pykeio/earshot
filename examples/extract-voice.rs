use core::{mem, ptr, slice};
use std::{
	env::args,
	fs::{self, File},
	io::Write
};

use earshot::{Detector, QuantizedPredictor};

fn main() {
	let mut args = args().skip(1);
	let Some(input) = args.next() else {
		eprintln!("cargo run --example extract-voice -- [wav] [out]");
		return;
	};
	let Some(output) = args.next() else {
		eprintln!("cargo run --example extract-voice -- [wav] [out]");
		return;
	};

	let mut detector = Detector::<QuantizedPredictor>::default();

	let mut out = File::create(output).unwrap();

	let wav = fs::read(input).unwrap();
	for x in wav[44..].chunks_exact(512) {
		let mut samples = vec![0; 256];
		for i in 0..256 {
			samples[i] = i16::from_le_bytes([x[(i * 2)], x[(i * 2) + 1]]);
		}

		let score = detector.predict_i16(&samples);
		if score >= 0.5 {
			println!("voice");
			out.write_all(&x).unwrap();
		} else {
			println!("silence {score}");
		}
	}

	out.flush().unwrap();
}
