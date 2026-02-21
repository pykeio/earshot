use std::{
	env::args,
	fs::{self, File},
	io::Write
};

use earshot::Detector;

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

	let mut detector = Detector::default();

	let mut out = File::create(output).unwrap();

	let wav = fs::read(input).unwrap();
	for x in wav[44..].chunks_exact(512) {
		let mut samples = vec![0; 256];
		for i in 0..256 {
			samples[i] = i16::from_le_bytes([x[i * 2], x[(i * 2) + 1]]);
		}

		let score = detector.predict_i16(&samples);
		if score >= 0.5 {
			out.write_all(&x).unwrap();
		} else {
		}
	}

	out.flush().unwrap();
}
