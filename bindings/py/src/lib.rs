use std::borrow::Cow;

use earshot::{DefaultPredictor, Detector as RustDetector};
use numpy::PyArrayLike1;
use pyo3::{exceptions::PyTypeError, prelude::*};

#[pyclass(module = "earshot._earshot")]
struct Detector {
	inner: Box<RustDetector<DefaultPredictor>>
}

#[pymethods]
impl Detector {
	#[new]
	fn new() -> Self {
		Self { inner: RustDetector::default_boxed() }
	}

	fn reset(&mut self) {
		self.inner.reset();
	}

	fn predict_i16(&mut self, frame: PyArrayLike1<'_, i16>) -> PyResult<f32> {
		let frame = frame.as_array();
		let frame = frame
			.as_slice()
			.map(Cow::Borrowed)
			.unwrap_or_else(|| Cow::Owned(frame.iter().copied().collect::<Vec<_>>()));
		if frame.len() != 256 {
			return Err(PyErr::new::<PyTypeError, _>("frame must be exactly 256 samples"));
		}
		Ok(self.inner.predict_i16(&frame))
	}

	fn predict_f32(&mut self, frame: PyArrayLike1<'_, f32>) -> PyResult<f32> {
		let frame = frame.as_array();
		let frame = frame
			.as_slice()
			.map(Cow::Borrowed)
			.unwrap_or_else(|| Cow::Owned(frame.iter().copied().collect::<Vec<_>>()));
		if frame.len() != 256 {
			return Err(PyErr::new::<PyTypeError, _>("frame must be exactly 256 samples"));
		}
		Ok(self.inner.predict_f32(&frame))
	}
}

#[pymodule]
fn _earshot(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<Detector>()?;
	m.add("__version__", env!("CARGO_PKG_VERSION"))?;
	Ok(())
}
