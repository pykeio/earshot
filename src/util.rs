use core::{
	cell::UnsafeCell,
	hint::spin_loop,
	marker::PhantomData,
	mem::{MaybeUninit, forget},
	sync::atomic::{AtomicU8, Ordering}
};

pub(crate) struct OnceLock<T> {
	data: UnsafeCell<MaybeUninit<T>>,
	status: AtomicU8,
	phantom: PhantomData<T>
}

unsafe impl<T: Send> Send for OnceLock<T> {}
unsafe impl<T: Send + Sync> Sync for OnceLock<T> {}

const STATUS_UNINITIALIZED: u8 = 0;
const STATUS_RUNNING: u8 = 1;
const STATUS_INITIALIZED: u8 = 2;

impl<T> OnceLock<T> {
	pub const fn new() -> Self {
		Self {
			data: UnsafeCell::new(MaybeUninit::uninit()),
			status: AtomicU8::new(STATUS_UNINITIALIZED),
			phantom: PhantomData
		}
	}

	#[inline]
	unsafe fn get_unchecked(&self) -> &T {
		&*(*self.data.get()).as_ptr()
	}

	#[inline]
	pub fn get(&self) -> Option<&T> {
		match self.status.load(Ordering::Acquire) {
			STATUS_INITIALIZED => Some(unsafe { self.get_unchecked() }),
			_ => None
		}
	}

	#[inline]
	pub fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> &T {
		if let Some(value) = self.get() { value } else { self.init_inner(f) }
	}

	#[cold]
	fn init_inner<F: FnOnce() -> T>(&self, f: F) -> &T {
		'a: loop {
			match self
				.status
				.compare_exchange(STATUS_UNINITIALIZED, STATUS_RUNNING, Ordering::Acquire, Ordering::Acquire)
			{
				Ok(_) => {
					struct SetStatusOnPanic<'a> {
						status: &'a AtomicU8
					}
					impl Drop for SetStatusOnPanic<'_> {
						fn drop(&mut self) {
							self.status.store(STATUS_UNINITIALIZED, Ordering::SeqCst);
						}
					}

					let panic_catcher = SetStatusOnPanic { status: &self.status };
					let val = f();
					unsafe {
						(*self.data.get()).as_mut_ptr().write(val);
					};
					forget(panic_catcher);

					self.status.store(STATUS_INITIALIZED, Ordering::Release);

					return unsafe { self.get_unchecked() };
				}
				Err(STATUS_INITIALIZED) => return unsafe { self.get_unchecked() },
				Err(STATUS_RUNNING) => loop {
					match self.status.load(Ordering::Acquire) {
						STATUS_RUNNING => spin_loop(),
						STATUS_INITIALIZED => return unsafe { self.get_unchecked() },
						_ => continue 'a
					}
				},
				_ => continue
			}
		}
	}
}
