//! The CLI's GPU (its wgpu, which the export renderer uses) keeps error scopes per thread.
//!
//! An export runs up to four GPU renderers on its worker threads, and each builds the pipelines
//! of the transitions it meets inside an error scope. wgpu built without its `std` feature (the
//! CLI's build until 2026-10-21: nothing asked for it) keeps one scope stack for the whole
//! process, so two renderers popped each other's scopes and wgpu panicked ("error scopes must be
//! popped in reverse order"): the export finished on the CPU, 298 of 300 frames of a showcase range.

use std::future::Future;
use std::sync::Barrier;

/// A future that native wgpu resolves at once.
fn now<F: Future>(f: F) -> Option<F::Output> {
    let mut f = std::pin::pin!(f);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    match f.as_mut().poll(&mut cx) {
        std::task::Poll::Ready(v) => Some(v),
        std::task::Poll::Pending => None,
    }
}

#[test]
fn error_scopes_are_per_thread() {
    if wgpu::Instance::enabled_backend_features().is_empty() {
        eprintln!("SKIPPED (error scopes): no wgpu backend");
        return;
    }
    let instance = wgpu::Instance::default();
    let Some(Ok(adapter)) = now(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
        eprintln!("SKIPPED (error scopes): no adapter");
        return;
    };
    let Some(Ok((device, _queue))) = now(adapter.request_device(&wgpu::DeviceDescriptor::default())) else {
        eprintln!("SKIPPED (error scopes): no device");
        return;
    };
    // thread A pushes, thread B pushes, A pops, B pops: per-thread stacks take that order
    let (pushed_a, pushed_b, popped_a) = (Barrier::new(2), Barrier::new(2), Barrier::new(2));
    let (a, b) = std::thread::scope(|s| {
        let a = s.spawn(|| {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            pushed_a.wait();
            pushed_b.wait();
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| now(scope.pop()).is_some()));
            popped_a.wait();
            r
        });
        let b = s.spawn(|| {
            pushed_a.wait();
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            pushed_b.wait();
            popped_a.wait();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| now(scope.pop()).is_some()))
        });
        (a.join(), b.join())
    });
    assert!(matches!(a, Ok(Ok(true))), "thread A's scope: {a:?}");
    assert!(matches!(b, Ok(Ok(true))), "thread B's scope: {b:?}");
}
