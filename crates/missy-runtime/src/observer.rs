//! Payload-free observation of actual runtime work. Observer failures never alter execution.
use std::{
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeEventKind {
    Turn,
    Provider,
    Tool,
    Settlement,
}

/// A completed span. IDs are freshly generated, never derived from tenant/user/turn data.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeEvent {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub parent_span_id: Option<[u8; 8]>,
    pub kind: RuntimeEventKind,
    pub start_time_unix_nano: u64,
    pub end_time_unix_nano: u64,
    pub succeeded: bool,
}

/// Implementations MUST return immediately and must not perform network/disk I/O.
pub trait RuntimeObserver: Send + Sync {
    fn event(&self, event: RuntimeEvent);
}

struct TraceContext {
    observer: Arc<dyn RuntimeObserver>,
    trace_id: [u8; 16],
    root_id: [u8; 8],
}
tokio::task_local! { static TRACE: TraceContext; }

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
fn id() -> [u8; 8] {
    uuid::Uuid::new_v4().as_bytes()[..8]
        .try_into()
        .expect("fixed UUID length")
}

pub(crate) struct Span {
    observer: Option<Arc<dyn RuntimeObserver>>,
    event: RuntimeEvent,
}
impl Span {
    pub(crate) fn child(kind: RuntimeEventKind) -> Self {
        let (observer, trace_id, parent) = TRACE
            .try_with(|t| (Some(t.observer.clone()), t.trace_id, Some(t.root_id)))
            .unwrap_or((None, [0; 16], None));
        Self {
            observer,
            event: RuntimeEvent {
                trace_id,
                span_id: id(),
                parent_span_id: parent,
                kind,
                start_time_unix_nano: now(),
                end_time_unix_nano: 0,
                succeeded: false,
            },
        }
    }
    pub(crate) fn finish(mut self, succeeded: bool) {
        self.event.succeeded = succeeded;
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        self.event.end_time_unix_nano = now().max(self.event.start_time_unix_nano);
        if let Some(observer) = &self.observer {
            let event = self.event;
            let _ = catch_unwind(AssertUnwindSafe(|| observer.event(event)));
        }
    }
}

pub(crate) async fn observe_run<F: Future<Output = anyhow::Result<crate::RunResult>>>(
    observer: Option<Arc<dyn RuntimeObserver>>,
    future: F,
) -> anyhow::Result<crate::RunResult> {
    let Some(observer) = observer else {
        return future.await;
    };
    let context = TraceContext {
        observer,
        trace_id: *uuid::Uuid::new_v4().as_bytes(),
        root_id: id(),
    };
    TRACE
        .scope(context, async {
            let mut span = Span::child(RuntimeEventKind::Turn);
            TRACE.with(|t| {
                span.event.span_id = t.root_id;
                span.event.parent_span_id = None;
            });
            let result = future.await;
            span.finish(result.as_ref().is_ok_and(|r| r.status == "completed"));
            result
        })
        .await
}
