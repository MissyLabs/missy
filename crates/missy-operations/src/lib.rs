//! Explicit operational surfaces. Telemetry is opt-in; no implicit listeners.
mod diagnostics;
mod eventbus;
mod export;
mod telemetry;
mod webhook;
mod workflow;
pub use diagnostics::*;
pub use eventbus::*;
pub use export::*;
pub use telemetry::*;
pub use webhook::*;
pub use workflow::*;
pub type SharedRuntime = std::sync::Arc<missy_runtime::Runtime>;
#[cfg(test)]
mod tests;
