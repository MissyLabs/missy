//! Explicit opt-in automation. No background work is started by opening a scheduler.
//! Planning and skill discovery are data-only; neither grants tool authority.
mod execution;
mod maintenance;
mod observations;
mod planning;
mod scheduler;
mod skills;
mod tools;
mod triggers;

pub use execution::*;
pub use maintenance::*;
pub use observations::*;
pub use planning::*;
pub use scheduler::*;
pub use skills::*;
pub use tools::*;
pub use triggers::*;
