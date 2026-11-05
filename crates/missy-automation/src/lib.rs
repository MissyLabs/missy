//! Explicit opt-in automation. No background work is started by opening a scheduler.
//! Planning and skill discovery are data-only; neither grants tool authority.
mod planning;
mod scheduler;
mod skills;

pub use planning::*;
pub use scheduler::*;
pub use skills::*;
