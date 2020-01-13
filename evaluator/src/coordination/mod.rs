mod controller;
mod event_driven_manager;
mod monitor;
mod time_driven_manager;

// Re-exports
pub(crate) use self::controller::Controller;
pub(crate) use self::event_driven_manager::EventEvaluation;
pub use self::monitor::Monitor;
pub(crate) use self::time_driven_manager::TimeEvaluation;
use crate::basics::Time;
use crate::storage::Value;

pub type Event = Vec<Value>;

#[derive(Debug, Clone)]
pub(crate) enum WorkItem {
    Event(EventEvaluation, Time),
    Time(TimeEvaluation, Time),
    End,
}

pub(crate) const CAP_WORK_QUEUE: usize = 8;
pub(crate) const CAP_LOCAL_QUEUE: usize = 4096;
