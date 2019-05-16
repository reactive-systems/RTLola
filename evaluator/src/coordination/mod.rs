mod controller;
mod event_driven_manager;
mod time_driven_manager;

use std::time::SystemTime;

// Re-exports
pub(crate) use self::controller::Controller;
pub(crate) use self::event_driven_manager::EventEvaluation;
pub(crate) use self::time_driven_manager::TimeEvaluation;

#[derive(Debug)]
pub(crate) enum WorkItem {
    Event(EventEvaluation, SystemTime),
    Time(TimeEvaluation, SystemTime),
    Start(SystemTime),
    End,
}
