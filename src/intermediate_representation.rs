
pub struct IntermediateRepresentation {
    pub inputs: Vec<InputStream>,
    pub event_outputs: Vec<OutputStream>,
    pub time_outputs: Vec<OutputStream>,
    pub feature_flags: Vec<FeatureFlag>,
    pub sliding_windows: Vec<SlidingWindow>,
}

impl<'a> IntermediateRepresentation {
    pub fn outputs(&'a self) -> Vec<Stream<'a>> {
        self.event_outputs
            .iter()
            .chain(self.time_outputs.iter())
            .map(|s| Stream::Out(s))
            .collect()
    }
    pub fn get(&'a self, reference: StreamReference) -> Stream<'a> {
        match reference {
            StreamReference::InRef(ix) => Stream::In(&self.inputs[ix]),
            StreamReference::OutRef { timed: true, ix } => Stream::Out(&self.time_outputs[ix]),
            StreamReference::OutRef { timed: false, ix } => Stream::Out(&self.event_outputs[ix]),
        }
    }
}

pub enum StreamReference {
    InRef(usize),
    OutRef { timed: bool, ix: usize },
}

pub struct SlidingWindow {
    target: StreamReference,
    duration: Duration,
}

pub struct InputStream {}

pub struct OutputStream {}

pub enum Stream<'a> {
    In(&'a InputStream),
    Out(&'a OutputStream),
}

/// Specifies a duration as well as a rate, normalized to ms.
struct Duration {
    expr: Expression,
    factor: f64,
}

struct Expression {}

#[derive(Clone, Copy, Debug)]
pub enum FeatureFlag {
    DiscreteFutureOffset,
    RealTimeOffset,
    RealTimeFutureOffset,
    SlidingWindows,
    DiscreteWindows,
}
