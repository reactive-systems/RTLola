use super::Value;
use crate::evaluator::{OutInstance, Window};

use crate::storage::SlidingWindow;
use std::collections::VecDeque;
use std::time::SystemTime;
use streamlab_frontend::ir::{LolaIR, OutputStream, StreamReference, Type};

pub(crate) struct GlobalStore {
    /// Access by stream reference.
    inputs: Vec<InstanceStore>,

    /// Transforms a output stream reference into the respective index of the stream vectors ((non-)parametrized).
    index_map: Vec<usize>,

    /// Non-parametrized outputs. Access by index.
    np_outputs: Vec<InstanceStore>,

    /// Non-parametrized windows, access by WindowReference.
    np_windows: Vec<SlidingWindow>,
}

impl GlobalStore {
    pub(crate) fn new(ir: &LolaIR, ts: SystemTime) -> GlobalStore {
        let mut index_map: Vec<Option<usize>> = vec![None; ir.outputs.len()];

        let nps: Vec<&OutputStream> = index_map
            .iter()
            .enumerate()
            .flat_map(|(ix, v)| if v.is_none() { Some(ix) } else { None })
            .map(|ix| &ir.outputs[ix])
            .collect(); // Give it a type.

        for (np_ix, o) in nps.iter().enumerate() {
            index_map[o.reference.out_ix()] = Some(np_ix);
        }

        assert!(index_map.iter().all(Option::is_some));

        let index_map = index_map.into_iter().flatten().collect();
        let np_outputs = nps.iter().map(|o| InstanceStore::new(&o.ty)).collect();
        let inputs = ir.inputs.iter().map(|i| InstanceStore::new(&i.ty)).collect();
        let np_windows = ir.sliding_windows.iter().map(|w| SlidingWindow::new(w.duration, w.op, ts, &w.ty)).collect();

        GlobalStore { inputs, index_map, np_outputs, np_windows }
    }

    pub(crate) fn get_in_instance(&self, sr: StreamReference) -> &InstanceStore {
        &self.inputs[sr.in_ix()]
    }

    pub(crate) fn get_in_instance_mut(&mut self, sr: StreamReference) -> &mut InstanceStore {
        &mut self.inputs[sr.in_ix()]
    }

    pub(crate) fn get_out_instance(&self, inst: OutInstance) -> Option<&InstanceStore> {
        let (ix, p) = inst;
        if p.is_empty() {
            Some(&self.np_outputs[self.index_map[ix]])
        } else {
            unimplemented!("Parametrized streams not implemented.")
        }
    }

    pub(crate) fn get_out_instance_mut(&mut self, inst: OutInstance) -> Option<&mut InstanceStore> {
        let (ix, p) = inst;
        if p.is_empty() {
            Some(&mut self.np_outputs[self.index_map[ix]])
        } else {
            unimplemented!("Parametrized streams not implemented.")
        }
    }

    pub(crate) fn get_window(&self, window: Window) -> &SlidingWindow {
        let (ix, p) = window;
        assert!(p.is_empty());
        &self.np_windows[ix]
    }

    pub(crate) fn get_window_mut(&mut self, window: Window) -> &mut SlidingWindow {
        let (ix, p) = window;
        assert!(p.is_empty());
        &mut self.np_windows[ix]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InstanceStore {
    buffer: VecDeque<Value>,
}

const SIZE: usize = 256; // TODO!

impl InstanceStore {
    // _for type might be used later.
    pub(crate) fn new(_for_type: &Type) -> InstanceStore {
        InstanceStore { buffer: VecDeque::with_capacity(SIZE) }
    }

    pub(crate) fn get_value(&self, offset: i16) -> Option<Value> {
        assert!(offset <= 0);
        let offset = offset.abs() as usize;
        if self.buffer.len() < (offset + 1) {
            None
        } else {
            let ix = self.buffer.len() - offset - 1;
            self.buffer.get(ix).cloned()
        }
    }

    pub(crate) fn push_value(&mut self, v: Value) {
        if self.buffer.len() == SIZE {
            self.buffer.pop_front();
        }
        self.buffer.push_back(v);
    }
}
