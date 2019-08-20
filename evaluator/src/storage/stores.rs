use super::Value;

use crate::basics::Time;
use crate::storage::SlidingWindow;
use std::collections::VecDeque;
use streamlab_frontend::ir::{
    InputReference, LolaIR, MemorizationBound, OutputReference, OutputStream, Type, WindowReference,
};

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

pub(crate) type InInstance = InputReference;
pub(crate) type OutInstance = OutputReference;

impl GlobalStore {
    pub(crate) fn new(ir: &LolaIR, ts: Time) -> GlobalStore {
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
        let np_outputs = nps.iter().map(|o| InstanceStore::new(&o.ty, o.memory_bound)).collect();
        let inputs = ir.inputs.iter().map(|i| InstanceStore::new(&i.ty, i.memory_bound)).collect();
        let np_windows =
            ir.sliding_windows.iter().map(|w| SlidingWindow::new(w.duration, w.wait, w.op, ts, &w.ty)).collect();

        GlobalStore { inputs, index_map, np_outputs, np_windows }
    }

    pub(crate) fn get_in_instance(&self, inst: InInstance) -> &InstanceStore {
        let ix = inst;
        &self.inputs[ix]
    }

    pub(crate) fn get_in_instance_mut(&mut self, inst: InInstance) -> &mut InstanceStore {
        let ix = inst;
        &mut self.inputs[ix]
    }

    pub(crate) fn get_out_instance(&self, inst: OutInstance) -> Option<&InstanceStore> {
        let ix = inst;
        Some(&self.np_outputs[self.index_map[ix]])
    }

    pub(crate) fn get_out_instance_mut(&mut self, inst: OutInstance) -> Option<&mut InstanceStore> {
        let ix = inst;
        Some(&mut self.np_outputs[self.index_map[ix]])
    }

    pub(crate) fn get_window(&self, window: WindowReference) -> &SlidingWindow {
        let ix = window.idx();
        &self.np_windows[ix]
    }

    pub(crate) fn get_window_mut(&mut self, window: WindowReference) -> &mut SlidingWindow {
        let ix = window.idx();
        &mut self.np_windows[ix]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InstanceStore {
    // New elements get stored at the front
    buffer: VecDeque<Value>,
    bound: MemorizationBound,
}

const SIZE: usize = 256;

impl InstanceStore {
    // _type might be used later.
    pub(crate) fn new(_type: &Type, bound: MemorizationBound) -> InstanceStore {
        match bound {
            MemorizationBound::Bounded(limit) => {
                InstanceStore { buffer: VecDeque::with_capacity(limit as usize), bound }
            }
            MemorizationBound::Unbounded => InstanceStore { buffer: VecDeque::with_capacity(SIZE), bound },
        }
    }

    pub(crate) fn get_value(&self, offset: i16) -> Option<Value> {
        assert!(offset <= 0);
        if offset == 0 {
            self.buffer.front().cloned()
        } else {
            let offset = offset.abs() as usize;
            self.buffer.get(offset).cloned()
        }
    }

    pub(crate) fn push_value(&mut self, v: Value) {
        if let MemorizationBound::Bounded(limit) = self.bound {
            if self.buffer.len() == limit as usize {
                self.buffer.pop_back();
            }
        }
        self.buffer.push_front(v);
    }
}
