
use crate::evaluator::{OutInstance, Window};
use super::Value;

use lola_parser::{LolaIR, StreamReference, Type, WindowOperation, OutputStream};
use std::time::Instant;
use std::collections::HashMap;
use std::collections::VecDeque;
use super::window::SlidingWindow;

pub(crate) type Parameter = Vec<Value>;

pub(crate) struct GlobalStore {

    /// Access by stream reference.
    inputs: Vec<InstanceStore>,

    /// Transforms a output stream reference into the respective index of the stream vector.
    index_map: Vec<usize>,

    /// Non-parametrized outputs. Access by index.
    np_outputs: Vec<InstanceStore>,

    /// Parametrized outputs. Access by index, followed by parameters.
    p_outputs: Vec<HashMap<Parameter, InstanceStore>>,

    /// Non-parametrized windows, access by WindowReference.
    np_windows: Vec<SlidingWindow>,

}

impl GlobalStore {
    pub(crate) fn new(ir: &LolaIR, ts: Instant) -> GlobalStore {
        let mut index_map: Vec<Option<usize>> = vec![None; ir.outputs.len()];
        let mut p_cnt = 0usize;
        for p in &ir.parametrized {
            index_map[p.reference.out_ix()] = Some(p_cnt);
            p_cnt += 1;
        }
        let nps: Vec<&OutputStream> = index_map.iter_mut()
            .enumerate()
            .flat_map(|(ix, v)| if v.is_some() { None } else { Some(ix) })
            .map(|ix| &ir.outputs[ix])
            .collect(); // Give it a type.

        nps.iter().map(|o| o.reference.out_ix())
            .enumerate()
            .for_each(|(np_ix, ir_ix)| index_map[ir_ix] = Some(np_ix));
        assert!(index_map.iter().all(Option::is_some));

        let index_map = index_map.into_iter().flatten().collect();
        let np_outputs = nps.iter().map(|o| &o.ty).map(InstanceStore::new).collect();
        let p_outputs = vec![HashMap::new(); p_cnt];
        let inputs = ir.inputs.iter().map(|i| &i.ty).map(InstanceStore::new).collect();
        let np_windows = ir.sliding_windows.iter().map(|w| SlidingWindow::new(w.duration, w.op, ts)).collect();

        GlobalStore { inputs, index_map, np_outputs, p_outputs, np_windows }
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
            self.p_outputs[self.index_map[ix]].get(&p)
        }
    }

    pub(crate) fn get_out_instance_mut(&mut self, inst: OutInstance) -> Option<&mut InstanceStore> {
        let (ix, p) = inst;
        if p.is_empty() {
            Some(&mut self.np_outputs[self.index_map[ix]])
        } else {
            self.p_outputs[self.index_map[ix]].get_mut(&p)
        }
    }

    pub(crate) fn get_window_mut(&mut self, window: Window) -> &mut SlidingWindow {
        let (ix, p, _) = window;
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
        InstanceStore {
            buffer: VecDeque::with_capacity(SIZE),
        }
    }

    pub(crate) fn get_value(&self, offset: i16) -> Option<Value> {
        assert!(offset <= 0);
        let ix = self.buffer.len() - (offset as usize) - 1;
        self.buffer.get(ix).cloned()
    }

    pub(crate) fn push_value(&mut self, v: Value) {
        if self.buffer.len() == SIZE {
            self.buffer.pop_front();
        }
        self.buffer.push_back(v);
    }
}
