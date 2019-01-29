
use crate::evaluator::{OutInstance, Window};
use super::Value;

use lola_parser::{LolaIR, StreamReference, Type, WindowOperation};
use std::time::SystemTime;

pub(crate) struct GlobalStore {

}

impl GlobalStore {
    pub(crate) fn new(ir: &LolaIR) -> GlobalStore {
        unimplemented!()
    }

    pub(crate) fn get_in_instance(&self, sr: StreamReference) -> InstanceStore {
        unimplemented!()
    }

    pub(crate) fn get_out_instance(&self, inst: OutInstance) -> InstanceStore {
        unimplemented!()
    }

    pub(crate) fn get_window(&self, window: Window) -> WindowStore {
        unimplemented!()
    }
}

pub(crate) struct InstanceStore {
    // TODO: Parametrize by value?
    // InstanceStore<u64> ...?
}

impl InstanceStore {
    pub(crate) fn new(for_type: &Type) -> InstanceStore {
        unimplemented!()
    }

    pub(crate) fn get_value(&self, offset: i16) -> Value {
        unimplemented!()
    }

    pub(crate) fn push_value(&self, v: Value) {
        unimplemented!()
    }
}

pub(crate) struct WindowStore {

}

impl WindowStore {
    pub(crate) fn new(kind: WindowOperation) -> WindowStore {
        unimplemented!()
    }

    pub(crate) fn get_value(&self) -> Value {
        unimplemented!()
    }

    pub(crate) fn push_value(&self, v: Value, ts: SystemTime) {
        unimplemented!()
    }
}