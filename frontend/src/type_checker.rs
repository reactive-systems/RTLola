use std::collections::HashMap;
use crate::parse::NodeId;
use crate::type_checker::types::Ty;

pub(crate) mod types;
pub(crate) mod check;
pub(crate) mod unification;

#[derive(Debug, Clone, Copy, Default)]
pub struct TypeConfig {
    /// allow only 64bit types, i.e., `Int64`, `UInt64`, and `Float64`
    pub use_64bit_only: bool,
    /// include type aliases `Int` -> `Int64`, `UInt` -> `UInt64`, and `Float` -> `Float64`
    pub type_aliases: bool,
}

pub(crate) struct TypeTable {
    tt: HashMap<NodeId, Ty>
}

impl TypeTable {
    pub(crate) fn get_value_type(&self, nid: NodeId) -> ! {
        unimplemented!()
    }

    pub(crate) fn get_stream_type(&self, nid: NodeId) -> ! {
        unimplemented!()
    }

    pub(crate) fn get_func_arg_types(&self, nid: NodeId) -> ! {
        unimplemented!()
    }

    pub(crate) fn get_acti_cond(&self, nid: NodeId) -> ! {
        unimplemented!()
    }
}
