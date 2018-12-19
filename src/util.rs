use crate::LolaSpec;
use ast_node::AstNode;
use ast_node::NodeId;
use ast_node::Span;

pub(crate) fn get_node_id_from_spec(spec: &LolaSpec, span: &Span) -> Option<NodeId> {
    for input in &spec.inputs {
        if input.span() == span {
            return Some(*input.id());
        }
    }
    for output in &spec.outputs {
        if output.span() == span {
            return Some(*output.id());
        }
    }
    for trigger in &spec.trigger {
        if trigger.span() == span {
            return Some(*trigger.id());
        }
    }

    unimplemented!("We do not support all nodes")
}
