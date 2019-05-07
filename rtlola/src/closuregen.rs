//! An attempt to implement dynamic dispatch codegen
//!
//! See [Building fast intepreters in Rust](https://blog.cloudflare.com/building-fast-interpreters-in-rust/)

use crate::evaluator::EvaluationContext;
use crate::storage::Value;
use streamlab_frontend::ir::Expression;

pub(crate) trait Expr<'s> {
    fn compile(self) -> CompiledExpr<'s>;
}

pub(crate) struct CompiledExpr<'s>(Box<dyn 's + Fn(&EvaluationContext<'_>) -> Value>);
// alternative: using Higher-Rank Trait Bounds (HRTBs)
// pub(crate) struct CompiledExpr<'s>(Box<dyn 's + for<'a> Fn(&EvaluationContext<'a>) -> Value>);

impl<'s> CompiledExpr<'s> {
    /// Creates a compiled expression IR from a generic closure.
    pub(crate) fn new(closure: impl 's + Fn(&EvaluationContext<'_>) -> Value) -> Self {
        CompiledExpr(Box::new(closure))
    }

    /// Executes a filter against a provided context with values.
    pub fn execute(&self, ctx: &EvaluationContext<'s>) -> Value {
        self.0(ctx)
    }
}

impl<'s> Expr<'s> for Expression {
    fn compile(self) -> CompiledExpr<'s> {
        CompiledExpr::new(move |ctx| Value::None)
    }
}
