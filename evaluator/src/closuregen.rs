//! An attempt to implement dynamic dispatch codegen
//!
//! See [Building fast intepreters in Rust](https://blog.cloudflare.com/building-fast-interpreters-in-rust/)

use crate::evaluator::EvaluationContext;
use crate::storage::Value;
use std::ops::{Add, Div, Mul, Neg, Not, Rem, Sub};
use streamlab_frontend::ir::{Constant, Expression, Offset, StreamAccessKind, StreamReference, Type};

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
        use Expression::*;
        match self {
            LoadConstant(c) => {
                let v = match c {
                    Constant::Bool(b) => Value::Bool(b),
                    Constant::UInt(u) => Value::Unsigned(u),
                    Constant::Int(i) => Value::Signed(i),
                    Constant::Float(f) => Value::Float(f.into()),
                    Constant::Str(s) => Value::Str(s),
                };
                CompiledExpr::new(move |_| v.clone())
            }

            ArithLog(op, operands, _ty) => {
                let f_operands: Vec<CompiledExpr> = operands.into_iter().map(|e| e.compile()).collect();

                macro_rules! create_unop {
                    ($fn:ident) => {
                        CompiledExpr::new(move |ctx| {
                            let lhs = f_operands[0].execute(ctx);
                            lhs.$fn()
                        })
                    };
                }
                macro_rules! create_binop {
                    ($fn:ident) => {
                        CompiledExpr::new(move |ctx| {
                            let lhs = f_operands[0].execute(ctx);
                            let rhs = f_operands[1].execute(ctx);
                            lhs.$fn(rhs)
                        })
                    };
                }
                macro_rules! create_cmp {
                    ($fn:ident) => {
                        CompiledExpr::new(move |ctx| {
                            let lhs = f_operands[0].execute(ctx);
                            let rhs = f_operands[1].execute(ctx);
                            Value::Bool(lhs.$fn(&rhs))
                        })
                    };
                }
                macro_rules! create_lazyop {
                    ($b:expr) => {
                        CompiledExpr::new(move |ctx| {
                            let lhs = f_operands[0].execute(ctx).get_bool();
                            if lhs == $b {
                                Value::Bool($b)
                            } else {
                                let res = f_operands[1].execute(ctx);
                                assert!(res.is_bool());
                                res
                            }
                        })
                    };
                }

                use streamlab_frontend::ir::ArithLogOp::*;
                match op {
                    Not => create_unop!(not),
                    Neg => create_unop!(neg),
                    Add => create_binop!(add),
                    Sub => create_binop!(sub),
                    Mul => create_binop!(mul),
                    Div => create_binop!(div),
                    Rem => create_binop!(rem),
                    Pow => create_binop!(pow),
                    Eq => create_cmp!(eq),
                    Lt => create_cmp!(lt),
                    Le => create_cmp!(le),
                    Ne => create_cmp!(ne),
                    Ge => create_cmp!(ge),
                    Gt => create_cmp!(gt),
                    And => create_lazyop!(false),
                    Or => create_lazyop!(true),
                }
            }

            OffsetLookup { target, offset } => {
                let offset = match offset {
                    Offset::FutureDiscreteOffset(_) | Offset::FutureRealTimeOffset(_) => unimplemented!(),
                    Offset::PastDiscreteOffset(u) => -(u as i16),
                    Offset::PastRealTimeOffset(_dur) => unimplemented!(),
                };
                CompiledExpr::new(move |ctx| ctx.lookup_with_offset(target, offset))
            }

            StreamAccess(str_ref, kind) => {
                use StreamAccessKind::*;
                match kind {
                    Hold => CompiledExpr::new(move |ctx| ctx.lookup(str_ref)),
                    Optional => {
                        use StreamReference::*;
                        match str_ref {
                            InRef(ix) => CompiledExpr::new(move |ctx| {
                                if ctx.fresh_inputs.contains(ix) {
                                    ctx.lookup(str_ref)
                                } else {
                                    Value::None
                                }
                            }),
                            OutRef(ix) => CompiledExpr::new(move |ctx| {
                                if ctx.fresh_outputs.contains(ix) {
                                    ctx.lookup(str_ref)
                                } else {
                                    Value::None
                                }
                            }),
                        }
                    }
                }
            }

            SyncStreamLookup(str_ref) => CompiledExpr::new(move |ctx| ctx.lookup(str_ref)),

            WindowLookup(win_ref) => CompiledExpr::new(move |ctx| ctx.lookup_window(win_ref)),

            Ite { condition, consequence, alternative } => {
                let f_condition = condition.compile();
                let f_consequence = consequence.compile();
                let f_alternative = alternative.compile();

                CompiledExpr::new(move |ctx| {
                    let cond = f_condition.execute(ctx).get_bool();
                    if cond {
                        f_consequence.execute(ctx)
                    } else {
                        f_alternative.execute(ctx)
                    }
                })
            }

            /*
            Tuple(entries) => {
                let f_entries: Vec<CompiledExpr> = entries.into_iter().map(|e| e.compile()).collect();

                CompiledExpr::new(move |ctx| {
                    let entries: Vec<Value> = f_entries.iter().map(|f| f.execute(ctx)).collect();
                    Value::Tuple(entries)
                })
            }
            */
            Function(name, args, _ty) => {
                //TODO(marvin): handle type
                let f_arg = args[0].clone().compile();

                macro_rules! create_floatfn {
                    ($fn:ident) => {
                        CompiledExpr::new(move |ctx| {
                            let arg = f_arg.execute(ctx);
                            match arg {
                                Value::Float(f) => Value::new_float(f.$fn()),
                                _ => panic!(),
                            }
                        })
                    };
                }

                match name.as_ref() {
                    "sqrt" => create_floatfn!(sqrt),
                    "sin" => create_floatfn!(sin),
                    "cos" => create_floatfn!(cos),
                    "arctan" => create_floatfn!(atan),
                    "abs" => CompiledExpr::new(move |ctx| {
                        let arg = f_arg.execute(ctx);
                        match arg {
                            Value::Float(f) => Value::new_float(f.abs()),
                            Value::Signed(i) => Value::Signed(i.abs()),
                            _ => panic!(),
                        }
                    }),
                    _ => panic!("Unknown function."),
                }
            }

            //Expression::Convert { from, to, expr } => CompiledExpr::new(move |ctx| {}),
            Convert { from, to, expr } => {
                let f_expr = expr.compile();

                macro_rules! create_convert {
                    (Float, $to:ident, $ty:ty) => {
                        CompiledExpr::new(move |ctx| {
                            let v = f_expr.execute(ctx);
                            match v {
                                Value::Float(f) => Value::$to(f.into_inner() as $ty),
                                _ => panic!(),
                            }
                        })
                    };
                    ($from:ident, Float, $ty:ty) => {
                        CompiledExpr::new(move |ctx| {
                            let v = f_expr.execute(ctx);
                            match v {
                                Value::$from(v) => Value::new_float(v as $ty),
                                _ => panic!(),
                            }
                        })
                    };
                    ($from:ident, $to:ident, $ty:ty) => {
                        CompiledExpr::new(move |ctx| {
                            let v = f_expr.execute(ctx);
                            match v {
                                Value::$from(v) => Value::$to(v as $ty),
                                _ => panic!(),
                            }
                        })
                    };
                }

                use Type::*;
                match (from, to) {
                    (UInt(_), UInt(_)) => CompiledExpr::new(move |ctx| f_expr.execute(ctx)),
                    (UInt(_), Int(_)) => create_convert!(Unsigned, Signed, i64),
                    (UInt(_), Float(_)) => create_convert!(Unsigned, Float, f64),
                    (Int(_), UInt(_)) => create_convert!(Signed, Unsigned, u64),
                    (Int(_), Int(_)) => CompiledExpr::new(move |ctx| f_expr.execute(ctx)),
                    (Int(_), Float(_)) => create_convert!(Signed, Float, f64),
                    (Float(_), UInt(_)) => create_convert!(Float, Unsigned, u64),
                    (Float(_), Int(_)) => create_convert!(Float, Signed, i64),
                    (Float(_), Float(_)) => CompiledExpr::new(move |ctx| f_expr.execute(ctx)),
                    _ => unimplemented!(),
                }
            }

            Default { expr, default } => {
                let f_expr = expr.compile();
                let f_default = default.compile();
                CompiledExpr::new(move |ctx| {
                    let v = f_expr.execute(ctx);
                    if let Value::None = v {
                        f_default.execute(ctx)
                    } else {
                        v
                    }
                })
            }
            _ => unimplemented!("not implemented yet"),
        }
    }
}