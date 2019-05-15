//! An attempt to implement dynamic dispatch codegen
//!
//! See [Building fast intepreters in Rust](https://blog.cloudflare.com/building-fast-interpreters-in-rust/)

use crate::evaluator::EvaluationContext;
use crate::storage::Value;
use streamlab_frontend::ir::{Constant, Expression, Offset, Type};

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
                use streamlab_frontend::ir::ArithLogOp::*;
                match op {
                    Not => CompiledExpr::new(move |ctx| {
                        let b = f_operands[0].execute(ctx);
                        !b
                    }),
                    Neg => CompiledExpr::new(move |ctx| {
                        let v = f_operands[0].execute(ctx);
                        -v
                    }),
                    Add => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        lhs + rhs
                    }),
                    Sub => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        lhs - rhs
                    }),
                    Mul => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        lhs * rhs
                    }),
                    Div => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        lhs / rhs
                    }),
                    Rem => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        lhs % rhs
                    }),
                    Pow => CompiledExpr::new(move |ctx| {
                        let base = f_operands[0].execute(ctx);
                        let exp = f_operands[1].execute(ctx);
                        base.pow(exp)
                    }),
                    And => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx).get_bool();
                        if !lhs {
                            Value::Bool(false)
                        } else {
                            let res = f_operands[1].execute(ctx);
                            assert!(res.is_bool());
                            res
                        }
                    }),
                    Or => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx).get_bool();
                        if lhs {
                            Value::Bool(true)
                        } else {
                            let res = f_operands[1].execute(ctx);
                            assert!(res.is_bool());
                            res
                        }
                    }),
                    Eq => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        Value::Bool(lhs == rhs)
                    }),
                    Lt => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        Value::Bool(lhs < rhs)
                    }),
                    Le => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        Value::Bool(lhs <= rhs)
                    }),
                    Ne => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        Value::Bool(lhs != rhs)
                    }),
                    Ge => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        Value::Bool(lhs >= rhs)
                    }),
                    Gt => CompiledExpr::new(move |ctx| {
                        let lhs = f_operands[0].execute(ctx);
                        let rhs = f_operands[1].execute(ctx);
                        Value::Bool(lhs > rhs)
                    }),
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

            StreamAccess(str_ref, kind) => CompiledExpr::new(move |ctx| ctx.lookup(str_ref)), // TODO: @Marvin

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
                match name.as_ref() {
                    "sqrt" => CompiledExpr::new(move |ctx| {
                        let arg = f_arg.execute(ctx);
                        match arg {
                            Value::Float(f) => Value::new_float(f.sqrt()),
                            _ => panic!(),
                        }
                    }),
                    "sin" => CompiledExpr::new(move |ctx| {
                        let arg = f_arg.execute(ctx);
                        match arg {
                            Value::Float(f) => Value::new_float(f.sin()),
                            _ => panic!(),
                        }
                    }),
                    "cos" => CompiledExpr::new(move |ctx| {
                        let arg = f_arg.execute(ctx);
                        match arg {
                            Value::Float(f) => Value::new_float(f.cos()),
                            _ => panic!(),
                        }
                    }),
                    "arctan" => CompiledExpr::new(move |ctx| {
                        let arg = f_arg.execute(ctx);
                        match arg {
                            Value::Float(f) => Value::new_float(f.atan()),
                            _ => panic!(),
                        }
                    }),
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
                use Type::*;
                let f_expr = expr.compile();
                match (from, to) {
                    (UInt(_), UInt(_)) => CompiledExpr::new(move |ctx| {
                        let v = f_expr.execute(ctx);
                        match v {
                            Value::Unsigned(u) => Value::Unsigned(u as u128),
                            _ => panic!(),
                        }
                    }),
                    (UInt(_), Int(_)) => CompiledExpr::new(move |ctx| {
                        let v = f_expr.execute(ctx);
                        match v {
                            Value::Unsigned(u) => Value::Signed(u as i128),
                            _ => panic!(),
                        }
                    }),
                    (UInt(_), Float(_)) => CompiledExpr::new(move |ctx| {
                        let v = f_expr.execute(ctx);
                        match v {
                            Value::Unsigned(u) => Value::new_float(u as f64),
                            _ => panic!(),
                        }
                    }),
                    (Int(_), UInt(_)) => CompiledExpr::new(move |ctx| {
                        let v = f_expr.execute(ctx);
                        match v {
                            Value::Signed(i) => Value::Unsigned(i as u128),
                            _ => panic!(),
                        }
                    }),
                    (Int(_), Int(_)) => CompiledExpr::new(move |ctx| {
                        let v = f_expr.execute(ctx);
                        match v {
                            Value::Signed(i) => Value::Signed(i as i128),
                            _ => panic!(),
                        }
                    }),
                    (Int(_), Float(_)) => CompiledExpr::new(move |ctx| {
                        let v = f_expr.execute(ctx);
                        match v {
                            Value::Signed(i) => Value::new_float(i as f64),
                            _ => panic!(),
                        }
                    }),
                    (Float(_), UInt(_)) => CompiledExpr::new(move |ctx| {
                        let v = f_expr.execute(ctx);
                        match v {
                            Value::Float(f) => Value::Unsigned(f.into_inner() as u128),
                            _ => panic!(),
                        }
                    }),
                    (Float(_), Int(_)) => CompiledExpr::new(move |ctx| {
                        let v = f_expr.execute(ctx);
                        match v {
                            Value::Float(f) => Value::Signed(f.into_inner() as i128),
                            _ => panic!(),
                        }
                    }),
                    (Float(_), Float(_)) => CompiledExpr::new(move |ctx| {
                        let v = f_expr.execute(ctx);
                        match v {
                            Value::Float(f) => Value::new_float(f.into_inner() as f64),
                            _ => panic!(),
                        }
                    }),
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
