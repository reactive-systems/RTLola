//! An attempt to implement dynamic dispatch codegen
//!
//! See [Building fast interpreters in Rust](https://blog.cloudflare.com/building-fast-interpreters-in-rust/)

use crate::evaluator::EvaluationContext;
use crate::storage::Value;
use regex::bytes::Regex as BytesRegex;
use regex::Regex;
use rtlola_frontend::ir::{Constant, Expression, ExpressionKind, Offset, StreamAccessKind, StreamReference, Type};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

pub(crate) trait Expr {
    fn compile(self) -> CompiledExpr;
}
pub(crate) struct CompiledExpr(Box<dyn Fn(&EvaluationContext<'_>) -> Value>);
// alternative: using Higher-Rank Trait Bounds (HRTBs)
// pub(crate) struct CompiledExpr<'s>(Box<dyn 's + for<'a> Fn(&EvaluationContext<'a>) -> Value>);

impl CompiledExpr {
    /// Creates a compiled expression IR from a generic closure.
    pub(crate) fn new(closure: impl 'static + Fn(&EvaluationContext<'_>) -> Value) -> Self {
        CompiledExpr(Box::new(closure))
    }

    /// Executes a filter against a provided context with values.
    pub(crate) fn execute(&self, ctx: &EvaluationContext) -> Value {
        self.0(ctx)
    }
}

impl Expr for Expression {
    fn compile(self) -> CompiledExpr {
        use ExpressionKind::*;
        match self.kind {
            LoadConstant(c) => {
                let v = match c {
                    Constant::Bool(b) => Value::Bool(b),
                    Constant::UInt(u) => Value::Unsigned(u),
                    Constant::Int(i) => Value::Signed(i),
                    Constant::Float(f) => Value::Float(f.into()),
                    Constant::Str(s) => Value::Str(s.into_boxed_str()),
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

                use rtlola_frontend::ir::ArithLogOp::*;
                match op {
                    Not => create_unop!(not),
                    BitNot => create_unop!(not),
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
                    BitAnd => create_binop!(bitand),
                    BitOr => create_binop!(bitor),
                    BitXor => create_binop!(bitxor),
                    Shl => create_binop!(shl),
                    Shr => create_binop!(shr),
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
                    Sync => CompiledExpr::new(move |ctx| ctx.lookup_latest_check(str_ref)),
                    Hold => CompiledExpr::new(move |ctx| ctx.lookup_latest(str_ref)),
                    Optional => {
                        use StreamReference::*;
                        match str_ref {
                            InRef(ix) => CompiledExpr::new(move |ctx| {
                                if ctx.fresh_inputs.contains(ix) {
                                    ctx.lookup_latest(str_ref)
                                } else {
                                    Value::None
                                }
                            }),
                            OutRef(ix) => CompiledExpr::new(move |ctx| {
                                if ctx.fresh_outputs.contains(ix) {
                                    ctx.lookup_latest(str_ref)
                                } else {
                                    Value::None
                                }
                            }),
                        }
                    }
                }
            }

            WindowLookup(win_ref) => CompiledExpr::new(move |ctx| ctx.lookup_window(win_ref)),

            Ite { condition, consequence, alternative, .. } => {
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

            Tuple(entries) => {
                let f_entries: Vec<CompiledExpr> = entries.into_iter().map(|e| e.compile()).collect();
                CompiledExpr::new(move |ctx| Value::Tuple(f_entries.iter().map(|f| f.execute(ctx)).collect()))
            }

            Function(name, args, ty) => {
                //TODO(marvin): handle type
                assert!(!args.is_empty());
                let f_arg = args[0].clone().compile();

                macro_rules! create_floatfn {
                    ($fn:ident) => {
                        CompiledExpr::new(move |ctx| {
                            let arg = f_arg.execute(ctx);
                            match arg {
                                Value::Float(f) => Value::new_float(f.$fn()),
                                _ => unreachable!(),
                            }
                        })
                    };
                }

                macro_rules! create_binary_arith {
                    ($fn:ident) => {{
                        if args.len() != 2 {
                            unreachable!("wrong number of arguments for function $fn")
                        }
                        CompiledExpr::new(move |ctx| {
                            let fst = f_arg.execute(ctx);
                            let snd = args[1].clone().compile().execute(ctx);
                            match (fst, snd) {
                                (Value::Float(f1), Value::Float(f2)) => Value::Float(f1.$fn(f2)),
                                (Value::Signed(s1), Value::Signed(s2)) => Value::Signed(s1.$fn(s2)),
                                (Value::Unsigned(u1), Value::Unsigned(u2)) => Value::Unsigned(u1.$fn(u2)),
                                (v1, v2) => unreachable!("wrong Value types of {:?}, {:?} for function $fn", v1, v2),
                            }
                        })
                    }};
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
                            v => unreachable!("wrong Value type of {:?}, for function abs", v),
                        }
                    }),
                    "min" => create_binary_arith!(min),
                    "max" => create_binary_arith!(max),
                    "matches" => {
                        assert!(args.len() >= 2);
                        let operand_ty = match &ty {
                            Type::Function(args, _ret) => &args[0],
                            _ => unreachable!(),
                        };
                        let is_bytes = operand_ty == &Type::Bytes;
                        let re_str = match &args[1].kind {
                            LoadConstant(Constant::Str(s)) => s,
                            _ => unreachable!("regex should be a string literal"),
                        };
                        if !is_bytes {
                            let re = Regex::new(&re_str).expect("Given regular expression was invalid");
                            CompiledExpr::new(move |ctx| {
                                let val = f_arg.execute(ctx);
                                if let Value::Str(s) = &val {
                                    Value::Bool(re.is_match(s))
                                } else {
                                    unreachable!("expected `String`, found {:?}", val);
                                }
                            })
                        } else {
                            let re = BytesRegex::new(&re_str).expect("Given regular expression was invalid");
                            CompiledExpr::new(move |ctx| {
                                let val = f_arg.execute(ctx);
                                if let Value::Bytes(b) = &val {
                                    Value::Bool(re.is_match(b))
                                } else {
                                    unreachable!("expected `Bytes`, found {:?}", val);
                                }
                            })
                        }
                    }
                    "at" => {
                        assert_eq!(args.len(), 2);
                        let index_arg = args[1].clone().compile();
                        CompiledExpr::new(move |ctx| {
                            let val = f_arg.execute(ctx);
                            let index = index_arg.execute(ctx);
                            match (val, index) {
                                (Value::Bytes(b), Value::Unsigned(idx)) => {
                                    if let Some(&byte) = b.get(idx as usize) {
                                        Value::Unsigned(byte.into())
                                    } else {
                                        Value::None
                                    }
                                }
                                (val, _) => unreachable!("expected `Bytes`, found {:?}", val),
                            }
                        })
                    }
                    f => unreachable!("Unknown function: {}, args: {:?}", f, args),
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
                                v => unreachable!(
                                    "Value type of {:?} does not match convert from type {:?}",
                                    v,
                                    Value::new_float(0.0)
                                ),
                            }
                        })
                    };
                    ($from:ident, Float, $ty:ty) => {
                        CompiledExpr::new(move |ctx| {
                            let v = f_expr.execute(ctx);
                            match v {
                                Value::$from(v) => Value::new_float(v as $ty),
                                v => unreachable!(
                                    "Value type of {:?} does not match convert from type {:?}",
                                    v,
                                    Value::$from(0)
                                ),
                            }
                        })
                    };
                    ($from:ident, $to:ident, $ty:ty) => {
                        CompiledExpr::new(move |ctx| {
                            let v = f_expr.execute(ctx);
                            match v {
                                Value::$from(v) => Value::$to(v as $ty),
                                v => unreachable!(
                                    "Value type of {:?} does not match convert from type {:?}",
                                    v,
                                    Value::$from(0)
                                ),
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
                    (from, to) => unreachable!("from: {:?}, to: {:?}", from, to),
                }
            }

            Default { expr, default, .. } => {
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

            TupleAccess(expr, num) => {
                let f_expr = expr.compile();
                CompiledExpr::new(move |ctx| {
                    if let Value::Tuple(args) = f_expr.execute(ctx) {
                        args[num].clone()
                    } else {
                        unreachable!("verified by type checker");
                    }
                })
            }
        }
    }
}
