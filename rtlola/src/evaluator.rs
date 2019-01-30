
use lola_parser::*;

use crate::storage::{Value, TempStore, GlobalStore};

pub(crate) type OutInstance = (usize, Vec<Value>);
pub(crate) type Window = (usize, Vec<Value>);
use ordered_float::NotNaN;

pub(crate) struct Evaluator {
    // Indexed by stream reference.
    temp_stores: Vec<TempStore>,
    // Indexed by stream reference.
    exprs: Vec<Expression>,
    global_store: GlobalStore,
}

impl Evaluator {
    pub(crate) fn new(ir: &LolaIR) -> Evaluator {
        let temp_stores = ir.outputs.iter().map(|o| TempStore::new(&o.expr)).collect();
        let global_store = GlobalStore::new(&ir);
        let exprs = ir.outputs.iter().map(|o| o.expr.clone()).collect();
        Evaluator { temp_stores, exprs, global_store }
    }

    pub(crate) fn eval_stream(&mut self, expr: &Expression, inst: OutInstance) {
        for stmt in &self.exprs[inst.0].stmts.clone() {
            self.eval_stmt(stmt, &inst);
        }
        // TODO: Put value in global store.
    }

    fn type_of(&self, temp: Temporary, inst: &OutInstance) -> &Type {
        &self.exprs[inst.0].temporaries[temp.0]
    }

    fn get_signed(&self, temp: Temporary, inst: &OutInstance) -> i128 {
        self.temp_stores[inst.0].get_signed(temp)
    }

    fn get_unsigned(&self, temp: Temporary, inst: &OutInstance) -> u128 {
        self.temp_stores[inst.0].get_unsigned(temp)
    }

    fn get_bool(&self, temp: Temporary, inst: &OutInstance) -> bool {
        self.temp_stores[inst.0].get_bool(temp)
    }

    fn get(&self, temp: Temporary, inst: &OutInstance) -> Value {
        self.temp_stores[inst.0].get_value(temp)
    }

    fn write(&mut self, temp: Temporary, value: Value, inst: &OutInstance) {
        self.temp_stores[inst.0].write_value(temp, value);
    }

    fn eval_stmt(&mut self, stmt: &Statement, inst: &OutInstance) {
        match &stmt.op {
            Op::Convert | Op::Move => {
                let arg = stmt.args[0];
                let v = self.get(arg, inst);
                self.write(stmt.target, v, inst);
            }
            Op::Ite { consequence, alternative } => {
                let cont_with = if self.get_bool(stmt.args[0], inst) {
                    consequence
                } else {
                    alternative
                };
                for stmt in cont_with {
                    self.eval_stmt(stmt, inst);
                }
            },
            Op::LoadConstant(c) => {
                let val = match c {
                    Constant::Bool(b) => Value::Bool(*b),
                    Constant::Int(i) if *i >= 0 => Value::Unsigned(*i as u128),
                    Constant::Int(i) => Value::Signed(*i),
                    Constant::Float(_) | Constant::Str(_) => unimplemented!(),
                };
                self.write(stmt.target, val, inst);
            }
            Op::ArithLog(op) => {
                use lola_parser::ArithLogOp::*;
                // The explicit match here enables a compiler warning when a case was missed.
                // Useful when the list in the parser is extended.
                let arity = match op {
                    Neg | Not => 1,
                    Add | Sub | Mul | Div | Rem | Pow | And | Or | Eq | Lt | Le | Ne | Ge | Gt => 2,
                };
                if arity == 1 {
                    let operand = self.get(stmt.args[0], inst);
                    self.write(stmt.target, !operand, inst)
                } else if arity == 2 {
                    let lhs = self.get(stmt.args[0], inst);
                    let rhs = self.get(stmt.args[1], inst);

                    let res = match op {
                        Add => lhs + rhs,
                        Sub => lhs - rhs,
                        Mul => lhs * rhs,
                        Div => lhs / rhs,
                        Rem => lhs % rhs,
                        Pow => unimplemented!(),
                        And => lhs & rhs,
                        Or => lhs | rhs,
                        Eq => Value::Bool(lhs == rhs),
                        Lt => Value::Bool(lhs <= rhs),
                        Le => Value::Bool(lhs < rhs),
                        Ne => Value::Bool(lhs != rhs),
                        Ge => Value::Bool(lhs >= rhs),
                        Gt => Value::Bool(lhs > rhs),
                        Neg | Not => panic!(),
                    };
                    self.write(stmt.target, res, inst);
                }
            }
            Op::SyncStreamLookup(tar_inst) => {
                let res = self.perform_lookup(inst, tar_inst, 0);
                self.write(stmt.target, res, inst);
            },
            Op::StreamLookup { instance: tar_inst, offset } => {
                let res = match offset {
                    Offset::FutureDiscreteOffset(_) | Offset::FutureRealTimeOffset(_) => unimplemented!(),
                    Offset::PastDiscreteOffset(u) => self.perform_lookup(inst, tar_inst, *u as i16),
                    Offset::PastRealTimeOffset(dur) => unimplemented!(),
                };
                self.write(stmt.target, res, inst);
            },
            Op::WindowLookup(window_ref) => {
                let window: Window = (window_ref.ix, Vec::new());
                let ws = self.global_store.get_window(window);
                let res = ws.get_value();
                self.write(stmt.target, res, inst);
            }
            Op::Function(name) => {
                let arg = self.get(stmt.args[0], inst);
                if let Value::Float(f) = arg {
                    let res = match name.as_ref() {
                        "sqrt" => f.sqrt(),
                        "sin" => f.sin(),
                        "cos" => f.cos(),
                        _ => panic!("Unknown function!")
                    };
                    let res = Value::Float(NotNaN::new(res).expect(unimplemented!()));
                    self.write(stmt.target, res, inst);
                } else {
                    panic!();
                }
            }
            Op::Tuple => unimplemented!("Who needs tuples, anyway?"),
        }
    }

    fn perform_lookup(&self, inst: &OutInstance, tar_inst: &StreamInstance, offset: i16) -> Value {
        let is = match tar_inst.reference {
            StreamReference::InRef(_) => self.global_store.get_in_instance(tar_inst.reference),
            StreamReference::OutRef(i) => {
                let args = tar_inst.arguments.iter().map(|a| self.get(*a, inst)).collect();
                let target: OutInstance = (i, args);
                self.global_store.get_out_instance(target)
            }
        };
        is.get_value(offset)
    }
}