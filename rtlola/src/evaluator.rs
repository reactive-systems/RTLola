use lola_parser::*;

use crate::basics::{EvalConfig, OutputHandler};
use crate::storage::{GlobalStore, TempStore, Value};

pub(crate) type OutInstance = (usize, Vec<Value>);
pub(crate) type Window = (usize, Vec<Value>);
use ordered_float::NotNan;

use std::time::Instant;

pub(crate) struct Evaluator {
    // Indexed by stream reference.
    temp_stores: Vec<TempStore>,
    // Indexed by stream reference.
    exprs: Vec<Expression>,
    global_store: GlobalStore,
    ir: LolaIR,
    handler: OutputHandler,
}

impl Evaluator {
    pub(crate) fn new(ir: LolaIR, ts: Instant, config: EvalConfig) -> Evaluator {
        let temp_stores = ir.outputs.iter().map(|o| TempStore::new(&o.expr)).collect();
        let global_store = GlobalStore::new(&ir, ts);
        let exprs = ir.outputs.iter().map(|o| o.expr.clone()).collect();
        let handler = OutputHandler::new(&config);
        Evaluator { temp_stores, exprs, global_store, ir, handler }
    }

    pub(crate) fn eval_stream(&mut self, inst: OutInstance, ts: Option<Instant>) {
        let (ix, _) = inst;
        let ts = ts.unwrap_or_else(Instant::now);
        for stmt in self.exprs[ix].stmts.clone() {
            self.eval_stmt(&stmt, &inst, ts);
        }
        let res = self.get(self.exprs[ix].stmts.last().unwrap().target, &inst);

        // Register value in global store.
        self.global_store.get_out_instance_mut(inst.clone()).unwrap().push_value(res.clone()); // TODO: unsafe unwrap.

        self.handler.output(|| format!("OutputStream[{}] := {:?}.", inst.0, res.clone()));

        // Check linked streams and inform them.
        let extended = self.ir.get_out(StreamReference::OutRef(ix));
        for win in &extended.dependent_windows {
            self.global_store.get_window_mut((win.ix, Vec::new())).accept_value(res.clone(), ts)
        }
        // TODO: Dependent streams?
    }

    pub(crate) fn accept_input(&mut self, input: StreamReference, v: Value, ts: Option<Instant>) {
        self.global_store.get_in_instance_mut(input).push_value(v.clone());
        self.handler.debug(|| format!("InputStream[{}] := {:?}.", input.in_ix(), v.clone()));
        let extended = self.ir.get_in(input);
        for win in &extended.dependent_windows {
            self.global_store
                .get_window_mut((win.ix, Vec::new()))
                .accept_value(v.clone(), ts.unwrap_or_else(Instant::now))
        }
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

    fn eval_stmt(&mut self, stmt: &Statement, inst: &OutInstance, ts: Instant) {
        match &stmt.op {
            Op::Convert | Op::Move => {
                let arg = stmt.args[0];
                let v = self.get(arg, inst);
                self.write(stmt.target, v, inst);
            }
            Op::Ite { consequence, alternative } => {
                let cont_with = if self.get_bool(stmt.args[0], inst) { consequence } else { alternative };
                for stmt in cont_with {
                    self.eval_stmt(stmt, inst, ts);
                }
            }
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
                self.write(stmt.target, res.unwrap(), inst); // Unwrap sound in sync lookups.
            }
            Op::StreamLookup { instance: tar_inst, offset }
            | Op::SampleAndHoldStreamLookup { instance: tar_inst, offset } => {
                let res = match offset {
                    Offset::FutureDiscreteOffset(_) | Offset::FutureRealTimeOffset(_) => unimplemented!(),
                    Offset::PastDiscreteOffset(u) => self.perform_lookup(inst, tar_inst, -(*u as i16)),
                    Offset::PastRealTimeOffset(_dur) => unimplemented!(),
                };
                let v = res.unwrap_or_else(|| self.get(stmt.args[0], inst));
                self.write(stmt.target, v, inst);
            }
            Op::WindowLookup(window_ref) => {
                let window: Window = (window_ref.ix, Vec::new() /*, self.window_ops[window_ref.ix]*/);
                let res = self.global_store.get_window_mut(window).get_value(ts);
                self.write(stmt.target, res, inst);
            }
            Op::Function(name) => {
                let arg = self.get(stmt.args[0], inst);
                if let Value::Float(f) = arg {
                    let res = match name.as_ref() {
                        "sqrt" => f.sqrt(),
                        "sin" => f.sin(),
                        "cos" => f.cos(),
                        _ => panic!("Unknown function!"),
                    };
                    let res = Value::Float(NotNan::new(res).expect("TODO: Handle"));
                    self.write(stmt.target, res, inst);
                } else {
                    panic!();
                }
            }
            Op::Tuple => unimplemented!("Who needs tuples, anyway?"),
        }
    }

    fn perform_lookup(&self, inst: &OutInstance, tar_inst: &StreamInstance, offset: i16) -> Option<Value> {
        let is = match tar_inst.reference {
            StreamReference::InRef(_) => Some(self.global_store.get_in_instance(tar_inst.reference)),
            StreamReference::OutRef(i) => {
                let args = tar_inst.arguments.iter().map(|a| self.get(*a, inst)).collect();
                let target: OutInstance = (i, args);
                self.global_store.get_out_instance(target)
            }
        };
        is.and_then(|is| is.get_value(offset))
    }

    fn __peek_value(&self, sr: StreamReference, args: &[Value], offset: i16) -> Option<Value> {
        match sr {
            StreamReference::InRef(_) => {
                assert!(args.is_empty());
                self.global_store.get_in_instance(sr).get_value(offset)
            }
            StreamReference::OutRef(ix) => {
                let inst = (ix, Vec::from(args));
                self.global_store.get_out_instance(inst).and_then(|st| st.get_value(offset))
            }
        }
    }
}

mod tests {

    use super::*;
    use lola_parser::LolaIR;
    use std::time::{Duration, Instant};

    #[allow(dead_code)]
    fn setup(spec: &str) -> (LolaIR, Evaluator) {
        setup_time(spec, Instant::now())
    }

    #[allow(dead_code)]
    fn setup_time(spec: &str, ts: Instant) -> (LolaIR, Evaluator) {
        let ir = lola_parser::parse(spec);
        let eval = Evaluator::new(ir.clone(), ts, EvalConfig::default());
        (ir, eval)
    }

    #[test]
    fn test_empty_outputs() {
        setup("input a: UInt8");
    }

    #[test]
    fn test_const_output() {
        let (_, mut eval) = setup("output a: UInt8 := 3");
        let inst = (0, Vec::new());
        eval.eval_stream(inst.clone(), None);
        assert_eq!(eval.__peek_value(StreamReference::OutRef(0), &Vec::new(), 0).unwrap(), Value::Unsigned(3))
    }

    #[test]
    fn test_const_output_arith() {
        let (_, mut eval) = setup("output a: UInt8 := 3 + 5");
        let inst = (0, Vec::new());
        eval.eval_stream(inst.clone(), None);
        assert_eq!(eval.__peek_value(StreamReference::OutRef(0), &Vec::new(), 0).unwrap(), Value::Unsigned(8))
    }

    #[test]
    fn test_input_only() {
        let (_, mut eval) = setup("input a: UInt8");
        let sr = StreamReference::InRef(0);
        let v = Value::Unsigned(3);
        eval.accept_input(sr, v.clone(), None);
        assert_eq!(eval.__peek_value(sr, &Vec::new(), 0).unwrap(), v)
    }

    #[test]
    fn test_sync_lookup() {
        let (_, mut eval) = setup("input a: UInt8 output b: UInt8 := a");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v = Value::Unsigned(9);
        eval.accept_input(in_ref, v.clone(), None);
        eval.eval_stream((0, Vec::new()), None);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v)
    }

    #[test]
    fn test_oob_lookup() {
        let (_, mut eval) = setup("input a: UInt8\noutput b: UInt8 { extend @5Hz }:= a[-1] ! 3");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        eval.accept_input(in_ref, v1, None);
        eval.eval_stream((0, Vec::new()), None);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
    }

    #[test]
    fn test_output_lookup() {
        let (_, mut eval) =
            setup("input a: UInt8\noutput mirror: UInt8 := a\noutput c: UInt8 { extend @5Hz }:= mirror[-1] ! 3");
        let out_ref = StreamReference::OutRef(1);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        eval.accept_input(in_ref, v1.clone(), None);
        eval.eval_stream((0, Vec::new()), None);
        eval.accept_input(in_ref, v2, None);
        eval.eval_stream((0, Vec::new()), None);
        eval.eval_stream((1, Vec::new()), None);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
    }

    #[test]
    fn test_conversion_if() {
        let (_, mut eval) = setup("input a: UInt8\noutput b: UInt16 := if true then a else a[-1] ? 0");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        eval.accept_input(in_ref, v1.clone(), None);
        eval.eval_stream((0, Vec::new()), None);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
    }

    #[test]
    fn test_conversion_lookup() {
        let (ir, mut eval) = setup("input a: UInt8\noutput b: UInt32 := a + 100000");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let expected = Value::Unsigned(7 + 100000);
        let v1 = Value::Unsigned(7);
        eval.accept_input(in_ref, v1, None);
        eval.eval_stream((0, Vec::new()), None);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_bin_op() {
        let (_, mut eval) = setup("input a: UInt16\n input b: UInt16\noutput c: UInt16 := a + b");
        let out_ref = StreamReference::OutRef(0);
        let a = StreamReference::InRef(0);
        let b = StreamReference::InRef(1);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        eval.accept_input(a, v1.clone(), None);
        eval.accept_input(b, v2.clone(), None);
        eval.eval_stream((0, Vec::new()), None);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1 + v2);
    }

    #[test]
    fn test_regular_lookup() {
        let (_, mut eval) = setup("input a: UInt8 output b: UInt8 { extend @5Hz }:= a[-1] ! 3");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        let v3 = Value::Unsigned(3);
        eval.accept_input(in_ref, v1, None);
        eval.accept_input(in_ref, v2.clone(), None);
        eval.accept_input(in_ref, v3, None);
        eval.eval_stream((0, Vec::new()), None);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v2)
    }

    #[test]
    fn test_sum_window() {
        let mut time = Instant::now();
        let (_, mut eval) = setup_time("input a: Int16\noutput b: Int16 { extend @0.25Hz } := a[40s, sum] ? -3", time);
        time += Duration::from_secs(45);
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let n = 25;
        for v in 1..=n {
            eval.accept_input(in_ref, Value::Signed(v), Some(time));
            time += Duration::from_secs(1);
        }
        time += Duration::from_secs(1);
        // 71 secs have passed. All values should be within the window.
        eval.eval_stream((0, Vec::new()), Some(time));
        let expected = Value::Signed((n * n + n) / 2);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }
}