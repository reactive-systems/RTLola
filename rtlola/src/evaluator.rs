use lola_parser::ir::*;

use crate::basics::{EvalConfig, OutputHandler};
use crate::storage::{GlobalStore, TempStore, Value};

pub(crate) type OutInstance = (usize, Vec<Value>);
pub(crate) type Window = (usize, Vec<Value>);
use ordered_float::NotNan;

use std::time::SystemTime;

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
    pub(crate) fn new(ir: LolaIR, ts: SystemTime, config: EvalConfig) -> Evaluator {
        let temp_stores = ir.outputs.iter().map(|o| TempStore::new(&o.expr)).collect();
        let global_store = GlobalStore::new(&ir, ts);
        let exprs = ir.outputs.iter().map(|o| o.expr.clone()).collect();
        let handler = OutputHandler::new(&config);
        Evaluator { temp_stores, exprs, global_store, ir, handler }
    }

    pub(crate) fn eval_stream(&mut self, inst: OutInstance, ts: SystemTime) {
        self.handler.debug(|| {
            format!("Evaluating stream {}: {}.", inst.0, self.ir.get_out(StreamReference::OutRef(inst.0)).name)
        });
        let (ix, _) = inst;
        for stmt in self.exprs[ix].stmts.clone() {
            self.eval_stmt(&stmt, &inst, ts);
        }
        let res = self.get(self.exprs[ix].stmts.last().unwrap().target, &inst);

        // Register value in global store.
        self.global_store.get_out_instance_mut(inst.clone()).unwrap().push_value(res.clone()); // TODO: unsafe unwrap.

        self.handler.output(|| format!("OutputStream[{}] := {:?}.", inst.0, res.clone()));

        // Check if we have to emit a warning.
        if let Value::Bool(true) = res {
            if let Some(trig) = self.is_trigger(inst.clone()) {
                self.handler
                    .trigger(|| format!("Trigger: {}", trig.message.as_ref().unwrap_or(&String::from("Warning!"))))
            }
        }

        // Check linked streams and inform them.
        let extended = self.ir.get_out(StreamReference::OutRef(ix));
        for win in &extended.dependent_windows {
            self.global_store.get_window_mut((win.ix, Vec::new())).accept_value(res.clone(), ts)
        }
        // TODO: Dependent streams?
    }

    fn is_trigger(&self, inst: OutInstance) -> Option<&Trigger> {
        self.ir.triggers.iter().find(|t| t.reference.out_ix() == inst.0)
    }

    pub(crate) fn accept_input(&mut self, input: StreamReference, v: Value, ts: SystemTime) {
        self.global_store.get_in_instance_mut(input).push_value(v.clone());
        self.handler.debug(|| format!("InputStream[{}] := {:?}.", input.in_ix(), v.clone()));
        let extended = self.ir.get_in(input);
        for win in &extended.dependent_windows {
            self.global_store.get_window_mut((win.ix, Vec::new())).accept_value(v.clone(), ts)
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

    fn write_forcefully(&mut self, temp: Temporary, value: Value, inst: &OutInstance) {
        self.temp_stores[inst.0].write_value_forcefully(temp, value);
    }

    fn eval_stmt(&mut self, stmt: &Statement, inst: &OutInstance, ts: SystemTime) {
        match &stmt.op {
            Op::Convert => {
                let arg = stmt.args[0];
                let v = self.get(arg, inst);
                self.write_forcefully(stmt.target, v, inst);
            }
            Op::Move => {
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
                    Constant::Float(f) => Value::Float((*f).into()),
                    Constant::Str(_) => unimplemented!(),
                };
                self.write(stmt.target, val, inst);
            }
            Op::ArithLog(op) => {
                use lola_parser::ir::ArithLogOp::*;
                // The explicit match here enables a compiler warning when a case was missed.
                // Useful when the list in the parser is extended.
                let arity = match op {
                    Neg | Not => 1,
                    Add | Sub | Mul | Div | Rem | Pow | And | Or | Eq | Lt | Le | Ne | Ge | Gt => 2,
                };
                match arity {
                    1 => {
                        let operand = self.get(stmt.args[0], inst);
                        self.write(stmt.target, !operand, inst)
                    }
                    2 => {
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
                    _ => unreachable!(),
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
                match arg {
                    Value::Float(f) => {
                        let res = match name.as_ref() {
                            "sqrt" => f.sqrt(),
                            "sin" => f.sin(),
                            "cos" => f.cos(),
                            "arctan" => f.atan(),
                            "abs" => f.abs(),
                            _ => panic!("Unknown function."),
                        };
                        let res = Value::Float(NotNan::new(res).expect("TODO: Handle"));
                        self.write(stmt.target, res, inst);
                    }
                    Value::Signed(i) => {
                        let res = match name.as_ref() {
                            "abs" => i.abs(),
                            _ => panic!("Unknown function."),
                        };
                        self.write(stmt.target, Value::Signed(res), inst);
                    }
                    _ => panic!("Unknown function."),
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

    #[cfg(test)]
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

#[cfg(test)]
mod tests {

    use super::*;
    use lola_parser::ir::LolaIR;
    use std::time::{Duration, SystemTime};

    fn setup(spec: &str) -> (LolaIR, Evaluator) {
        setup_time(spec, SystemTime::now())
    }

    fn setup_time(spec: &str, ts: SystemTime) -> (LolaIR, Evaluator) {
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
        eval.eval_stream(inst.clone(), SystemTime::now());
        assert_eq!(eval.__peek_value(StreamReference::OutRef(0), &Vec::new(), 0).unwrap(), Value::Unsigned(3))
    }

    #[test]
    fn test_const_output_arith() {
        let (_, mut eval) = setup("output a: UInt8 := 3 + 5");
        let inst = (0, Vec::new());
        eval.eval_stream(inst.clone(), SystemTime::now());
        assert_eq!(eval.__peek_value(StreamReference::OutRef(0), &Vec::new(), 0).unwrap(), Value::Unsigned(8))
    }

    #[test]
    fn test_input_only() {
        let (_, mut eval) = setup("input a: UInt8");
        let sr = StreamReference::InRef(0);
        let v = Value::Unsigned(3);
        eval.accept_input(sr, v.clone(), SystemTime::now());
        assert_eq!(eval.__peek_value(sr, &Vec::new(), 0).unwrap(), v)
    }

    #[test]
    fn test_sync_lookup() {
        let (_, mut eval) = setup("input a: UInt8 output b: UInt8 := a");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v = Value::Unsigned(9);
        eval.accept_input(in_ref, v.clone(), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v)
    }

    #[test]
    fn test_oob_lookup() {
        let (_, mut eval) = setup("input a: UInt8\noutput b: UInt8 @5Hz := a[-1] ! 3");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        eval.accept_input(in_ref, v1, SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
    }

    #[test]
    fn test_output_lookup() {
        let (_, mut eval) = setup("input a: UInt8\noutput mirror: UInt8 := a\noutput c: UInt8 @5Hz := mirror[-1] ! 3");
        let out_ref = StreamReference::OutRef(1);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        eval.accept_input(in_ref, v1.clone(), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.accept_input(in_ref, v2, SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
    }

    #[test]
    fn test_conversion_if() {
        let (_, mut eval) = setup("input a: UInt8\noutput b: UInt16 := if true then a else a[-1].defaults(to: 0)");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        eval.accept_input(in_ref, v1.clone(), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
    }

    #[test]
    #[ignore] // See issue #32 in LolaParser.
    fn test_conversion_lookup() {
        let (_, mut eval) = setup("input a: UInt8\noutput b: UInt32 := a + 100000");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let expected = Value::Unsigned(7 + 100000);
        let v1 = Value::Unsigned(7);
        eval.accept_input(in_ref, v1, SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
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
        let expected = Value::Unsigned(1 + 2);
        eval.accept_input(a, v1.clone(), SystemTime::now());
        eval.accept_input(b, v2.clone(), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_bin_op_float() {
        let (_, mut eval) = setup("input a: Float64\n input b: Float64\noutput c: Float64 := a + b");
        let out_ref = StreamReference::OutRef(0);
        let a = StreamReference::InRef(0);
        let b = StreamReference::InRef(1);
        let v1 = Value::Float(NotNan::new(3.5f64).unwrap());
        let v2 = Value::Float(NotNan::new(39.347568f64).unwrap());
        let expected = Value::Float(NotNan::new(3.5f64 + 39.347568f64).unwrap());
        eval.accept_input(a, v1.clone(), SystemTime::now());
        eval.accept_input(b, v2.clone(), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_regular_lookup() {
        let (_, mut eval) = setup("input a: UInt8 output b: UInt8 @5Hz := a[-1] ! 3");
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        let v3 = Value::Unsigned(3);
        eval.accept_input(in_ref, v1, SystemTime::now());
        eval.accept_input(in_ref, v2.clone(), SystemTime::now());
        eval.accept_input(in_ref, v3, SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v2)
    }

    #[test]
    fn test_trigger() {
        let (_, mut eval) = setup("input a: UInt8 output b: UInt8 @5Hz := a[-1] ! 3\n trigger b > 4");
        let out_ref = StreamReference::OutRef(0);
        let trig_ref = StreamReference::OutRef(1);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(8);
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
        assert_eq!(eval.__peek_value(trig_ref, &Vec::new(), 0).unwrap(), Value::Bool(false));
        eval.accept_input(in_ref, v1.clone(), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
        assert_eq!(eval.__peek_value(trig_ref, &Vec::new(), 0).unwrap(), Value::Bool(false));
        eval.accept_input(in_ref, Value::Unsigned(17), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
        assert_eq!(eval.__peek_value(trig_ref, &Vec::new(), 0).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_sum_window() {
        let mut time = SystemTime::now();
        let (_, mut eval) = setup_time("input a: Int16\noutput b: Int16 @0.25Hz := a[40s, sum].defaults(to: -3)", time);
        time += Duration::from_secs(45);
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let n = 25;
        for v in 1..=n {
            eval.accept_input(in_ref, Value::Signed(v), time);
            time += Duration::from_secs(1);
        }
        time += Duration::from_secs(1);
        // 71 secs have passed. All values should be within the window.
        eval.eval_stream((0, Vec::new()), time);
        let expected = Value::Signed((n * n + n) / 2);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_count_window() {
        let mut time = SystemTime::now();
        let (_, mut eval) = setup_time("input a: UInt16\noutput b: UInt16 @0.25Hz := a[40s, #].defaults(to: 3)", time);
        time += Duration::from_secs(45);
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let n = 25;
        for v in 1..=n {
            eval.accept_input(in_ref, Value::Unsigned(v), time);
            time += Duration::from_secs(1);
        }
        time += Duration::from_secs(1);
        // 71 secs have passed. All values should be within the window.
        eval.eval_stream((0, Vec::new()), time);
        let expected = Value::Unsigned(n);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_integral_window() {
        let mut time = SystemTime::now();
        let (_, mut eval) =
            setup_time("input a: Float64\noutput b: Float64 @0.25Hz := a[40s, integral].defaults(to: -3.0)", time);
        time += Duration::from_secs(45);
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);

        fn mv(f: f64) -> Value {
            Value::Float(NotNan::new(f).unwrap())
        }

        eval.accept_input(in_ref, mv(1f64), time);
        time += Duration::from_secs(2);
        eval.accept_input(in_ref, mv(5f64), time);
        // Value so far: (1+5) / 2 * 2 = 6
        time += Duration::from_secs(5);
        eval.accept_input(in_ref, mv(25f64), time);
        // Value so far: 6 + (5+25) / 2 * 5 = 6 + 75 = 81
        time += Duration::from_secs(1);
        eval.accept_input(in_ref, mv(0f64), time);
        // Value so far: 81 + (25+0) / 2 * 1 = 81 + 12.5 = 93.5
        time += Duration::from_secs(10);
        eval.accept_input(in_ref, mv(-40f64), time);
        // Value so far: 93.5 + (0+(-40)) / 2 * 10 = 93.5 - 200 = -106.5
        // Time passed: 2 + 5 + 1 + 10 = 18.

        eval.eval_stream((0, Vec::new()), time);

        let expected = Value::Float(NotNan::new(-106.5).unwrap());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }
}
