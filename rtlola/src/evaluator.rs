use streamlab_frontend::ir::{
    Activation, Constant, Expression, LolaIR, Offset, StreamAccessKind, StreamReference, Trigger, Type, WindowReference,
};

use crate::basics::{EvalConfig, OutputHandler};
use crate::closuregen::{CompiledExpr, Expr};
use crate::storage::{GlobalStore, Value};
use std::sync::Arc;

pub(crate) type OutInstance = (usize, Vec<Value>);
pub(crate) type Window = (usize, Vec<Value>);
use bit_set::BitSet;
use ordered_float::NotNan;

use std::time::SystemTime;

enum ActivationCondition {
    //TODO(marvin): implement fast path for conjunctions
    TimeDriven,
    General(Activation<StreamReference>),
}

pub(crate) struct EvaluatorData<'c> {
    // Evaluation order of output streams
    layers: Vec<Vec<StreamReference>>,
    // Indexed by stream reference.
    activation_conditions: Vec<ActivationCondition>,
    // Indexed by stream reference.
    exprs: Vec<Expression>,
    // Indexed by stream reference.
    compiled_exprs: Vec<CompiledExpr<'c>>,
    global_store: GlobalStore,
    fresh_inputs: BitSet,
    fresh_outputs: BitSet,
    ir: LolaIR,
    handler: Arc<OutputHandler>,
    config: EvalConfig,
}

pub(crate) struct Evaluator<'e, 'c> {
    // Evaluation order of output streams
    layers: &'e Vec<Vec<StreamReference>>,
    // Indexed by stream reference.
    activation_conditions: &'e Vec<ActivationCondition>,
    // Indexed by stream reference.
    exprs: &'e Vec<Expression>,
    // Indexed by stream reference.
    compiled_exprs: &'e Vec<CompiledExpr<'c>>,
    global_store: &'e mut GlobalStore,
    fresh_inputs: &'e mut BitSet,
    fresh_outputs: &'e mut BitSet,
    ir: &'e LolaIR,
    handler: &'e OutputHandler,
    config: &'e EvalConfig,
}

pub(crate) struct ExpressionEvaluator<'e> {
    global_store: &'e GlobalStore,
    fresh_inputs: &'e BitSet,
    fresh_outputs: &'e BitSet,
}

pub(crate) struct EvaluationContext<'e> {
    ts: SystemTime,
    global_store: &'e GlobalStore,
    fresh_inputs: &'e BitSet,
    fresh_outputs: &'e BitSet,
}

impl<'c> EvaluatorData<'c> {
    pub(crate) fn new(
        ir: LolaIR,
        ts: SystemTime,
        config: EvalConfig,
        handler: Arc<OutputHandler>,
    ) -> EvaluatorData<'c> {
        // Layers of event based output streams
        let layers = ir.get_event_driven_layers();
        let activation_conditions = ir
            .outputs
            .iter()
            .map(|o| if let Some(ac) = &o.ac { ac.clone().into() } else { ActivationCondition::TimeDriven })
            .collect();
        let exprs = ir.outputs.iter().map(|o| o.expr.clone()).collect();
        let compiled_exprs = if config.closure_based_evaluator {
            ir.outputs.iter().map(|o| o.expr.clone().compile()).collect()
        } else {
            vec![]
        };
        let global_store = GlobalStore::new(&ir, ts);
        let fresh_inputs = BitSet::with_capacity(ir.inputs.len());
        let fresh_outputs = BitSet::with_capacity(ir.outputs.len());
        handler.debug(|| format!("Evaluation layers: {:?}", layers));
        EvaluatorData {
            layers,
            activation_conditions,
            exprs,
            compiled_exprs,
            global_store,
            fresh_inputs,
            fresh_outputs,
            ir,
            handler,
            config,
        }
    }

    #[allow(non_snake_case)]
    pub(crate) fn as_Evaluator<'n>(&'n mut self) -> Evaluator<'n, 'c> {
        Evaluator {
            layers: &self.layers,
            activation_conditions: &self.activation_conditions,
            exprs: &self.exprs,
            compiled_exprs: &self.compiled_exprs,
            global_store: &mut self.global_store,
            fresh_inputs: &mut self.fresh_inputs,
            fresh_outputs: &mut self.fresh_outputs,
            ir: &self.ir,
            handler: &self.handler,
            config: &self.config,
        }
    }
}

impl<'e, 'c> Evaluator<'e, 'c> {
    pub(crate) fn eval_event(&mut self, event: &Vec<(StreamReference, Value)>, ts: SystemTime) {
        self.clear_freshness();
        self.accept_inputs(event, ts);
        self.eval_all_event_driven_outputs(ts);
    }

    fn accept_inputs(&mut self, event: &Vec<(StreamReference, Value)>, ts: SystemTime) {
        for (str_ref, v) in event {
            self.accept_input(*str_ref, v.clone(), ts);
        }
    }

    fn accept_input(&mut self, input: StreamReference, v: Value, ts: SystemTime) {
        self.global_store.get_in_instance_mut(input).push_value(v.clone());
        self.fresh_inputs.insert(input.in_ix());
        self.handler.debug(|| format!("InputStream[{}] := {:?}.", input.in_ix(), v.clone()));
        let extended = self.ir.get_in(input);
        for win in &extended.dependent_windows {
            self.global_store.get_window_mut((win.ix, Vec::new())).accept_value(v.clone(), ts)
        }
    }

    pub fn eval_all_event_driven_outputs(&mut self, ts: SystemTime) {
        self.prepare_evaluation(ts);
        for layer in self.layers {
            self.eval_event_driven_outputs(layer, ts);
        }
    }

    fn eval_event_driven_outputs(&mut self, streams: &Vec<StreamReference>, ts: SystemTime) {
        for str_ref in streams {
            self.eval_event_driven_output(*str_ref, ts);
        }
    }

    fn eval_event_driven_output(&mut self, stream: StreamReference, ts: SystemTime) {
        if self.activation_conditions[stream.out_ix()].eval(self.fresh_inputs) {
            self.eval_output(stream, ts);
        }
    }

    pub(crate) fn eval_time_driven_outputs(&mut self, streams: &Vec<StreamReference>, ts: SystemTime) {
        self.clear_freshness();
        self.prepare_evaluation(ts);
        for str_ref in streams {
            self.eval_output(*str_ref, ts);
        }
    }

    fn prepare_evaluation(&mut self, ts: SystemTime) {
        // We need to copy the references first because updating needs exclusive access to `self`.
        let windows = &self.ir.sliding_windows;
        for win in windows {
            let ix = win.reference.ix;
            self.global_store.get_window_mut((ix, Vec::new())).update(ts);
        }
    }

    fn eval_output(&mut self, stream: StreamReference, ts: SystemTime) {
        let inst = (stream.out_ix(), Vec::new());
        self.eval_stream(inst, ts);
    }

    fn eval_stream(&mut self, inst: OutInstance, ts: SystemTime) {
        let (ix, _) = inst;
        self.handler
            .debug(|| format!("Evaluating stream {}: {}.", ix, self.ir.get_out(StreamReference::OutRef(ix)).name));

        let res = if self.config.closure_based_evaluator {
            let (ctx, compiled_exprs) = self.as_EvaluationContext(ts);
            compiled_exprs[ix].execute(&ctx)
        } else {
            let (expr_eval, exprs) = self.as_ExpressionEvaluator();
            expr_eval.eval_expr(&exprs[ix], ts)
        };

        // Register value in global store.
        self.global_store.get_out_instance_mut(inst.clone()).unwrap().push_value(res.clone()); // TODO: unsafe unwrap.
        self.fresh_outputs.insert(ix);

        self.handler.output(|| format!("OutputStream[{}] := {:?}.", ix, res.clone()));

        // Check if we have to emit a warning.
        if let Value::Bool(true) = res {
            //TODO(marvin): cache trigger info in vector
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

    fn clear_freshness(&mut self) {
        self.fresh_inputs.clear();
        self.fresh_outputs.clear();
    }

    fn is_trigger(&self, inst: OutInstance) -> Option<&Trigger> {
        self.ir.triggers.iter().find(|t| t.reference.out_ix() == inst.0)
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

    #[allow(non_snake_case)]
    fn as_ExpressionEvaluator<'n>(&'n self) -> (ExpressionEvaluator<'n>, &'e Vec<Expression>) {
        (
            ExpressionEvaluator {
                global_store: &self.global_store,
                fresh_inputs: &self.fresh_inputs,
                fresh_outputs: &self.fresh_outputs,
            },
            &self.exprs,
        )
    }

    #[allow(non_snake_case)]
    fn as_EvaluationContext<'n>(&'n self, ts: SystemTime) -> (EvaluationContext<'n>, &'e Vec<CompiledExpr<'c>>) {
        (
            EvaluationContext {
                ts,
                global_store: &self.global_store,
                fresh_inputs: &self.fresh_inputs,
                fresh_outputs: &self.fresh_outputs,
            },
            &self.compiled_exprs,
        )
    }
}

impl<'a> ExpressionEvaluator<'a> {
    fn eval_expr(&self, expr: &Expression, ts: SystemTime) -> Value {
        use streamlab_frontend::ir::Expression::*;
        match expr {
            LoadConstant(c) => match c {
                Constant::Bool(b) => Value::Bool(*b),
                Constant::UInt(u) => Value::Unsigned(*u),
                Constant::Int(i) => Value::Signed(*i),
                Constant::Float(f) => Value::Float((*f).into()),
                Constant::Str(s) => Value::Str(s.to_string()),
            },

            ArithLog(op, operands, _ty) => {
                use streamlab_frontend::ir::ArithLogOp::*;
                // The explicit match here enables a compiler warning when a case was missed.
                // Useful when the list in the parser is extended.
                let arity = match op {
                    Neg | Not => 1,
                    Add | Sub | Mul | Div | Rem | Pow | And | Or | Eq | Lt | Le | Ne | Ge | Gt => 2,
                };
                match arity {
                    1 => {
                        let operand = self.eval_expr(&operands[0], ts);
                        match *op {
                            Not => !operand,
                            Neg => -operand,
                            _ => unreachable!(),
                        }
                    }
                    2 => {
                        let lhs = self.eval_expr(&operands[0], ts);

                        if *op == And {
                            // evaluate lazy
                            if lhs.get_bool() == false {
                                return Value::Bool(false);
                            } else {
                                let rhs = self.eval_expr(&operands[1], ts);
                                return rhs;
                            }
                        }
                        if *op == Or {
                            // evaluate lazy
                            if lhs.get_bool() == true {
                                return Value::Bool(true);
                            } else {
                                let rhs = self.eval_expr(&operands[1], ts);
                                return rhs;
                            }
                        }

                        let rhs = self.eval_expr(&operands[1], ts);

                        match *op {
                            Add => lhs + rhs,
                            Sub => lhs - rhs,
                            Mul => lhs * rhs,
                            Div => lhs / rhs,
                            Rem => lhs % rhs,
                            Pow => lhs.pow(rhs),
                            Eq => Value::Bool(lhs == rhs),
                            Lt => Value::Bool(lhs < rhs),
                            Le => Value::Bool(lhs <= rhs),
                            Ne => Value::Bool(lhs != rhs),
                            Ge => Value::Bool(lhs >= rhs),
                            Gt => Value::Bool(lhs > rhs),
                            Not | Neg | And | Or => unreachable!(),
                        }
                    }
                    _ => unreachable!(),
                }
            }

            Ite { condition, consequence, alternative } => {
                if self.eval_expr(condition, ts).get_bool() {
                    self.eval_expr(consequence, ts)
                } else {
                    self.eval_expr(alternative, ts)
                }
            }

            SyncStreamLookup(str_ref) => {
                let res = self.perform_lookup(*str_ref, 0);
                assert!(res.is_some());
                // Unwrap sound in sync lookups.
                res.unwrap()
            }

            OffsetLookup { target: str_ref, offset } => {
                let res = match offset {
                    Offset::FutureDiscreteOffset(_) | Offset::FutureRealTimeOffset(_) => unimplemented!(),
                    Offset::PastDiscreteOffset(u) => self.perform_lookup(*str_ref, -(*u as i16)),
                    Offset::PastRealTimeOffset(_dur) => unimplemented!(),
                };
                res.unwrap_or(Value::None)
            }

            StreamAccess(str_ref, kind) => {
                use StreamAccessKind::*;
                match kind {
                    Hold => self.perform_lookup(*str_ref, 0).unwrap_or(Value::None),
                    Optional => {
                        use StreamReference::*;
                        match *str_ref {
                            InRef(ix) => {
                                if self.fresh_inputs.contains(ix) {
                                    self.perform_lookup(*str_ref, 0).expect("value must be there")
                                } else {
                                    Value::None
                                }
                            }
                            OutRef(ix) => {
                                if self.fresh_outputs.contains(ix) {
                                    self.perform_lookup(*str_ref, 0).expect("value must be there")
                                } else {
                                    Value::None
                                }
                            }
                        }
                    }
                }
            }

            WindowLookup(win_ref) => {
                let window: Window = (win_ref.ix, Vec::new());
                self.global_store.get_window(window).get_value(ts)
            }

            Expression::Function(name, args, _ty) => {
                //TODO(marvin): handle type
                let arg = self.eval_expr(&args[0], ts);
                match arg {
                    Value::Float(f) => {
                        let res = match name.as_ref() {
                            "sqrt" => f.sqrt(),
                            "sin" => f.sin(),
                            "cos" => f.cos(),
                            "arctan" => f.atan(),
                            "abs" => f.abs(),
                            _ => panic!("Unknown function: {}", name),
                        };
                        Value::Float(NotNan::new(res).expect("TODO: Handle"))
                    }
                    Value::Signed(i) => {
                        let res = match name.as_ref() {
                            "abs" => i.abs(),
                            _ => panic!("Unknown function: {}", name),
                        };
                        Value::Signed(res)
                    }
                    _ => panic!("Unknown function: {}", name),
                }
            }

            Tuple(_entries) => unimplemented!("Who needs tuples, anyway?"),

            Convert { from, to, expr } => {
                use Type::*;
                let v = self.eval_expr(expr, ts);
                match (from, v) {
                    (UInt(_), Value::Unsigned(u)) => match to {
                        UInt(_) => Value::Unsigned(u),
                        Int(_) => Value::Signed(u as i64),
                        Float(_) => Value::new_float(u as f64),
                        _ => unimplemented!(),
                    },
                    (Int(_), Value::Signed(i)) => match to {
                        UInt(_) => Value::Unsigned(i as u64),
                        Int(_) => Value::Signed(i),
                        Float(_) => Value::new_float(i as f64),
                        _ => unimplemented!(),
                    },
                    (Float(_), Value::Float(f)) => match to {
                        UInt(_) => Value::Unsigned(f.into_inner() as u64),
                        Int(_) => Value::Signed(f.into_inner() as i64),
                        Float(_) => Value::new_float(f.into_inner() as f64),
                        _ => unimplemented!(),
                    },
                    _ => unimplemented!(),
                }
            }

            Default { expr, default } => {
                let v = self.eval_expr(expr, ts);
                if let Value::None = v {
                    self.eval_expr(default, ts)
                } else {
                    v
                }
            }
        }
    }

    fn perform_lookup(&self, str_ref: StreamReference, offset: i16) -> Option<Value> {
        let is = match str_ref {
            StreamReference::InRef(_) => Some(self.global_store.get_in_instance(str_ref)),
            StreamReference::OutRef(i) => {
                let target: OutInstance = (i, Vec::new());
                self.global_store.get_out_instance(target)
                //TODO(marvin): shouldn't this panic if there is no instance?
            }
        };
        is.and_then(|is| is.get_value(offset))
    }
}

impl<'e> EvaluationContext<'e> {
    pub(crate) fn lookup(&self, target: StreamReference) -> Value {
        self.lookup_with_offset(target, 0)
    }

    pub(crate) fn lookup_with_offset(&self, stream_ref: StreamReference, offset: i16) -> Value {
        let inst_opt = match stream_ref {
            StreamReference::InRef(_) => Some(self.global_store.get_in_instance(stream_ref)),
            StreamReference::OutRef(i) => {
                let inst: OutInstance = (i, Vec::new());
                self.global_store.get_out_instance(inst)
                //TODO(marvin): shouldn't this panic if there is no instance?
            }
        };
        match inst_opt {
            Some(inst) => inst.get_value(offset).unwrap_or(Value::None),
            None => Value::None,
        }
    }

    pub(crate) fn lookup_window(&self, window_ref: WindowReference) -> Value {
        let window: Window = (window_ref.ix, Vec::new());
        self.global_store.get_window(window).get_value(self.ts)
    }
}

impl From<Activation<StreamReference>> for ActivationCondition {
    fn from(ac: Activation<StreamReference>) -> Self {
        ActivationCondition::General(ac)
    }
}

impl ActivationCondition {
    fn eval(&self, inputs: &BitSet) -> bool {
        use ActivationCondition::*;
        match self {
            General(ac) => Self::eval_(ac, inputs),
            TimeDriven => panic!(),
        }
    }
    fn eval_(ac: &Activation<StreamReference>, inputs: &BitSet) -> bool {
        use Activation::*;
        match ac {
            True => true,
            Stream(var) => inputs.contains(var.in_ix()),
            Conjunction(vec) => vec.iter().all(|ac| Self::eval_(ac, inputs)),
            Disjunction(vec) => vec.iter().any(|ac| Self::eval_(ac, inputs)),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::storage::Value::*;
    use std::time::{Duration, SystemTime};
    use streamlab_frontend::ir::LolaIR;

    fn setup(spec: &str) -> (LolaIR, EvaluatorData) {
        setup_time(spec, SystemTime::now())
    }

    fn setup_time(spec: &str, ts: SystemTime) -> (LolaIR, EvaluatorData) {
        let ir = streamlab_frontend::parse(spec);
        let mut config = EvalConfig::default();
        config.verbosity = crate::basics::Verbosity::WarningsOnly;
        let handler = Arc::new(OutputHandler::new(&config));
        let eval = EvaluatorData::new(ir.clone(), ts, config, handler);
        (ir, eval)
    }

    macro_rules! peek_assert_eq {
        ($eval:expr,$str_ref:expr,$value:expr) => {
            $eval.eval_stream(($str_ref, Vec::new()), SystemTime::now());
            assert_eq!($eval.__peek_value(StreamReference::OutRef($str_ref), &Vec::new(), 0).unwrap(), $value);
        };
    }

    #[test]
    fn test_empty_outputs() {
        setup("input a: UInt8");
    }

    #[test]
    fn test_const_output_literals() {
        let (_, mut eval) = setup(
            r#"
        output o_0: Bool := true
        output o_1: UInt8 := 3
        output o_2: Int8 := -5
        output o_3: Float32 := -123.456
        "#,
        );
        let mut eval = eval.as_Evaluator();
        peek_assert_eq!(eval, 0, Bool(true));
        peek_assert_eq!(eval, 1, Unsigned(3));
        peek_assert_eq!(eval, 2, Signed(-5));
        peek_assert_eq!(eval, 3, Value::new_float(-123.456));
    }

    #[test]
    fn test_const_output_arithlog() {
        let (_, mut eval) = setup(
            r#"
        output o_0:   Bool := !false
        output o_1:   Bool := !true
        output o_2:  UInt8 := 8 + 3
        output o_3:  UInt8 := 8 - 3
        output o_4:  UInt8 := 8 * 3
        output o_5:  UInt8 := 8 / 3
        output o_6:  UInt8 := 8 % 3
        output o_7:  UInt8 := 8 ** 3
        output o_8:   Bool := false || false
        output o_9:   Bool := false || true
        output o_10:  Bool := true  || false
        output o_11:  Bool := true  || true
        output o_12:  Bool := false && false
        output o_13:  Bool := false && true
        output o_14:  Bool := true  && false
        output o_15:  Bool := true  && true
        output o_16:  Bool := 0 < 1
        output o_17:  Bool := 0 < 0
        output o_18:  Bool := 1 < 0
        output o_19:  Bool := 0 <= 1
        output o_20:  Bool := 0 <= 0
        output o_21:  Bool := 1 <= 0
        output o_22:  Bool := 0 >= 1
        output o_23:  Bool := 0 >= 0
        output o_24:  Bool := 1 >= 0
        output o_25:  Bool := 0 > 1
        output o_26:  Bool := 0 > 0
        output o_27:  Bool := 1 > 0
        output o_28:  Bool := 0 == 0
        output o_29:  Bool := 0 == 1
        output o_30:  Bool := 0 != 0
        output o_31:  Bool := 0 != 1
        "#,
        );
        let mut eval = eval.as_Evaluator();
        peek_assert_eq!(eval, 0, Bool(!false));
        peek_assert_eq!(eval, 1, Bool(!true));
        peek_assert_eq!(eval, 2, Unsigned(8 + 3));
        peek_assert_eq!(eval, 3, Unsigned(8 - 3));
        peek_assert_eq!(eval, 4, Unsigned(8 * 3));
        peek_assert_eq!(eval, 5, Unsigned(8 / 3));
        peek_assert_eq!(eval, 6, Unsigned(8 % 3));
        peek_assert_eq!(eval, 7, Unsigned(8 * 8 * 8));
        peek_assert_eq!(eval, 8, Bool(false || false));
        peek_assert_eq!(eval, 9, Bool(false || true));
        peek_assert_eq!(eval, 10, Bool(true || false));
        peek_assert_eq!(eval, 11, Bool(true || true));
        peek_assert_eq!(eval, 12, Bool(false && false));
        peek_assert_eq!(eval, 13, Bool(false && true));
        peek_assert_eq!(eval, 14, Bool(true && false));
        peek_assert_eq!(eval, 15, Bool(true && true));
        peek_assert_eq!(eval, 16, Bool(0 < 1));
        peek_assert_eq!(eval, 17, Bool(0 < 0));
        peek_assert_eq!(eval, 18, Bool(1 < 0));
        peek_assert_eq!(eval, 19, Bool(0 <= 1));
        peek_assert_eq!(eval, 20, Bool(0 <= 0));
        peek_assert_eq!(eval, 21, Bool(1 <= 0));
        peek_assert_eq!(eval, 22, Bool(0 >= 1));
        peek_assert_eq!(eval, 23, Bool(0 >= 0));
        peek_assert_eq!(eval, 24, Bool(1 >= 0));
        peek_assert_eq!(eval, 25, Bool(0 > 1));
        peek_assert_eq!(eval, 26, Bool(0 > 0));
        peek_assert_eq!(eval, 27, Bool(1 > 0));
        peek_assert_eq!(eval, 28, Bool(0 == 0));
        peek_assert_eq!(eval, 29, Bool(0 == 1));
        peek_assert_eq!(eval, 30, Bool(0 != 0));
        peek_assert_eq!(eval, 31, Bool(0 != 1));
    }

    #[test]
    fn test_input_only() {
        let (_, mut eval) = setup("input a: UInt8");
        let mut eval = eval.as_Evaluator();
        let sr = StreamReference::InRef(0);
        let v = Value::Unsigned(3);
        eval.accept_input(sr, v.clone(), SystemTime::now());
        assert_eq!(eval.__peek_value(sr, &Vec::new(), 0).unwrap(), v)
    }

    #[test]
    fn test_sync_lookup() {
        let (_, mut eval) = setup("input a: UInt8 output b: UInt8 := a");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v = Value::Unsigned(9);
        eval.accept_input(in_ref, v.clone(), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v)
    }

    #[test]
    fn test_oob_lookup() {
        let (_, mut eval) =
            setup("input a: UInt8\noutput b := a.offset(by: -1)\noutput x: UInt8 @5Hz := b.hold().defaults(to: 3)");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(1);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        eval.accept_input(in_ref, v1, SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
    }

    #[test]
    fn test_output_lookup() {
        let (_, mut eval) = setup(
            "input a: UInt8\noutput mirror: UInt8 := a\noutput mirror_offset := mirror.offset(by: -1)\noutput c: UInt8 @5Hz := mirror_offset.hold().defaults(to: 3)",
        );
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(2);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        eval.accept_input(in_ref, v1.clone(), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        eval.accept_input(in_ref, v2, SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        eval.eval_stream((2, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
    }

    #[test]
    fn test_conversion_if() {
        let (_, mut eval) = setup("input a: UInt8\noutput b: UInt16 := if true then a else a[-1].defaults(to: 0)");
        let mut eval = eval.as_Evaluator();
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
        let mut eval = eval.as_Evaluator();
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
        let mut eval = eval.as_Evaluator();
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
        let mut eval = eval.as_Evaluator();
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
        let (_, mut eval) =
            setup("input a: UInt8 output b := a.offset(by: -1) output x: UInt8 @5Hz := b.hold().defaults(to: 3)");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(1);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        let v3 = Value::Unsigned(3);
        eval.accept_input(in_ref, v1, SystemTime::now());
        eval.accept_input(in_ref, v2.clone(), SystemTime::now());
        eval.accept_input(in_ref, v3, SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v2)
    }

    #[test]
    fn test_trigger() {
        let (_, mut eval) =
            setup("input a: UInt8 output b := a.offset(by: -1) output x: UInt8 @5Hz := b.hold().defaults(to: 3)\n trigger x > 4");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(1);
        let trig_ref = StreamReference::OutRef(2);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(8);
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        eval.eval_stream((2, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
        assert_eq!(eval.__peek_value(trig_ref, &Vec::new(), 0).unwrap(), Value::Bool(false));
        eval.accept_input(in_ref, v1.clone(), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        eval.eval_stream((2, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
        assert_eq!(eval.__peek_value(trig_ref, &Vec::new(), 0).unwrap(), Value::Bool(false));
        eval.accept_input(in_ref, Value::Unsigned(17), SystemTime::now());
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        eval.eval_stream((1, Vec::new()), SystemTime::now());
        eval.eval_stream((2, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
        assert_eq!(eval.__peek_value(trig_ref, &Vec::new(), 0).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_sum_window() {
        let mut time = SystemTime::now();
        let (_, mut eval) = setup_time(
            "input a: Int16\noutput b: Int16 @0.25Hz := a.aggregate(over: 40s, using: sum).defaults(to: -3)",
            time,
        );
        let mut eval = eval.as_Evaluator();
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
        let (_, mut eval) = setup_time(
            "input a: UInt16\noutput b: UInt16 @0.25Hz := a.aggregate(over: 40s, using: #).defaults(to: 3)",
            time,
        );
        let mut eval = eval.as_Evaluator();
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
        let (_, mut eval) = setup_time(
            "input a: Float64\noutput b: Float64 @0.25Hz := a.aggregate(over: 40s, using: integral).defaults(to: -3.0)",
            time,
        );
        let mut eval = eval.as_Evaluator();
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

    #[test]
    fn test_window_type_count() {
        let (_, mut eval) =
            setup_time("input a: Int32\noutput b @ 10Hz := a.aggregate(over: 0.1s, using: count)", SystemTime::now());
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(0);
        let a = StreamReference::InRef(0);
        let expected = Value::Unsigned(0);
        eval.eval_stream((0, Vec::new()), SystemTime::now());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }
}
