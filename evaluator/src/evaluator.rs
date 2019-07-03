use crate::basics::{EvalConfig, EvaluatorChoice::*, ExecutionMode, OutputHandler, Time};
use crate::closuregen::{CompiledExpr, Expr};
use crate::storage::{GlobalStore, Value};
use bit_set::BitSet;
use ordered_float::NotNan;
use regex::Regex;
use std::sync::Arc;
use std::time::Instant;
use streamlab_frontend::ir::{
    Activation, Constant, Expression, InputReference, LolaIR, Offset, OutputReference, StreamAccessKind,
    StreamReference, Trigger, Type, WindowReference,
};

pub(crate) enum ActivationCondition {
    TimeDriven,
    True,
    Conjunction(BitSet),
    General(Activation<StreamReference>),
}

pub(crate) struct EvaluatorData<'c> {
    // Evaluation order of output streams
    layers: Vec<Vec<OutputReference>>,
    // Indexed by stream reference.
    activation_conditions: Vec<ActivationCondition>,
    // Indexed by stream reference.
    exprs: Vec<Expression>,
    // Indexed by stream reference.
    compiled_exprs: Vec<CompiledExpr<'c>>,
    global_store: GlobalStore,
    start_time: Instant,           // only valid in online mode
    time_last_event: Option<Time>, // only valid in offline mode
    fresh_inputs: BitSet,
    fresh_outputs: BitSet,
    triggers: Vec<Option<Trigger>>,
    ir: LolaIR,
    handler: Arc<OutputHandler>,
    config: EvalConfig,
}

pub(crate) struct Evaluator<'e, 'c> {
    // Evaluation order of output streams
    layers: &'e Vec<Vec<OutputReference>>,
    // Indexed by stream reference.
    activation_conditions: &'e Vec<ActivationCondition>,
    // Indexed by stream reference.
    exprs: &'e Vec<Expression>,
    // Indexed by stream reference.
    compiled_exprs: &'e Vec<CompiledExpr<'c>>,
    global_store: &'e mut GlobalStore,
    start_time: &'e Instant,               // only valid in online mode
    time_last_event: &'e mut Option<Time>, // only valid in offline mode
    fresh_inputs: &'e mut BitSet,
    fresh_outputs: &'e mut BitSet,
    triggers: &'e Vec<Option<Trigger>>,
    ir: &'e LolaIR,
    handler: &'e OutputHandler,
    config: &'e EvalConfig,
}

struct ExpressionEvaluator<'e> {
    global_store: &'e GlobalStore,
    fresh_inputs: &'e BitSet,
    fresh_outputs: &'e BitSet,
}

pub(crate) struct EvaluationContext<'e> {
    ts: Time,
    pub(crate) global_store: &'e GlobalStore,
    pub(crate) fresh_inputs: &'e BitSet,
    pub(crate) fresh_outputs: &'e BitSet,
}

impl<'c> EvaluatorData<'c> {
    pub(crate) fn new(ir: LolaIR, config: EvalConfig, handler: Arc<OutputHandler>, start_time: Instant) -> Self {
        // Layers of event based output streams
        let layers = ir.get_event_driven_layers();
        handler.debug(|| format!("Evaluation layers: {:?}", layers));
        let activation_conditions = ir
            .outputs
            .iter()
            .map(|o| {
                if let Some(ac) = &o.ac {
                    ActivationCondition::new(ac, ir.inputs.len())
                } else {
                    ActivationCondition::TimeDriven
                }
            })
            .collect();
        let exprs = ir.outputs.iter().map(|o| o.expr.clone()).collect();
        let compiled_exprs = if config.evaluator == ClosureBased {
            ir.outputs.iter().map(|o| o.expr.clone().compile()).collect()
        } else {
            vec![]
        };
        let global_store = GlobalStore::new(&ir, Time::default());
        let fresh_inputs = BitSet::with_capacity(ir.inputs.len());
        let fresh_outputs = BitSet::with_capacity(ir.outputs.len());
        let mut triggers = vec![None; ir.outputs.len()];
        for t in &ir.triggers {
            triggers[t.reference.out_ix()] = Some(t.clone());
        }
        EvaluatorData {
            layers,
            activation_conditions,
            exprs,
            compiled_exprs,
            global_store,
            start_time,
            time_last_event: None,
            fresh_inputs,
            fresh_outputs,
            triggers,
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
            start_time: &self.start_time,
            time_last_event: &mut self.time_last_event,
            fresh_inputs: &mut self.fresh_inputs,
            fresh_outputs: &mut self.fresh_outputs,
            triggers: &self.triggers,
            ir: &self.ir,
            handler: &self.handler,
            config: &self.config,
        }
    }
}

impl<'e, 'c> Evaluator<'e, 'c> {
    pub(crate) fn eval_event(&mut self, event: &[Value], mut ts: Time) {
        if self.config.mode == ExecutionMode::Offline {
            assert!(
                self.time_last_event.is_none() || self.time_last_event.unwrap() <= ts,
                "time does not behave monotonic"
            );
            *self.time_last_event = Some(ts);
        } else {
            ts = self.start_time.elapsed();
        }
        self.clear_freshness();
        self.accept_inputs(event, ts);
        self.eval_all_event_driven_outputs(ts);
    }

    fn accept_inputs(&mut self, event: &[Value], ts: Time) {
        for (ix, v) in event.iter().enumerate() {
            match v {
                Value::None => {}
                v => self.accept_input(ix, v.clone(), ts),
            }
        }
    }

    fn accept_input(&mut self, input: InputReference, v: Value, ts: Time) {
        self.global_store.get_in_instance_mut(input).push_value(v.clone());
        self.fresh_inputs.insert(input);
        self.handler.debug(|| format!("InputStream[{}] := {:?}.", input, v.clone()));
        let extended = &self.ir.inputs[input];
        for win in &extended.dependent_windows {
            self.global_store.get_window_mut(win.ix).accept_value(v.clone(), ts)
        }
    }

    fn eval_all_event_driven_outputs(&mut self, ts: Time) {
        self.prepare_evaluation(ts);
        for layer in self.layers {
            self.eval_event_driven_outputs(layer, ts);
        }
    }

    fn eval_event_driven_outputs(&mut self, outputs: &[OutputReference], ts: Time) {
        for output in outputs {
            self.eval_event_driven_output(*output, ts);
        }
    }

    fn eval_event_driven_output(&mut self, output: OutputReference, ts: Time) {
        if self.activation_conditions[output].eval(self.fresh_inputs) {
            self.eval_stream(output, ts);
        }
    }

    pub(crate) fn eval_time_driven_outputs(&mut self, outputs: &[OutputReference], mut ts: Time) {
        if self.config.mode == ExecutionMode::Offline {
            assert!(
                self.time_last_event.is_none() || self.time_last_event.unwrap() <= ts,
                "time does not behave monotonic"
            );
            *self.time_last_event = Some(ts);
        } else {
            ts = self.start_time.elapsed();
        }
        self.clear_freshness();
        self.prepare_evaluation(ts);
        for output in outputs {
            self.eval_stream(*output, ts);
        }
        self.clear_freshness();
    }

    fn prepare_evaluation(&mut self, ts: Time) {
        // We need to copy the references first because updating needs exclusive access to `self`.
        let windows = &self.ir.sliding_windows;
        for win in windows {
            let ix = win.reference.ix;
            self.global_store.get_window_mut(ix).update(ts);
        }
    }

    fn eval_stream(&mut self, output: OutputReference, ts: Time) {
        let ix = output;
        self.handler
            .debug(|| format!("Evaluating stream {}: {}.", ix, self.ir.get_out(StreamReference::OutRef(ix)).name));

        let res = match self.config.evaluator {
            ClosureBased => {
                let (ctx, compiled_exprs) = self.as_EvaluationContext(ts);
                compiled_exprs[ix].execute(&ctx)
            }
            Interpreted => {
                let (expr_eval, exprs) = self.as_ExpressionEvaluator();
                expr_eval.eval_expr(&exprs[ix], ts)
            }
        };

        match self.is_trigger(output) {
            None => {
                // Register value in global store.
                self.global_store.get_out_instance_mut(output).unwrap().push_value(res.clone()); // TODO: unsafe unwrap.
                self.fresh_outputs.insert(ix);

                self.handler.output(|| format!("OutputStream[{}] := {:?}.", ix, res.clone()));
            }

            Some(trig) => {
                // Check if we have to emit a warning.
                if let Value::Bool(true) = res {
                    self.handler.trigger(|| format!("Trigger: {}", trig.message), trig.trigger_idx, ts)
                }
            }
        }

        // Check linked streams and inform them.
        let extended = &self.ir.outputs[ix];
        for win in &extended.dependent_windows {
            self.global_store.get_window_mut(win.ix).accept_value(res.clone(), ts)
        }
        // TODO: Dependent streams?
    }

    fn clear_freshness(&mut self) {
        self.fresh_inputs.clear();
        self.fresh_outputs.clear();
    }

    fn is_trigger(&self, ix: OutputReference) -> Option<&Trigger> {
        self.triggers[ix].as_ref()
    }

    #[cfg(test)]
    fn __peek_value(&self, sr: StreamReference, args: &[Value], offset: i16) -> Option<Value> {
        match sr {
            StreamReference::InRef(ix) => {
                assert!(args.is_empty());
                self.global_store.get_in_instance(ix).get_value(offset)
            }
            StreamReference::OutRef(ix) => {
                //let inst = (ix, Vec::from(args));
                let inst = ix;
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
    fn as_EvaluationContext<'n>(&'n self, ts: Time) -> (EvaluationContext<'n>, &'e Vec<CompiledExpr<'c>>) {
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
    fn eval_expr(&self, expr: &Expression, ts: Time) -> Value {
        use streamlab_frontend::ir::Expression::*;
        match expr {
            LoadConstant(c, _) => match c {
                Constant::Bool(b) => Value::Bool(*b),
                Constant::UInt(u) => Value::Unsigned(*u),
                Constant::Int(i) => Value::Signed(*i),
                Constant::Float(f) => Value::Float((*f).into()),
                Constant::Str(s) => Value::Str(s.clone().into_boxed_str()),
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

            Ite { condition, consequence, alternative, .. } => {
                if self.eval_expr(condition, ts).get_bool() {
                    self.eval_expr(consequence, ts)
                } else {
                    self.eval_expr(alternative, ts)
                }
            }

            SyncStreamLookup(str_ref) => self.lookup_latest_check(*str_ref),

            OffsetLookup { target: str_ref, offset } => match offset {
                Offset::FutureDiscreteOffset(_) | Offset::FutureRealTimeOffset(_) => unimplemented!(),
                Offset::PastDiscreteOffset(u) => self.lookup_with_offset(*str_ref, -(*u as i16)),
                Offset::PastRealTimeOffset(_dur) => unimplemented!(),
            },

            StreamAccess(str_ref, kind) => {
                use StreamAccessKind::*;
                match kind {
                    Hold => self.lookup_latest(*str_ref),
                    Optional => {
                        use StreamReference::*;
                        match *str_ref {
                            InRef(ix) => {
                                if self.fresh_inputs.contains(ix) {
                                    self.lookup_latest(*str_ref)
                                } else {
                                    Value::None
                                }
                            }
                            OutRef(ix) => {
                                if self.fresh_outputs.contains(ix) {
                                    self.lookup_latest(*str_ref)
                                } else {
                                    Value::None
                                }
                            }
                        }
                    }
                }
            }

            WindowLookup(win_ref) => self.lookup_window(*win_ref, ts),

            Function(name, args, _ty) => {
                //TODO(marvin): handle type
                assert!(!args.is_empty());
                let arg = self.eval_expr(&args[0], ts);
                match arg {
                    Value::Float(f) => {
                        let res = match name.as_ref() {
                            "sqrt" => f.sqrt(),
                            "sin" => f.sin(),
                            "cos" => f.cos(),
                            "arctan" => f.atan(),
                            "abs" => f.abs(),
                            _ => unreachable!("Unknown function: {}, args: {:?}", name, args),
                        };
                        Value::Float(NotNan::new(res).expect("TODO: Handle"))
                    }
                    Value::Signed(i) => {
                        let res = match name.as_ref() {
                            "abs" => i.abs(),
                            _ => unreachable!("Unknown function: {}, args: {:?}", name, args),
                        };
                        Value::Signed(res)
                    }
                    Value::Str(s) => match name.as_ref() {
                        "matches" => {
                            let re_str = match &args[1] {
                                Expression::LoadConstant(Constant::Str(s), _) => s,
                                _ => unreachable!("regex should be a string literal"),
                            };
                            // compiling regex every time it is used is a performance problem
                            // TODO: move it out of the eval loop
                            let re = Regex::new(&re_str).expect("Given regular expression was invalid");
                            Value::Bool(re.is_match(&s))
                        }
                        _ => unreachable!("unknown `String` function: {}, args: {:?}", name, args),
                    },
                    _ => unreachable!("Unknown function: {}, args: {:?}", name, args),
                }
            }

            Tuple(entries) => Value::Tuple(entries.iter().map(|e| self.eval_expr(e, ts)).collect()),

            Convert { from, to, expr } => {
                use Type::*;
                let v = self.eval_expr(expr, ts);
                match (from, v) {
                    (UInt(_), Value::Unsigned(u)) => match to {
                        UInt(_) => Value::Unsigned(u as u64),
                        Int(_) => Value::Signed(u as i64),
                        Float(_) => Value::new_float(u as f64),
                        _ => unreachable!(),
                    },
                    (Int(_), Value::Signed(i)) => match to {
                        UInt(_) => Value::Unsigned(i as u64),
                        Int(_) => Value::Signed(i as i64),
                        Float(_) => Value::new_float(i as f64),
                        _ => unreachable!(),
                    },
                    (Float(_), Value::Float(f)) => match to {
                        UInt(_) => Value::Unsigned(f.into_inner() as u64),
                        Int(_) => Value::Signed(f.into_inner() as i64),
                        Float(_) => Value::new_float(f.into_inner() as f64),
                        _ => unreachable!(),
                    },
                    (from, v) => panic!("Value type of {:?} does not match convert from type {:?}", v, from),
                }
            }

            Default { expr, default, .. } => {
                let v = self.eval_expr(expr, ts);
                if let Value::None = v {
                    self.eval_expr(default, ts)
                } else {
                    v
                }
            }

            TupleAccess(expr, num) => {
                if let Value::Tuple(entries) = self.eval_expr(expr, ts) {
                    entries[*num].clone()
                } else {
                    unreachable!("verified by type checker")
                }
            }
        }
    }

    fn lookup_latest(&self, stream_ref: StreamReference) -> Value {
        let inst = match stream_ref {
            StreamReference::InRef(ix) => self.global_store.get_in_instance(ix),
            StreamReference::OutRef(ix) => self.global_store.get_out_instance(ix).expect("no out instance"),
        };
        inst.get_value(0).unwrap_or(Value::None)
    }

    fn lookup_latest_check(&self, stream_ref: StreamReference) -> Value {
        let inst = match stream_ref {
            StreamReference::InRef(ix) => {
                debug_assert!(self.fresh_inputs.contains(ix), "ix={}", ix);
                self.global_store.get_in_instance(ix)
            }
            StreamReference::OutRef(ix) => {
                debug_assert!(self.fresh_outputs.contains(ix), "ix={}", ix);
                self.global_store.get_out_instance(ix).expect("no out instance")
            }
        };
        inst.get_value(0).unwrap_or(Value::None)
    }

    fn lookup_with_offset(&self, stream_ref: StreamReference, offset: i16) -> Value {
        let (inst, fresh) = match stream_ref {
            StreamReference::InRef(ix) => (self.global_store.get_in_instance(ix), self.fresh_inputs.contains(ix)),
            StreamReference::OutRef(ix) => {
                (self.global_store.get_out_instance(ix).expect("no out instance"), self.fresh_outputs.contains(ix))
            }
        };
        if fresh {
            inst.get_value(offset).unwrap_or(Value::None)
        } else {
            inst.get_value(offset + 1).unwrap_or(Value::None)
        }
    }

    fn lookup_window(&self, window_ref: WindowReference, ts: Time) -> Value {
        self.global_store.get_window(window_ref.ix).get_value(ts)
    }
}

impl<'e> EvaluationContext<'e> {
    pub(crate) fn lookup_latest(&self, stream_ref: StreamReference) -> Value {
        let inst = match stream_ref {
            StreamReference::InRef(ix) => self.global_store.get_in_instance(ix),
            StreamReference::OutRef(ix) => self.global_store.get_out_instance(ix).expect("no out instance"),
        };
        inst.get_value(0).unwrap_or(Value::None)
    }

    pub(crate) fn lookup_latest_check(&self, stream_ref: StreamReference) -> Value {
        let inst = match stream_ref {
            StreamReference::InRef(ix) => {
                debug_assert!(self.fresh_inputs.contains(ix), "ix={}", ix);
                self.global_store.get_in_instance(ix)
            }
            StreamReference::OutRef(ix) => {
                debug_assert!(self.fresh_outputs.contains(ix), "ix={}", ix);
                self.global_store.get_out_instance(ix).expect("no out instance")
            }
        };
        inst.get_value(0).unwrap_or(Value::None)
    }

    pub(crate) fn lookup_with_offset(&self, stream_ref: StreamReference, offset: i16) -> Value {
        let (inst, fresh) = match stream_ref {
            StreamReference::InRef(ix) => (self.global_store.get_in_instance(ix), self.fresh_inputs.contains(ix)),
            StreamReference::OutRef(ix) => {
                (self.global_store.get_out_instance(ix).expect("no out instance"), self.fresh_outputs.contains(ix))
            }
        };
        if fresh {
            inst.get_value(offset).unwrap_or(Value::None)
        } else {
            inst.get_value(offset + 1).unwrap_or(Value::None)
        }
    }

    pub(crate) fn lookup_window(&self, window_ref: WindowReference) -> Value {
        self.global_store.get_window(window_ref.ix).get_value(self.ts)
    }
}

impl ActivationCondition {
    fn new(ac: &Activation<StreamReference>, n_inputs: usize) -> Self {
        use ActivationCondition::*;
        if let Activation::True = ac {
            // special case for constant output streams
            return True;
        }
        if let Activation::Conjunction(vec) = ac {
            assert!(!vec.is_empty());
            let ixs: Vec<usize> = vec
                .iter()
                .flat_map(|ac| if let Activation::Stream(var) = ac { Some(var.in_ix()) } else { None })
                .collect();
            if vec.len() == ixs.len() {
                // fast path for conjunctive activation conditions
                let mut bs = BitSet::with_capacity(n_inputs);
                for ix in ixs {
                    bs.insert(ix);
                }
                return Conjunction(bs);
            }
        }
        General(ac.clone())
    }

    pub(crate) fn eval(&self, inputs: &BitSet) -> bool {
        use ActivationCondition::*;
        match self {
            True => true,
            Conjunction(bs) => bs.is_subset(inputs),
            General(ac) => Self::eval_(ac, inputs),
            TimeDriven => unreachable!(),
        }
    }
    fn eval_(ac: &Activation<StreamReference>, inputs: &BitSet) -> bool {
        use Activation::*;
        match ac {
            Stream(var) => inputs.contains(var.in_ix()),
            Conjunction(vec) => vec.iter().all(|ac| Self::eval_(ac, inputs)),
            Disjunction(vec) => vec.iter().any(|ac| Self::eval_(ac, inputs)),
            True => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::storage::Value::*;
    use std::time::{Duration, Instant};
    use streamlab_frontend::ir::LolaIR;
    use streamlab_frontend::TypeConfig;

    fn parse(spec: &str) -> Result<LolaIR, String> {
        streamlab_frontend::parse("stdin", spec, TypeConfig::default())
    }

    fn setup(spec: &str) -> (LolaIR, EvaluatorData, Instant) {
        let ir = parse(spec).unwrap_or_else(|e| panic!("spec is invalid: {}", e));
        let mut config = EvalConfig::default();
        config.verbosity = crate::basics::Verbosity::WarningsOnly;
        let handler = Arc::new(OutputHandler::new(&config, ir.triggers.len()));
        let now = Instant::now();
        let eval = EvaluatorData::new(ir.clone(), config, handler, now);
        (ir, eval, now)
    }

    fn setup_time(spec: &str) -> (LolaIR, EvaluatorData, Time) {
        let (ir, eval, _) = setup(spec);
        (ir, eval, Time::default())
    }

    macro_rules! eval_stream {
        ($eval:expr, $start:expr, $ix:expr) => {
            $eval.eval_stream($ix, $start.elapsed());
        };
    }

    macro_rules! eval_stream_timed {
        ($eval:expr, $ix:expr, $time:expr) => {
            $eval.eval_stream($ix, $time);
        };
    }

    macro_rules! accept_input {
        ($eval:expr, $start:expr, $str_ref:expr, $v:expr) => {
            $eval.accept_input($str_ref.in_ix(), $v.clone(), $start.elapsed());
        };
    }

    macro_rules! accept_input_timed {
        ($eval:expr, $str_ref:expr, $v:expr, $time:expr) => {
            $eval.accept_input($str_ref.in_ix(), $v.clone(), $time);
        };
    }

    macro_rules! peek_assert_eq {
        ($eval:expr, $start:expr, $ix:expr, $value:expr) => {
            eval_stream!($eval, $start, $ix);
            assert_eq!($eval.__peek_value(StreamReference::OutRef($ix), &Vec::new(), 0).unwrap(), $value);
        };
    }

    #[test]
    fn test_empty_outputs() {
        setup("input a: UInt8");
    }

    #[test]
    fn test_const_output_literals() {
        let (_, mut eval, start) = setup(
            r#"
        output o_0: Bool := true
        output o_1: UInt8 := 3
        output o_2: Int8 := -5
        output o_3: Float32 := -123.456
        output o_4: String := "foobar"
        "#,
        );
        let mut eval = eval.as_Evaluator();
        peek_assert_eq!(eval, start, 0, Bool(true));
        peek_assert_eq!(eval, start, 1, Unsigned(3));
        peek_assert_eq!(eval, start, 2, Signed(-5));
        peek_assert_eq!(eval, start, 3, Value::new_float(-123.456));
        peek_assert_eq!(eval, start, 4, Str("foobar".into()));
    }

    #[test]
    fn test_const_output_arithlog() {
        let (_, mut eval, start) = setup(
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
        peek_assert_eq!(eval, start, 0, Bool(!false));
        peek_assert_eq!(eval, start, 1, Bool(!true));
        peek_assert_eq!(eval, start, 2, Unsigned(8 + 3));
        peek_assert_eq!(eval, start, 3, Unsigned(8 - 3));
        peek_assert_eq!(eval, start, 4, Unsigned(8 * 3));
        peek_assert_eq!(eval, start, 5, Unsigned(8 / 3));
        peek_assert_eq!(eval, start, 6, Unsigned(8 % 3));
        peek_assert_eq!(eval, start, 7, Unsigned(8 * 8 * 8));
        peek_assert_eq!(eval, start, 8, Bool(false || false));
        peek_assert_eq!(eval, start, 9, Bool(false || true));
        peek_assert_eq!(eval, start, 10, Bool(true || false));
        peek_assert_eq!(eval, start, 11, Bool(true || true));
        peek_assert_eq!(eval, start, 12, Bool(false && false));
        peek_assert_eq!(eval, start, 13, Bool(false && true));
        peek_assert_eq!(eval, start, 14, Bool(true && false));
        peek_assert_eq!(eval, start, 15, Bool(true && true));
        peek_assert_eq!(eval, start, 16, Bool(0 < 1));
        peek_assert_eq!(eval, start, 17, Bool(0 < 0));
        peek_assert_eq!(eval, start, 18, Bool(1 < 0));
        peek_assert_eq!(eval, start, 19, Bool(0 <= 1));
        peek_assert_eq!(eval, start, 20, Bool(0 <= 0));
        peek_assert_eq!(eval, start, 21, Bool(1 <= 0));
        peek_assert_eq!(eval, start, 22, Bool(0 >= 1));
        peek_assert_eq!(eval, start, 23, Bool(0 >= 0));
        peek_assert_eq!(eval, start, 24, Bool(1 >= 0));
        peek_assert_eq!(eval, start, 25, Bool(0 > 1));
        peek_assert_eq!(eval, start, 26, Bool(0 > 0));
        peek_assert_eq!(eval, start, 27, Bool(1 > 0));
        peek_assert_eq!(eval, start, 28, Bool(0 == 0));
        peek_assert_eq!(eval, start, 29, Bool(0 == 1));
        peek_assert_eq!(eval, start, 30, Bool(0 != 0));
        peek_assert_eq!(eval, start, 31, Bool(0 != 1));
    }

    #[test]
    fn test_input_only() {
        let (_, mut eval, start) = setup("input a: UInt8");
        let mut eval = eval.as_Evaluator();
        let sr = StreamReference::InRef(0);
        let v = Value::Unsigned(3);
        accept_input!(eval, start, sr, v.clone());
        assert_eq!(eval.__peek_value(sr, &Vec::new(), 0).unwrap(), v)
    }

    #[test]
    fn test_sync_lookup() {
        let (_, mut eval, start) = setup("input a: UInt8 output b: UInt8 := a");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v = Value::Unsigned(9);
        accept_input!(eval, start, in_ref, v.clone());
        eval_stream!(eval, start, 0);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v)
    }

    #[test]
    fn test_oob_lookup() {
        let (_, mut eval, start) =
            setup("input a: UInt8\noutput b := a.offset(by: -1).defaults(to: 3)\noutput x: UInt8 @5Hz := b.hold().defaults(to: 3)");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(1);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        accept_input!(eval, start, in_ref, v1);
        eval_stream!(eval, start, 0);
        eval_stream!(eval, start, 1);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
    }

    #[test]
    fn test_output_lookup() {
        let (_, mut eval, start) = setup(
            "input a: UInt8\noutput mirror: UInt8 := a\noutput mirror_offset := mirror.offset(by: -1).defaults(to: 5)\noutput c: UInt8 @5Hz := mirror_offset.hold().defaults(to: 3)",
        );
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(2);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        accept_input!(eval, start, in_ref, v1.clone());
        eval_stream!(eval, start, 0);
        eval_stream!(eval, start, 1);
        accept_input!(eval, start, in_ref, v2);
        eval_stream!(eval, start, 0);
        eval_stream!(eval, start, 1);
        eval_stream!(eval, start, 2);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
    }

    #[test]
    fn test_conversion_if() {
        let (_, mut eval, start) =
            setup("input a: UInt8\noutput b: UInt16 := if true then a else a[-1].defaults(to: 0)");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        accept_input!(eval, start, in_ref, v1.clone());
        eval_stream!(eval, start, 0);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
    }

    #[test]
    #[ignore] // See issue #32 in LolaParser.
    fn test_conversion_lookup() {
        let (_, mut eval, start) = setup("input a: UInt8\noutput b: UInt32 := a + 100000");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let expected = Value::Unsigned(7 + 100000);
        let v1 = Value::Unsigned(7);
        accept_input!(eval, start, in_ref, v1);
        eval_stream!(eval, start, 0);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_bin_op() {
        let (_, mut eval, start) = setup("input a: UInt16\n input b: UInt16\noutput c: UInt16 := a + b");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(0);
        let a = StreamReference::InRef(0);
        let b = StreamReference::InRef(1);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        let expected = Value::Unsigned(1 + 2);
        accept_input!(eval, start, a, v1.clone());
        accept_input!(eval, start, b, v2.clone());
        eval_stream!(eval, start, 0);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_bin_op_float() {
        let (_, mut eval, start) = setup("input a: Float64\n input b: Float64\noutput c: Float64 := a + b");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(0);
        let a = StreamReference::InRef(0);
        let b = StreamReference::InRef(1);
        let v1 = Value::Float(NotNan::new(3.5f64).unwrap());
        let v2 = Value::Float(NotNan::new(39.347568f64).unwrap());
        let expected = Value::Float(NotNan::new(3.5f64 + 39.347568f64).unwrap());
        accept_input!(eval, start, a, v1.clone());
        accept_input!(eval, start, b, v2.clone());
        eval_stream!(eval, start, 0);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_bin_tuple() {
        let (_, mut eval, start) =
            setup("input a: Int32\n input b: Bool\noutput c := (a, b) output d := c.0 output e := c.1");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(0);
        let out_ref0 = StreamReference::OutRef(1);
        let out_ref1 = StreamReference::OutRef(2);
        let a = StreamReference::InRef(0);
        let b = StreamReference::InRef(1);
        let v1 = Value::Signed(1);
        let v2 = Value::Bool(true);
        let expected = Value::Tuple(Box::new([v1.clone(), v2.clone()]));
        accept_input!(eval, start, a, v1.clone());
        accept_input!(eval, start, b, v2.clone());
        eval_stream!(eval, start, 0);
        eval_stream!(eval, start, 1);
        eval_stream!(eval, start, 2);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
        assert_eq!(eval.__peek_value(out_ref0, &Vec::new(), 0).unwrap(), v1);
        assert_eq!(eval.__peek_value(out_ref1, &Vec::new(), 0).unwrap(), v2);
    }

    #[test]
    fn test_regular_lookup() {
        let (_, mut eval, start) =
            setup("input a: UInt8 output b := a.offset(by: -1).defaults(to: 5) output x: UInt8 @5Hz := b.hold().defaults(to: 3)");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(1);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(1);
        let v2 = Value::Unsigned(2);
        let v3 = Value::Unsigned(3);
        accept_input!(eval, start, in_ref, v1);
        accept_input!(eval, start, in_ref, v2.clone());
        accept_input!(eval, start, in_ref, v3);
        eval_stream!(eval, start, 0);
        eval_stream!(eval, start, 1);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v2)
    }

    #[ignore] // triggers no longer store values
    #[test]
    fn test_trigger() {
        let (_, mut eval, start) =
            setup("input a: UInt8 output b := a.offset(by: -1) output x: UInt8 @5Hz := b.hold().defaults(to: 3)\n trigger x > 4");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(1);
        let trig_ref = StreamReference::OutRef(2);
        let in_ref = StreamReference::InRef(0);
        let v1 = Value::Unsigned(8);
        eval_stream!(eval, start, 0);
        eval_stream!(eval, start, 1);
        eval_stream!(eval, start, 2);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
        assert_eq!(eval.__peek_value(trig_ref, &Vec::new(), 0).unwrap(), Value::Bool(false));
        accept_input!(eval, start, in_ref, v1.clone());
        eval_stream!(eval, start, 0);
        eval_stream!(eval, start, 1);
        eval_stream!(eval, start, 2);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), Value::Unsigned(3));
        assert_eq!(eval.__peek_value(trig_ref, &Vec::new(), 0).unwrap(), Value::Bool(false));
        accept_input!(eval, start, in_ref, Value::Unsigned(17));
        eval_stream!(eval, start, 0);
        eval_stream!(eval, start, 1);
        eval_stream!(eval, start, 2);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), v1);
        assert_eq!(eval.__peek_value(trig_ref, &Vec::new(), 0).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_sum_window() {
        let (_, mut eval, mut time) =
            setup_time("input a: Int16\noutput b: Int16 @0.25Hz := a.aggregate(over: 40s, using: sum)");
        let mut eval = eval.as_Evaluator();
        time += Duration::from_secs(45);
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let n = 25;
        for v in 1..=n {
            accept_input_timed!(eval, in_ref, Value::Signed(v), time);
            time += Duration::from_secs(1);
        }
        time += Duration::from_secs(1);
        // 71 secs have passed. All values should be within the window.
        eval_stream_timed!(eval, 0, time);
        let expected = Value::Signed((n * n + n) / 2);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_count_window() {
        let (_, mut eval, mut time) =
            setup_time("input a: UInt16\noutput b: UInt16 @0.25Hz := a.aggregate(over: 40s, using: #)");
        let mut eval = eval.as_Evaluator();
        time += Duration::from_secs(45);
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);
        let n = 25;
        for v in 1..=n {
            accept_input_timed!(eval, in_ref, Value::Unsigned(v), time);
            time += Duration::from_secs(1);
        }
        time += Duration::from_secs(1);
        // 71 secs have passed. All values should be within the window.
        eval_stream_timed!(eval, 0, time);
        let expected = Value::Unsigned(n);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_average_window() {
        let (_, mut eval, mut time) = setup_time(
            "input a: Float32\noutput b @0.25Hz := a.aggregate(over: 40s, using: average).defaults(to: -3.0)",
        );
        let mut eval = eval.as_Evaluator();
        time += Duration::from_secs(45);
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);

        // No time has passed. No values should be within the window. We should se the default value.
        eval_stream_timed!(eval, 0, time);
        let expected = Value::new_float(-3.0);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);

        let n = 25;
        for v in 1..=n {
            accept_input_timed!(eval, in_ref, Value::new_float(v as f64), time);
            time += Duration::from_secs(1);
        }
        time += Duration::from_secs(1);

        // 71 secs have passed. All values should be within the window.
        eval_stream_timed!(eval, 0, time);
        let n = n as f64;
        let expected = Value::new_float(((n * n + n) / 2.0) / 25.0);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_integral_window() {
        let (_, mut eval, mut time) = setup_time(
            "input a: Float64\noutput b: Float64 @0.25Hz := a.aggregate(over_exactly: 40s, using: integral).defaults(to: -3.0)",
        );
        let mut eval = eval.as_Evaluator();
        time += Duration::from_secs(45);
        let out_ref = StreamReference::OutRef(0);
        let in_ref = StreamReference::InRef(0);

        fn mv(f: f64) -> Value {
            Value::Float(NotNan::new(f).unwrap())
        }

        accept_input_timed!(eval, in_ref, mv(1f64), time);
        time += Duration::from_secs(2);
        accept_input_timed!(eval, in_ref, mv(5f64), time);
        // Value so far: (1+5) / 2 * 2 = 6
        time += Duration::from_secs(5);
        accept_input_timed!(eval, in_ref, mv(25f64), time);
        // Value so far: 6 + (5+25) / 2 * 5 = 6 + 75 = 81
        time += Duration::from_secs(1);
        accept_input_timed!(eval, in_ref, mv(0f64), time);
        // Value so far: 81 + (25+0) / 2 * 1 = 81 + 12.5 = 93.5
        time += Duration::from_secs(10);
        accept_input_timed!(eval, in_ref, mv(-40f64), time);
        // Value so far: 93.5 + (0+(-40)) / 2 * 10 = 93.5 - 200 = -106.5
        // Time passed: 2 + 5 + 1 + 10 = 18.

        eval_stream_timed!(eval, 0, time);

        let expected = Value::Float(NotNan::new(-106.5).unwrap());
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }

    #[test]
    fn test_window_type_count() {
        let (_, mut eval, start) = setup("input a: Int32\noutput b @ 10Hz := a.aggregate(over: 0.1s, using: count)");
        let mut eval = eval.as_Evaluator();
        let out_ref = StreamReference::OutRef(0);
        let _a = StreamReference::InRef(0);
        let expected = Value::Unsigned(0);
        eval_stream!(eval, start, 0);
        assert_eq!(eval.__peek_value(out_ref, &Vec::new(), 0).unwrap(), expected);
    }
}
