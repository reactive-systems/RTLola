use super::super::common::BuiltinType;
use super::super::naming::{Declaration, DeclarationTable};
use ast::*;
use std::collections::HashMap;
extern crate ast_node;
use super::candidates::*;
use super::TypeCheckResult;
use analysis::reporting::{Handler, LabeledSpan};
use ast_node::Span;
use ast_node::{AstNode, NodeId};

pub(crate) struct TypeChecker<'a> {
    declarations: &'a DeclarationTable<'a>,
    spec: &'a LolaSpec,
    tt: HashMap<NodeId, Candidates>,
    handler: &'a Handler,
}

impl<'a> TypeChecker<'a> {
    pub(crate) fn new(
        dt: &'a DeclarationTable,
        spec: &'a LolaSpec,
        handler: &'a Handler,
    ) -> TypeChecker<'a> {
        TypeChecker {
            declarations: dt,
            spec,
            tt: HashMap::new(),
            handler,
        }
    }

    pub(crate) fn check(&mut self) -> TypeCheckResult {
        self.check_typedeclarations();
        self.check_constants();
        self.check_inputs();
        self.check_outputs();
        self.check_triggers();

        let type_table: HashMap<NodeId, super::super::common::Type> =
            self.tt.iter().map(|(nid, c)| (*nid, c.into())).collect();

        TypeCheckResult { type_table }
    }

    fn check_triggers(&mut self) {
        for trigger in &self.spec.trigger {
            let was = self.get_candidates(&trigger.expression);
            if !was.is_logic() {
                let expected = Candidates::Concrete(BuiltinType::Bool, TimingInfo::Unknown);
                self.unexpected_type(
                    &expected,
                    &was,
                    &trigger.expression,
                    Some("Boolean required."),
                    None,
                );
            }
        }
    }

    fn check_typedeclarations(&mut self) {
        for _td in &self.spec.type_declarations {
            unimplemented!();
        }
    }

    fn check_outputs(&mut self) {
        // First register all expected types, then check each in separation.
        for output in &self.spec.outputs {
            let ti: TimingInfo = TypeChecker::extract_timing_info(output);
            let declared = self
                .cands_from_opt_type(&output.ty)
                .meet(&Candidates::Any(ti));
            self.reg_cand(*output.id(), declared);
        }
        for output in &self.spec.outputs {
            // Check whether this output is time- or event-driven.
            for param in &output.params {
                let ty = self.cands_from_opt_type(&param.ty);
                self.reg_cand(*param.id(), ty);
            }
            let was = self.get_candidates(&output.expression);
            let declared = self.get_from_tt(*output.id()).unwrap(); // Registered earlier.
            if declared.meet(&was).is_none() {
                self.unexpected_type(
                    &declared,
                    &was,
                    &output.expression,
                    Some("Does not match declared type."),
                    None,
                );
            }
            // Type is already declared, return.
        }
    }

    fn extract_timing_info(stream: &Output) -> TimingInfo {
        if stream
            .template_spec
            .as_ref()
            .and_then(|spec| spec.ext.as_ref())
            .map(|ext| ext.freq.is_some())
            .unwrap_or(false)
        {
            TimingInfo::TimeBased
        } else {
            TimingInfo::EventBased
        }
    }

    fn cands_from_opt_type(&mut self, opt_ty: &'a Option<Type>) -> Candidates {
        opt_ty
            .as_ref()
            .map(|t| self.check_explicit_type(t))
            .unwrap_or(Candidates::Any(TimingInfo::Unknown))
    }

    fn check_constants(&mut self) {
        for constant in &self.spec.constants {
            let was = Candidates::from(&constant.literal);
            let declared = self.cands_from_opt_type(&constant.ty);
            if declared.meet(&was).is_none() {
                self.unexpected_type(
                    &declared,
                    &was,
                    &constant.literal,
                    Some("Does not match declared type."),
                    None,
                );
            }
            self.reg_cand(*constant.id(), declared);
        }
    }

    fn check_inputs(&mut self) {
        for input in &self.spec.inputs {
            let cands = self.check_explicit_type(&input.ty).as_event_driven();
            self.reg_cand(*input.id(), cands);
        }
    }

    fn get_candidates(&mut self, e: &'a Expression) -> Candidates {
        match e.kind {
            ExpressionKind::Lit(ref lit) => self.reg_cand(*e.id(), Candidates::from(lit)),
            ExpressionKind::Ident(_) => {
                let cand = self.get_ty_from_decl(e);
                self.reg_cand(*e.id(), cand)
            }
            ExpressionKind::Default(ref expr, ref dft) => self.cands_from_default(e, expr, dft),
            ExpressionKind::Lookup(ref inst, ref offset, op) => {
                self.cands_from_lookup(e, inst, offset, op)
            }
            ExpressionKind::Binary(op, ref lhs, ref rhs) => self.cands_from_binary(e, op, lhs, rhs),
            ExpressionKind::Unary(operator, ref operand) => {
                self.cands_from_unary(e, operator, operand)
            }
            ExpressionKind::Ite(ref cond, ref cons, ref alt) => {
                self.cands_from_ite(e, cond, cons, alt)
            }
            ExpressionKind::ParenthesizedExpression(_, ref expr, _) => {
                let res = self.get_candidates(expr);
                self.reg_cand(*e.id(), res)
            }
            ExpressionKind::MissingExpression() => unimplemented!(),
            ExpressionKind::Tuple(ref exprs) => {
                let cands: Vec<Candidates> = exprs.iter().map(|e| self.get_candidates(e)).collect();
                self.reg_cand(*e.id(), Candidates::Tuple(cands))
            }
            ExpressionKind::Function(ref kind, ref args) => self.check_function(e, *kind, args),
        }
    }

    fn cands_from_default(
        &mut self,
        e: &'a AstNode<'a>,
        expr: &'a Expression,
        dft: &'a Expression,
    ) -> Candidates {
        let expr_ty = self.get_candidates(expr);
        let dft_ty = self.get_candidates(dft);
        let mut res = expr_ty.meet(&dft_ty);
        if res.is_none() {
            self.incompatible_types(
                e,
                "Default value and expression need to have compatible types.",
            );
            res = dft_ty
        }
        self.reg_cand(*e.id(), res)
    }

    fn cands_from_unary(
        &mut self,
        e: &'a Expression,
        operator: UnOp,
        operand: &'a Expression,
    ) -> Candidates {
        let op_type = self.get_candidates(operand);
        match operator {
            UnOp::Neg => {
                if op_type.is_numeric() {
                    op_type.clone().into_signed()
                } else {
                    let ti = op_type.timing_info().unwrap_or(TimingInfo::Unknown);
                    let expected = Candidates::Numeric(NumConfig::new_unsigned(Some(8)), ti);
                    self.unexpected_type(
                        &expected,
                        &op_type,
                        e,
                        Some("Expected numeric value."),
                        None,
                    );
                    expected.into_signed()
                }
            }
            UnOp::Not => {
                if op_type.is_logic() {
                    op_type.clone()
                } else {
                    let ti = op_type.timing_info().unwrap_or(TimingInfo::Unknown);
                    let expected = Candidates::Concrete(BuiltinType::Bool, ti);
                    self.unexpected_type(
                        &expected,
                        &op_type,
                        e,
                        Some("Expected boolean value."),
                        None,
                    );
                    expected
                }
            }
        }
    }

    fn cands_from_binary(
        &mut self,
        e: &'a Expression,
        op: BinOp,
        lhs: &'a Expression,
        rhs: &'a Expression,
    ) -> Candidates {
        let logic_type = Candidates::Concrete(BuiltinType::Bool, TimingInfo::Unknown);
        let numeric_type =
            Candidates::Numeric(NumConfig::new_unsigned(Some(8)), TimingInfo::Unknown);
        let any_type = Candidates::top();
        let (expected_lhs, expected_rhs, expected_meet) = match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow => {
                (&numeric_type, &numeric_type, &numeric_type)
            }
            BinOp::And | BinOp::Or => (&logic_type, &logic_type, &logic_type),
            BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                (&any_type, &any_type, &any_type)
            }
        };
        self.check_n_ary_fn(e, &vec![lhs, rhs], &vec![expected_lhs, expected_rhs]);
        let lhs_ty = self.retrieve_type(lhs);
        let rhs_ty = self.retrieve_type(rhs);
        let ti = self.retrieve_and_check_ti(&vec![lhs, rhs]);
        let meet_type = lhs_ty.meet(&rhs_ty);
        let error_return = match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow => {
                Candidates::Numeric(NumConfig::new_unsigned(None), ti)
            }
            BinOp::And | BinOp::Or => Candidates::Concrete(BuiltinType::Bool, ti),
            BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                Candidates::Concrete(BuiltinType::Bool, ti)
            }
        };
        let res = if meet_type.is_none() {
            let msg = format!(
                "Binary operator {} is not applicable to incompatible types {} and {}.",
                op, lhs_ty, rhs_ty
            );
            self.incompatible_types(e, msg.as_str());
            error_return
        } else {
            meet_type
        };
        self.reg_cand(*e.id(), res)
    }

    fn cands_from_ite(
        &mut self,
        e: &'a Expression,
        cond_ex: &'a Expression,
        cons_ex: &'a Expression,
        alt_ex: &'a Expression,
    ) -> Candidates {
        let cond = self.get_candidates(cond_ex);
        let cons = self.get_candidates(cons_ex);
        let alt = self.get_candidates(alt_ex);
        self.retrieve_and_check_ti(&vec![cons_ex, alt_ex]);
        if !cond.is_logic() {
            let expected = Candidates::Concrete(BuiltinType::Bool, TimingInfo::Unknown);
            self.unexpected_type(
                &expected,
                &cond,
                cond_ex,
                Some("Boolean expected"),
                Some("Condition of an if expression needs to be boolean."),
            );
        }
        let res_type = cons.meet(&alt);
        if res_type.is_none() {
            self.unexpected_type(
                &cons,
                &alt,
                e,
                Some("All arms of an if expression need to be compatible."),
                Some("Arms have incompatible types."),
            );
            Candidates::top()
        } else {
            self.reg_cand(*e.id(), res_type)
        }
    }

    fn cands_from_lookup(
        &mut self,
        e: &'a Expression,
        inst: &'a StreamInstance,
        offset: &'a Offset,
        op: Option<WindowOperation>,
    ) -> Candidates {
        self.check_offset(offset, e); // Return value does not matter.
        let target_stream_type = self.check_stream_instance(inst);
        if op.is_none() {
            return self.reg_cand(*e.id(), target_stream_type);
        }
        let op = op.unwrap();
        // For all window operations, the source stream needs to be numeric.
        if !target_stream_type.is_numeric() {
            let expected =
                Candidates::Numeric(NumConfig::new_unsigned(Some(8)), TimingInfo::Unknown);
            self.unexpected_type(
                &expected,
                &target_stream_type,
                inst,
                Some("Numeric type required."),
                Some("Window operations require a numeric source type."),
            );
        }
        self.reg_cand(
            *e.id(),
            match op {
                WindowOperation::Sum | WindowOperation::Product => {
                    target_stream_type.as_time_driven()
                }
                WindowOperation::Average | WindowOperation::Integral => {
                    Candidates::Numeric(NumConfig::new_float(None), TimingInfo::TimeBased)
                }
                WindowOperation::Count => {
                    Candidates::Numeric(NumConfig::new_unsigned(None), TimingInfo::TimeBased)
                }
            },
        )
    }

    fn check_function(
        &mut self,
        e: &'a Expression,
        kind: FunctionKind,
        args: &'a [Box<Expression>],
    ) -> Candidates {
        // TODO: Remove magic number, more info on timing requirement.
        let numeric_type =
            Candidates::Numeric(NumConfig::new_unsigned(Some(8)), TimingInfo::Unknown);
        let integer_type = Candidates::Numeric(NumConfig::new_signed(Some(8)), TimingInfo::Unknown);
        let float_type = Candidates::Numeric(NumConfig::new_float(Some(8)), TimingInfo::Unknown);
        let args: Vec<&Expression> = args.iter().map(|a| a.as_ref()).collect();
        match kind {
            FunctionKind::NthRoot => {
                self.check_n_ary_fn(e, &args, &vec![&integer_type, &numeric_type]);
                let ti = self.retrieve_and_check_ti(&args);
                self.reg_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None), ti))
            }
            FunctionKind::Sqrt => {
                self.check_n_ary_fn(e, &args, &vec![&numeric_type]);
                let ti = self.retrieve_and_check_ti(&args);
                self.reg_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None), ti))
            }
            FunctionKind::Projection => {
                let cands: Vec<Candidates> = args.iter().map(|a| self.get_candidates(a)).collect();
                if cands.len() < 2 {
                    self.unexpected_number_of_arguments(2, cands.len() as u8, e);
                }
                if !cands[0].is_unsigned() {
                    let expected = Candidates::Concrete(BuiltinType::UInt(8), TimingInfo::Unknown);
                    self.unexpected_type(
                        &expected,
                        &cands[0],
                        e,
                        Some("Unsigned integer required."),
                        Some("Tuple projections require an unsigned integer as first argument."),
                    );
                }
                match cands[1] {
                    Candidates::Tuple(ref v) => {
                        match TypeChecker::convert_to_constant(args[0]) {
                            Some(n) if n < v.len() =>
                            // No way to recover from here.
                            {
                                let msg = format!(
                                    "Cannot access element {} in a tuple of length {}.",
                                    n,
                                    v.len()
                                );
                                let label = format!("Needs to be between 1 and {}.", v.len());
                                self.invalid_argument(args[0], msg.as_str(), label.as_str());
                                Candidates::top()
                            }
                            Some(n) => self.reg_cand(*e.id(), cands[n].clone()),
                            None => {
                                self.constant_value_required(args[0]);
                                Candidates::top()
                            }
                        }
                    }
                    _ => {
                        let expected = &Candidates::Tuple(Vec::new());
                        self.unexpected_type(
                            &expected,
                            &cands[1],
                            args[1],
                            Some("Tuple required."),
                            Some("Tuple projections require a tuple as second argument."),
                        );
                        Candidates::top()
                    }
                }
            }
            FunctionKind::Sin
            | FunctionKind::Cos
            | FunctionKind::Tan
            | FunctionKind::Arcsin
            | FunctionKind::Arccos
            | FunctionKind::Arctan => {
                self.check_n_ary_fn(e, &args, &vec![&numeric_type]);
                let ti = self.retrieve_and_check_ti(&args);
                self.reg_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None), ti))
            }
            FunctionKind::Exp => {
                self.check_n_ary_fn(e, &args, &vec![&numeric_type]);
                let ti = self.retrieve_and_check_ti(&args);
                self.reg_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None), ti))
            }
            FunctionKind::Floor | FunctionKind::Ceil => {
                self.check_n_ary_fn(e, &args, &vec![&float_type]);
                let ti = self.retrieve_and_check_ti(&args);
                // TODO: This makes little sense.
                let width = self.retrieve_type(args[0]).width();
                self.reg_cand(
                    *e.id(),
                    Candidates::Numeric(NumConfig::new_signed(width), ti),
                )
            }
        }
    }

    fn check_n_ary_fn(
        &mut self,
        call: &'a AstNode<'a>,
        args: &[&'a Expression],
        expected: &[&Candidates],
    ) {
        let was: Vec<Candidates> = args.iter().map(|a| self.get_candidates(a)).collect();
        if was.len() != expected.len() {
            self.unexpected_number_of_arguments(expected.len() as u8, was.len() as u8, call);
        } else {
            for ((expected, was), arg) in expected.iter().zip(was).zip(args) {
                if expected.meet(&was).is_none() {
                    // Register error and mask erroneous value in TypeTable.
                    self.unexpected_type(expected, &was, *arg, None, None);
                    let res: Candidates = (*expected).clone();
                    self.override_cand(*arg.id(), res);
                }
            }
        }
    }

    fn convert_to_constant(e: &'a Expression) -> Option<usize> {
        match e.kind {
            ExpressionKind::Lit(ref lit) => match lit.kind {
                LitKind::Int(i) => Some(i as usize),
                _ => unimplemented!(),
            },
            ExpressionKind::Ident(_) => unimplemented!(), // Here, we should check the declaration. If it is a constant -> fine.
            _ => unimplemented!(),
        }
    }

    fn check_offset(&mut self, offset: &'a Offset, origin: &'a Expression) {
        let (expr, is_discrete) = match offset {
            Offset::DiscreteOffset(e) => (e, true),
            Offset::RealTimeOffset(e, _) => (e, false),
        };

        let cand = self.get_candidates(expr);

        match (is_discrete, cand.is_numeric(), cand.is_integer()) {
            (true, _, true) | (false, true, _) => {} // fine
            (true, _, false) => {
                // TODO: weaken width requirement, strengthen timing requirement.
                let expected = Candidates::Concrete(BuiltinType::Int(8), TimingInfo::Unknown);
                self.unexpected_type(
                    &expected,
                    &cand,
                    origin,
                    Some("Integer required."),
                    Some("Discrete offsets require an integer offset."),
                );
            }
            (false, false, _) => {
                // TODO: strengthen timing requirement.
                let expected =
                    Candidates::Numeric(NumConfig::new_unsigned(Some(8)), TimingInfo::Unknown);
                self.unexpected_type(
                    &expected,
                    &cand,
                    origin,
                    Some("Numeric value required."),
                    Some("Real-time offset require a numeric offset."),
                );
            }
        }
    }

    fn check_stream_instance(&mut self, inst: &'a StreamInstance) -> Candidates {
        let result_type = self.get_ty_from_decl(inst);
        match self.declarations.get(inst.id()) {
            Some(Declaration::Out(ref o)) => {
                if o.params.len() != inst.arguments.len() {
                    self.unexpected_number_of_arguments(
                        o.params.len() as u8,
                        inst.arguments.len() as u8,
                        inst,
                    );
                } else {
                    // Check the parameter/argument pairs.
                    // We can only perform these checks of there is the correct number of arguments. Otherwise skip until correct number of arguments is provided.
                    let params: Vec<Candidates> =
                        o.params.iter().map(|p| self.retrieve_type(p)).collect();

                    let arguments: Vec<Candidates> = inst
                        .arguments
                        .iter()
                        .map(|e| self.get_candidates(e))
                        .collect();

                    if arguments.len() != inst.arguments.len() {
                        self.report_bug("Parameters and arguments should already be checked.")
                    }

                    for (i, (exp, was)) in params.iter().zip(arguments).enumerate() {
                        if exp.is_none() || was.is_none() {
                            continue; // Already reported in `get_cands`.
                        }
                        let res = exp.meet(&was);
                        if res.is_none() {
                            self.unexpected_type(exp, &was, inst.arguments[i].as_ref(), None, None);
                        }
                    }
                }
                // Independent of all checks, pretend everything worked fine.
                result_type
            }
            Some(Declaration::In(_)) => result_type, // Nothing to check for inputs.
            _ => {
                self.unknown_identifier(inst);
                Candidates::top() // Pretend everything's fine, but we have no information whatsoever.
            }
        }
    }

    fn check_explicit_type(&mut self, ty: &'a Type) -> Candidates {
        match ty.kind {
            TypeKind::Tuple(ref v) => {
                let vals = v.iter().map(|t| self.check_explicit_type(t)).collect();
                let res = Candidates::Tuple(vals);
                self.reg_cand(*ty.id(), res)
            }
            TypeKind::Malformed(_) => unreachable!(),
            TypeKind::Simple(_) => {
                let res = self.get_ty_from_decl(ty);
                self.reg_cand(*ty.id(), res)
            }
        }
    }

    fn get_ty_from_decl(&mut self, node: &'a AstNode<'a>) -> Candidates {
        match self.declarations.get(node.id()) {
            Some(Declaration::Const(ref c)) => {
                let cand = self.get_from_tt(*c.id());
                assert!(cand.is_some()); // Since we added all declarations in the beginning, this value needs to be present.
                cand.unwrap()
            }
            Some(Declaration::In(ref i)) => {
                let cand = self.get_from_tt(*i.id());
                assert!(cand.is_some()); // Since we added all declarations in the beginning, this value needs to be present.
                cand.unwrap()
            }
            Some(Declaration::Out(ref o)) => {
                let cand = self.get_from_tt(*o.id());
                assert!(cand.is_some()); // Since we added all declarations in the beginning, this value needs to be present.
                cand.unwrap()
            }
            Some(Declaration::UserDefinedType(_td)) => unimplemented!(),
            Some(Declaration::BuiltinType(ref b)) => Candidates::from(b),
            Some(Declaration::Param(ref p)) => self.cands_from_opt_type(&p.ty),
            None => {
                self.unknown_identifier(node);
                Candidates::bot()
            }
        }
    }

    fn retrieve_and_check_ti(&mut self, v: &[&'a Expression]) -> TimingInfo {
        let mut accu = TimingInfo::Unknown;
        let tis: Vec<Option<TimingInfo>> = v.iter().map(|e| self.retrieve_ti(*e)).collect();
        for (i, ti) in tis.iter().enumerate() {
            if let Some(ti) = ti {
                if let Some(new_ti) = TimingInfo::meet(accu, *ti) {
                    accu = new_ti;
                } else {
                    let span = Span {
                        start: v[0].span().start,
                        end: v[i].span().end,
                    };
                    self.incompatible_timing(accu, *ti, span);
                    return TimingInfo::Unknown;
                }
            } // Otherwise ignore, error already reported.
        }
        accu
    }

    fn unknown_identifier(&mut self, node: &'a AstNode<'a>) {
        let span = LabeledSpan::new(*node.span(), "Identifier unknown.", true);
        self.handler.bug_with_span(
            "Found unknown identifier. This must not happen after the naming analysis.",
            span,
        )
    }

    fn report_bug(&mut self, msg: &str) {
        self.handler.error(msg)
    }

    fn invalid_argument(&mut self, arg: &'a AstNode<'a>, msg: &str, label: &str) {
        let span = LabeledSpan::new(*arg.span(), label, true);
        self.handler.error_with_span(msg, span);
    }

    fn constant_value_required(&mut self, expr: &'a AstNode<'a>) {
        let span = LabeledSpan::new(*expr.span(), "Unknown at compile time.", true);
        self.handler
            .error_with_span("Value cannot be determined statically.", span);
    }

    fn unexpected_type(
        &mut self,
        expected: &Candidates,
        was: &Candidates,
        expr: &'a AstNode<'a>,
        label: Option<&str>,
        msg: Option<&str>,
    ) {
        let label_dft = format!("Expected {}.", expected);
        let span = LabeledSpan::new(*expr.span(), label.unwrap_or(label_dft.as_str()), true);
        let msg_dft = format!("Expected type {} but got {}.", expected, was);
        self.handler
            .error_with_span(msg.unwrap_or(msg_dft.as_str()), span);
    }

    fn unexpected_number_of_arguments(&mut self, expected: u8, was: u8, expr: &'a AstNode<'a>) {
        let span = LabeledSpan::new(*expr.span(), "Identifier not declared.", true);
        self.handler.error_with_span(
            format!("Expected {} argument but got {}.", expected, was).as_str(),
            span,
        )
    }

    fn incompatible_types(&mut self, expr: &'a AstNode<'a>, msg: &str) {
        let span = LabeledSpan::new(*expr.span(), "Incompatible types.", true);
        self.handler.error_with_span(msg, span);
    }

    fn value_not_present(&mut self, node: &'a AstNode<'a>) {
        let span = LabeledSpan::new(*node.span(), "Should have been reported before.", true);
        self.handler.bug_with_span(
            "Expected type to be computed already, but it was not.",
            span,
        );
    }

    fn incompatible_timing(&mut self, left: TimingInfo, right: TimingInfo, span: Span) {
        //        format!("Arguments {:?} have incompatible timing.", v),
        unimplemented!()
    }

    fn retrieve_ti(&mut self, node: &'a AstNode<'a>) -> Option<TimingInfo> {
        self.retrieve_type(node).timing_info()
    }

    fn retrieve_type(&mut self, node: &'a AstNode<'a>) -> Candidates {
        if let Some(c) = self.get_from_tt(*node.id()) {
            c
        } else {
            self.value_not_present(node);
            Candidates::top()
        }
    }

    fn get_from_tt(&self, nid: NodeId) -> Option<Candidates> {
        self.tt.get(&nid).cloned()
    }

    fn override_cand(&mut self, nid: NodeId, cand: Candidates) -> Candidates {
        assert!(self.tt.insert(nid, cand.clone()).is_some());
        cand
    }

    fn reg_cand(&mut self, nid: NodeId, cand: Candidates) -> Candidates {
        let res = match self.tt.get(&nid) {
            Some(_c) => panic!("Without type inference, we cannot concretize the type of a node."),
            None => cand,
        };
        self.tt.insert(nid, res.clone());
        res
    }
}
