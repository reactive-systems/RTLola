use super::super::common::BuiltinType;
use super::super::naming::{Declaration, DeclarationTable};
use super::candidates::*;
use super::TypeCheckResult;
use crate::ast::*;
use crate::parse::Ident;
use crate::reporting::{Handler, LabeledSpan};
use ast_node;
use ast_node::Span;
use ast_node::{AstNode, NodeId};
use std::collections::HashMap;

pub(crate) struct TypeChecker<'a> {
    declarations: &'a DeclarationTable<'a>,
    spec: &'a LolaSpec,
    tt: HashMap<NodeId, Candidates>,
    handler: &'a Handler,
}

/// Implementation for the Type Checker.
///
/// Note that functions with a `retrieve_` prefix generally do not change the state of the checker
/// except when encountering an error that needs to be reported.
/// Functions with a `check_` prefix check their arguments, report errors and register types they
/// discovered.
/// Functions with an `extract_` prefix are static and use only the information passed to them. They
/// do not report errors, nor register types.
impl<'a> TypeChecker<'a> {
    pub(crate) fn new(
        dt: &'a DeclarationTable<'_>,
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

    pub(crate) fn check_spec(&mut self) -> TypeCheckResult {
        self.check_typedeclarations();
        self.check_constants();
        self.check_inputs();
        self.check_outputs();
        self.check_triggers();

        let type_table: HashMap<NodeId, super::ExtendedType> =
            self.tt.iter().map(|(nid, c)| (*nid, c.into())).collect();

        TypeCheckResult { type_table }
    }

    fn check_typedeclarations(&mut self) {
        for _td in &self.spec.type_declarations {
            unimplemented!();
        }
    }

    fn check_constants(&mut self) {
        for constant in &self.spec.constants {
            let was = Candidates::from(&constant.literal);
            let declared = self.check_optional_type(&constant.ty);
            if declared.meet(&was).is_none() {
                self.report_unexpected_type(
                    &declared,
                    &was,
                    &constant.literal,
                    Some("Does not match declared type."),
                    None,
                );
            }
            self.register_cand(*constant.id(), declared);
        }
    }

    fn check_inputs(&mut self) {
        for input in &self.spec.inputs {
            let cands = self.check_ast_type(&input.ty).as_event_driven();
            self.register_cand(*input.id(), cands);
        }
    }

    fn check_outputs(&mut self) {
        // First register all expected types, then check each in separation.
        for output in &self.spec.outputs {
            let ti: TimingInfo = self.extract_timing_info(output);
            let declared = self
                .check_optional_type(&output.ty)
                .meet(&Candidates::Any(ti));
            self.register_cand(*output.id(), declared);
        }
        for output in &self.spec.outputs {
            // Check whether this output is time- or event-driven.
            for param in &output.params {
                let ty = self.check_optional_type(&param.ty);
                self.register_cand(*param.id(), ty);
            }
            let was = self.check_expression(&output.expression);
            let declared = self.retrieve_from_tt(*output.id()).unwrap(); // Registered earlier.
            if declared.meet(&was).is_none() {
                self.report_unexpected_type(
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

    fn check_triggers(&mut self) {
        for trigger in &self.spec.trigger {
            let was = self.check_expression(&trigger.expression);
            if !was.is_logic() {
                let expected = Candidates::Concrete(BuiltinType::Bool, TimingInfo::Unknown);
                self.report_unexpected_type(
                    &expected,
                    &was,
                    &trigger.expression,
                    Some("Boolean required."),
                    None,
                );
            }
        }
    }

    fn check_expression(&mut self, e: &'a Expression) -> Candidates {
        match e.kind {
            ExpressionKind::Lit(ref lit) => self.register_cand(*e.id(), Candidates::from(lit)),
            ExpressionKind::Ident(_) => {
                let cand = self.retrieve_from_declaration(e);
                self.register_cand(*e.id(), cand)
            }
            ExpressionKind::Default(ref expr, ref dft) => self.check_default(e, expr, dft),
            ExpressionKind::Lookup(ref inst, ref offset, op) => {
                self.check_lookup(e, inst, offset, op)
            }
            ExpressionKind::Binary(op, ref lhs, ref rhs) => self.check_binary(e, op, lhs, rhs),
            ExpressionKind::Unary(operator, ref operand) => self.check_unary(e, operator, operand),
            ExpressionKind::Ite(ref cond, ref cons, ref alt) => self.check_ite(e, cond, cons, alt),
            ExpressionKind::ParenthesizedExpression(_, ref expr, _) => {
                let res = self.check_expression(expr);
                self.register_cand(*e.id(), res)
            }
            ExpressionKind::MissingExpression() => unimplemented!(),
            ExpressionKind::Tuple(ref exprs) => {
                let cands: Vec<Candidates> =
                    exprs.iter().map(|e| self.check_expression(e)).collect();
                self.register_cand(*e.id(), Candidates::Tuple(cands))
            }
            ExpressionKind::Function(ref kind, ref args) => {
                unimplemented!();
                //self.check_function(e, *kind, args)
            }
            ExpressionKind::Field(ref expr, ref ident) => self.check_field_access(e, expr, ident),
            ExpressionKind::Method(_, _, _) => unimplemented!(),
        }
    }

    fn check_default(
        &mut self,
        e: &'a dyn AstNode<'a>,
        expr: &'a Expression,
        dft: &'a Expression,
    ) -> Candidates {
        let expr_ty = self.check_expression(expr);
        let dft_ty = self.check_expression(dft);
        let mut res = expr_ty.meet(&dft_ty);
        if res.is_none() {
            self.report_incompatible_types(
                e,
                "Default value and expression need to have compatible types.",
            );
            res = dft_ty
        }
        self.register_cand(*e.id(), res)
    }

    fn check_unary(
        &mut self,
        e: &'a Expression,
        operator: UnOp,
        operand: &'a Expression,
    ) -> Candidates {
        let op_type = self.check_expression(operand);
        match operator {
            UnOp::Neg => {
                if op_type.is_numeric() {
                    op_type.clone().into_signed()
                } else {
                    let ti = op_type.timing_info().unwrap_or(TimingInfo::Unknown);
                    let expected = Candidates::Numeric(NumConfig::new_unsigned(Some(8)), ti);
                    self.report_unexpected_type(
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
                    self.report_unexpected_type(
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

    fn check_binary(
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
        let (expected_lhs, expected_rhs) = match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow => {
                (&numeric_type, &numeric_type)
            }
            BinOp::And | BinOp::Or => (&logic_type, &logic_type),
            BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                (&any_type, &any_type)
            }
        };
        self.check_n_ary_fn(e, &[lhs, rhs], &[expected_lhs, expected_rhs]);
        let lhs_ty = self.retrieve_type(lhs);
        let rhs_ty = self.retrieve_type(rhs);
        let ti = self.retrieve_and_check_ti(&[lhs, rhs]);
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
            self.report_incompatible_types(e, msg.as_str());
            error_return
        } else {
            meet_type
        };
        self.register_cand(*e.id(), res)
    }

    fn check_ite(
        &mut self,
        e: &'a Expression,
        cond_ex: &'a Expression,
        cons_ex: &'a Expression,
        alt_ex: &'a Expression,
    ) -> Candidates {
        let cond = self.check_expression(cond_ex);
        let cons = self.check_expression(cons_ex);
        let alt = self.check_expression(alt_ex);
        self.retrieve_and_check_ti(&[cons_ex, alt_ex]);
        if !cond.is_logic() {
            let expected = Candidates::Concrete(BuiltinType::Bool, TimingInfo::Unknown);
            self.report_unexpected_type(
                &expected,
                &cond,
                cond_ex,
                Some("Boolean expected"),
                Some("Condition of an if expression needs to be boolean."),
            );
        }
        let res_type = cons.meet(&alt);
        if res_type.is_none() {
            self.report_unexpected_type(
                &cons,
                &alt,
                e,
                Some("All arms of an if expression need to be compatible."),
                Some("Arms have incompatible types."),
            );
            Candidates::top()
        } else {
            self.register_cand(*e.id(), res_type)
        }
    }

    fn check_lookup(
        &mut self,
        e: &'a Expression,
        inst: &'a StreamInstance,
        offset: &'a Offset,
        op: Option<WindowOperation>,
    ) -> Candidates {
        self.check_offset(offset, e); // Return value does not matter.
        let target_stream_type = self.check_stream_instance(inst);
        if op.is_none() {
            return self.register_cand(*e.id(), target_stream_type);
        }
        let op = op.unwrap();
        // For all window operations, the source stream needs to be numeric.
        if !target_stream_type.is_numeric() {
            let expected =
                Candidates::Numeric(NumConfig::new_unsigned(Some(8)), TimingInfo::Unknown);
            self.report_unexpected_type(
                &expected,
                &target_stream_type,
                inst,
                Some("Numeric type required."),
                Some("Window operations require a numeric source type."),
            );
        }
        self.register_cand(
            *e.id(),
            match op {
                WindowOperation::Sum | WindowOperation::Product => {
                    target_stream_type.as_time_driven(None)
                }
                WindowOperation::Average | WindowOperation::Integral => {
                    // TODO: We know it's time based, pass duration down.
                    Candidates::Numeric(NumConfig::new_float(None), TimingInfo::Unknown)
                }
                WindowOperation::Count => {
                    // TODO: We know it's time based, pass duration down.
                    Candidates::Numeric(NumConfig::new_unsigned(None), TimingInfo::Unknown)
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
                self.check_n_ary_fn(e, &args, &[&integer_type, &numeric_type]);
                let ti = self.retrieve_and_check_ti(&args);
                self.register_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None), ti))
            }
            FunctionKind::Sqrt => {
                self.check_n_ary_fn(e, &args, &[&numeric_type]);
                let ti = self.retrieve_and_check_ti(&args);
                self.register_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None), ti))
            }
            FunctionKind::Sin
            | FunctionKind::Cos
            | FunctionKind::Tan
            | FunctionKind::Arcsin
            | FunctionKind::Arccos
            | FunctionKind::Arctan => {
                self.check_n_ary_fn(e, &args, &[&numeric_type]);
                let ti = self.retrieve_and_check_ti(&args);
                self.register_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None), ti))
            }
            FunctionKind::Exp => {
                self.check_n_ary_fn(e, &args, &[&numeric_type]);
                let ti = self.retrieve_and_check_ti(&args);
                self.register_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None), ti))
            }
            FunctionKind::Floor | FunctionKind::Ceil => {
                self.check_n_ary_fn(e, &args, &[&float_type]);
                let ti = self.retrieve_and_check_ti(&args);
                // TODO: This makes little sense.
                let width = self.retrieve_type(args[0]).width();
                self.register_cand(
                    *e.id(),
                    Candidates::Numeric(NumConfig::new_signed(width), ti),
                )
            }
        }
    }

    fn check_field_access(
        &mut self,
        e: &'a Expression,
        expr: &'a Expression,
        ident: &'a Ident,
    ) -> Candidates {
        let u = ident
            .name
            .parse::<usize>()
            .expect("field is an unsigned integer");
        // check type of expression
        match self.check_expression(expr) {
            Candidates::Tuple(ref v) => {
                if u >= v.len() {
                    // No way to recover from here.
                    let msg = format!(
                        "Cannot access element {} in a tuple of length {}.",
                        u,
                        v.len()
                    );
                    let label = format!("Needs to be between 0 and {}.", v.len() - 1);
                    self.handler
                        .error_with_span(&msg, LabeledSpan::new(ident.span, &label, true));
                    Candidates::top()
                } else {
                    self.register_cand(*e.id(), v[u].clone())
                }
            }
            was => {
                let expected = &Candidates::Tuple(Vec::new());
                self.report_unexpected_type(
                    &expected,
                    &was,
                    expr,
                    Some("Tuple required."),
                    Some("Tuple projections require a tuple as second argument."),
                );
                Candidates::top()
            }
        }
    }

    fn check_offset(&mut self, offset: &'a Offset, origin: &'a Expression) {
        let (expr, is_discrete) = match offset {
            Offset::DiscreteOffset(e) => (e, true),
            Offset::RealTimeOffset(e, _) => (e, false),
        };

        let cand = self.check_expression(expr);

        match (is_discrete, cand.is_numeric(), cand.is_integer()) {
            (true, _, true) | (false, true, _) => {} // fine
            (true, _, false) => {
                // TODO: weaken width requirement, strengthen timing requirement.
                let expected = Candidates::Concrete(BuiltinType::Int(8), TimingInfo::Unknown);
                self.report_unexpected_type(
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
                self.report_unexpected_type(
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
        let result_type = self.retrieve_from_declaration(inst);
        match self.declarations.get(inst.id()) {
            Some(Declaration::Out(ref o)) => {
                if o.params.len() != inst.arguments.len() {
                    self.report_wrong_num_of_args(
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
                        .map(|e| self.check_expression(e))
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
                            self.report_unexpected_type(
                                exp,
                                &was,
                                inst.arguments[i].as_ref(),
                                None,
                                None,
                            );
                        }
                    }
                }
                // Independent of all checks, pretend everything worked fine.
                result_type
            }
            Some(Declaration::In(_)) => result_type, // Nothing to check for inputs.
            _ => {
                self.report_unknown_identifier(inst);
                Candidates::top() // Pretend everything's fine, but we have no information whatsoever.
            }
        }
    }

    fn check_n_ary_fn(
        &mut self,
        call: &'a dyn AstNode<'a>,
        args: &[&'a Expression],
        expected: &[&Candidates],
    ) {
        let was: Vec<Candidates> = args.iter().map(|a| self.check_expression(a)).collect();
        if was.len() != expected.len() {
            self.report_wrong_num_of_args(expected.len() as u8, was.len() as u8, call);
        } else {
            for ((expected, was), arg) in expected.iter().zip(was).zip(args) {
                if expected.meet(&was).is_none() {
                    // Register error and mask erroneous value in TypeTable.
                    self.report_unexpected_type(expected, &was, *arg, None, None);
                    let res: Candidates = (*expected).clone();
                    self.override_cand(*arg.id(), res);
                }
            }
        }
    }

    fn check_ast_type(&mut self, ty: &'a Type) -> Candidates {
        match ty.kind {
            TypeKind::Tuple(ref v) => {
                let vals = v.iter().map(|t| self.check_ast_type(t)).collect();
                let res = Candidates::Tuple(vals);
                self.register_cand(*ty.id(), res)
            }
            TypeKind::Malformed(_) => unreachable!(),
            TypeKind::Simple(_) => {
                let res = self.retrieve_from_declaration(ty);
                self.register_cand(*ty.id(), res)
            }
            TypeKind::Inferred => unreachable!(),
        }
    }

    fn check_optional_type(&mut self, opt_ty: &'a Option<Type>) -> Candidates {
        opt_ty
            .as_ref()
            .map(|t| self.check_ast_type(t))
            .unwrap_or(Candidates::Any(TimingInfo::Unknown))
    }

    fn retrieve_from_declaration(&mut self, node: &'a dyn AstNode<'a>) -> Candidates {
        let mut cand = match self.declarations.get(node.id()) {
            Some(Declaration::Const(ref c)) => self.retrieve_from_tt(*c.id()),
            Some(Declaration::In(ref i)) => self.retrieve_from_tt(*i.id()),
            Some(Declaration::Out(ref o)) => self.retrieve_from_tt(*o.id()),
            //Some(Declaration::UserDefinedType(_td)) => unimplemented!(),
            //Some(Declaration::BuiltinType(ref b)) => Some(Candidates::from(b)),
            Some(Declaration::Param(ref p)) => self.retrieve_from_tt(*p.id()),
            Some(_) => unimplemented!(),
            None => None,
        };
        if cand.is_none() {
            self.report_unknown_identifier(node);
            cand = Some(Candidates::top()) // No information available, pretend it's fine.
        }
        cand.unwrap()
    }

    // TODO: function with mixed semantics: candidates are retrieved, TI is checked. Split.
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
                    self.report_incompatible_timing(accu, *ti, span);
                    return TimingInfo::Unknown;
                }
            } // Otherwise ignore, error already reported.
        }
        accu
    }

    fn retrieve_ti(&mut self, node: &'a dyn AstNode<'a>) -> Option<TimingInfo> {
        self.retrieve_type(node).timing_info()
    }

    fn retrieve_type(&mut self, node: &'a dyn AstNode<'a>) -> Candidates {
        if let Some(c) = self.retrieve_from_tt(*node.id()) {
            c
        } else {
            self.report_value_not_present(node);
            Candidates::top()
        }
    }

    fn retrieve_from_tt(&self, nid: NodeId) -> Option<Candidates> {
        self.tt.get(&nid).cloned()
    }

    fn override_cand(&mut self, nid: NodeId, cand: Candidates) -> Candidates {
        assert!(self.tt.insert(nid, cand.clone()).is_some());
        cand
    }

    fn register_cand(&mut self, nid: NodeId, cand: Candidates) -> Candidates {
        let res = match self.tt.get(&nid) {
            Some(_c) => panic!("Without type inference, we cannot concretize the type of a node."),
            None => cand,
        };
        self.tt.insert(nid, res.clone());
        res
    }

    fn report_unknown_identifier(&mut self, node: &'a dyn AstNode<'a>) {
        let span = LabeledSpan::new(*node.span(), "Identifier unknown.", true);
        self.handler.bug_with_span(
            "Found unknown identifier. This must not happen after the naming analysis.",
            span,
        )
    }

    fn report_bug(&mut self, msg: &str) {
        self.handler.error(msg)
    }

    fn report_invalid_argument(&mut self, arg: &'a dyn AstNode<'a>, msg: &str, label: &str) {
        let span = LabeledSpan::new(*arg.span(), label, true);
        self.handler.error_with_span(msg, span);
    }

    fn report_not_constant(&mut self, expr: &'a dyn AstNode<'a>) {
        let span = LabeledSpan::new(*expr.span(), "Unknown at compile time.", true);
        self.handler
            .error_with_span("Value cannot be determined statically.", span);
    }

    fn report_unexpected_type(
        &mut self,
        expected: &Candidates,
        was: &Candidates,
        expr: &'a dyn AstNode<'a>,
        label: Option<&str>,
        msg: Option<&str>,
    ) {
        let label_dft = format!("Expected {}.", expected);
        let span = LabeledSpan::new(
            *expr.span(),
            label.unwrap_or_else(|| label_dft.as_str()),
            true,
        );
        let msg_dft = format!("Expected type {} but got {}.", expected, was);
        self.handler
            .error_with_span(msg.unwrap_or_else(|| msg_dft.as_str()), span);
    }

    fn report_wrong_num_of_args(&mut self, expected: u8, was: u8, expr: &'a dyn AstNode<'a>) {
        let span = LabeledSpan::new(*expr.span(), "Identifier not declared.", true);
        self.handler.error_with_span(
            format!("Expected {} argument but got {}.", expected, was).as_str(),
            span,
        )
    }

    fn report_incompatible_types(&mut self, expr: &'a dyn AstNode<'a>, msg: &str) {
        let span = LabeledSpan::new(*expr.span(), "Incompatible types.", true);
        self.handler.error_with_span(msg, span);
    }

    fn report_value_not_present(&mut self, node: &'a dyn AstNode<'a>) {
        let span = LabeledSpan::new(*node.span(), "Should have been reported before.", true);
        self.handler.bug_with_span(
            "Expected type to be computed already, but it was not.",
            span,
        );
    }

    fn report_incompatible_timing(&mut self, left: TimingInfo, right: TimingInfo, span: Span) {
        let span = LabeledSpan::new(span, "Timing incompatible.", true);
        let msg = format!("Incompatible: {} and {} are conflicting.", left, right);
        self.handler.error_with_span(msg.as_str(), span);
    }

    fn extract_timing_info(&self, stream: &Output) -> TimingInfo {
        let rate = stream
            .template_spec
            .as_ref()
            .and_then(|spec| spec.ext.as_ref())
            .and_then(|e| e.freq.as_ref())
            .map(|e| self.extract_extend_rate(e));
        if let Some(rate) = rate {
            TimingInfo::TimeBased(rate)
        } else {
            TimingInfo::EventBased
        }
    }

    fn extract_extend_rate(&self, rate: &crate::ast::ExtendRate) -> std::time::Duration {
        use crate::ast::{ExtendRate, FreqUnit, LitKind, TimeUnit};
        let (expr, factor) = match rate {
            ExtendRate::Duration(expr, unit) => {
                (
                    expr,
                    match unit {
                        TimeUnit::NanoSecond => 1u64,
                        TimeUnit::MicroSecond => 10u64.pow(3),
                        TimeUnit::MilliSecond => 10u64.pow(6),
                        TimeUnit::Second => 10u64.pow(9),
                        TimeUnit::Minute => 10u64.pow(9) * 60,
                        TimeUnit::Hour => 10u64.pow(9) * 60 * 60,
                        TimeUnit::Day => 10u64.pow(9) * 60 * 60 * 24,
                        TimeUnit::Week => 10u64.pow(9) * 60 * 24 * 24 * 7,
                        TimeUnit::Year => 10u64.pow(9) * 60 * 24 * 24 * 7 * 365, // fits in u57
                    },
                )
            }
            ExtendRate::Frequency(expr, unit) => {
                (
                    expr,
                    match unit {
                        FreqUnit::MicroHertz => 10u64.pow(15), // fits in u50,
                        FreqUnit::MilliHertz => 10u64.pow(12),
                        FreqUnit::Hertz => 10u64.pow(9),
                        FreqUnit::KiloHertz => 10u64.pow(6),
                        FreqUnit::MegaHertz => 10u64.pow(3),
                        FreqUnit::GigaHertz => 1u64,
                    },
                )
            }
        };
        match crate::analysis::common::extract_constant_numeric(expr.as_ref(), &self.declarations) {
            Some(LitKind::Int(i)) => {
                // TODO: Improve: Robust against overflows.
                let value = *i as u128 * factor as u128; // Multiplication might fail.
                let secs = (value / 10u128.pow(9)) as u64; // Cast might fail.
                let nanos = (value % 10u128.pow(9)) as u32; // Perfectly safe cast to u32.
                std::time::Duration::new(secs, nanos)
            }
            Some(LitKind::Float(f)) => {
                // TODO: Improve: Robust against overflows and inaccuracies.
                let value = *f * factor as f64;
                let secs = (value / 1_000_000_000f64) as u64;
                let nanos = (value % 1_000_000_000f64) as u32;
                std::time::Duration::new(secs, nanos)
            }
            _ => panic!(),
        }
    }

    fn extract_constant_value(e: &'a Expression) -> Option<usize> {
        match e.kind {
            ExpressionKind::Lit(ref lit) => match lit.kind {
                LitKind::Int(i) => Some(i as usize),
                _ => unimplemented!(),
            },
            ExpressionKind::Ident(_) => unimplemented!(), // Here, we should check the declaration. If it is a constant -> fine.
            _ => unimplemented!(),
        }
    }
}
