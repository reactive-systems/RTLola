use super::super::common::BuiltinType;
use super::super::naming::{Declaration, DeclarationTable};
use ast::*;
use std::collections::HashMap;
extern crate ast_node;
use super::candidates::*;
use super::type_error::*;
use super::TypeCheckResult;
use ast_node::{AstNode, NodeId};

pub(crate) struct TypeChecker<'a> {
    declarations: &'a DeclarationTable<'a>,
    spec: &'a LolaSpec,
    tt: HashMap<NodeId, Candidates>,
    errors: Vec<Box<TypeError<'a>>>,
}

// TODO: Check type of auxiliary streams!

impl<'a> TypeChecker<'a> {
    pub(crate) fn new(dt: &'a DeclarationTable, spec: &'a LolaSpec) -> TypeChecker<'a> {
        TypeChecker {
            declarations: dt,
            spec,
            tt: HashMap::new(),
            errors: Vec::new(),
        }
    }

    pub(crate) fn check(&mut self) -> TypeCheckResult<'a> {
        self.check_typedeclarations();
        //        println!("Errors after checking type declarations: {:?}.", self.errors);
        self.check_constants();
        //        println!("Errors after checking constants: {:?}.", self.errors);
        self.check_inputs();
        //        println!("Errors after checking inputs: {:?}.", self.errors);
        self.check_outputs();
        //        println!("Errors after checking outputs: {:?}.", self.errors);
        self.check_triggers();
        //        println!("Errors after checking triggers: {:?}.", self.errors);

        let type_table: HashMap<NodeId, super::super::common::Type> =
            self.tt.iter().map(|(nid, c)| (*nid, c.into())).collect();

        TypeCheckResult {
            errors: self.errors.clone(),
            type_table,
        }
    }

    fn check_triggers(&mut self) {
        for trigger in &self.spec.trigger {
            let was = self.get_candidates(&trigger.expression);
            if !was.is_logic() {
                // TODO: bool display
                self.reg_error(TypeError::IncompatibleTypes(
                    trigger,
                    format!("Expected {:?} but got {}.", BuiltinType::Bool, was),
                ));
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
            // TODO: Move into function: `Self × Option<ast::Type> -> Candidates`
            let cand = output
                .ty
                .as_ref()
                .map(|t| self.type_to_cands(t))
                .unwrap_or(Candidates::Any);
            self.reg_cand(*output.id(), cand);
        }
        for output in &self.spec.outputs {
            for param in &output.params {
                let ty = param
                    .ty
                    .as_ref()
                    .map(|t| self.type_to_cands(&t))
                    .unwrap_or(Candidates::Any);
                self.reg_cand(*param.id(), ty);
            }
            let was = self.get_candidates(&output.expression);
            let declared = output
                .ty
                .as_ref()
                .map(|t| self.type_to_cands(&t))
                .unwrap_or(Candidates::Any);
            if declared.meet(&was).is_none() {
                self.reg_error(TypeError::IncompatibleTypes(
                    output,
                    format!("Expected {} but got {}.", declared, was),
                ));
            }
            self.reg_cand(*output.id(), declared);
        }
    }

    fn check_constants(&mut self) {
        for constant in &self.spec.constants {
            let was = Candidates::from(&constant.literal);
            let declared = constant
                .ty
                .as_ref()
                .map(|t| self.type_to_cands(&t))
                .unwrap_or(Candidates::Any);
            if declared.meet(&was).is_none() {
                self.reg_error(TypeError::IncompatibleTypes(
                    constant,
                    format!("Expected {} but got {}.", declared, was),
                ));
            }
            self.reg_cand(*constant.id(), declared);
        }
    }

    fn check_inputs(&mut self) {
        for input in &self.spec.inputs {
            let cands = self.type_to_cands(&input.ty);
            self.reg_cand(*input.id(), cands);
        }
    }

    fn get_candidates(&mut self, e: &'a Expression) -> Candidates {
        match e.kind {
            ExpressionKind::Lit(ref lit) => self.reg_cand(*e.id(), Candidates::from(lit)),
            ExpressionKind::Ident(_) => {
                let cand = self.get_ty_from_decl(*e.id(), e);
                self.reg_cand(*e.id(), cand)
            }
            ExpressionKind::Default(ref expr, ref dft) => {
                let expr_ty = self.get_candidates(expr);
                let dft_ty = self.get_candidates(dft);
                let res = expr_ty.meet(&dft_ty);
                if res.is_none() {
                    self.reg_error(TypeError::IncompatibleTypes(e, String::from("A potentially undefined expression and its default need matching types.")));
                    dft_ty
                } else {
                    self.reg_cand(*e.id(), res)
                }
            }
            ExpressionKind::Lookup(ref inst, ref offset, ref op) => {
                self.check_offset(offset, e); // Return value does not matter.
                let target_stream_type = self.check_stream_instance(inst, e);

                if op.is_none() {
                    return self.reg_cand(*e.id(), target_stream_type);
                }
                let op = op.unwrap();
                // For all window operations, the source stream needs to be numeric.
                if !target_stream_type.is_numeric() {
                    return self.reg_error(TypeError::IncompatibleTypes(
                        e,
                        format!(
                            "Window source needs to be numeric for the {:?} aggregation.",
                            op
                        ),
                    ));
                }

                self.reg_cand(
                    *e.id(),
                    match op {
                        WindowOperation::Sum | WindowOperation::Product => target_stream_type,
                        WindowOperation::Average | WindowOperation::Integral => {
                            Candidates::Numeric(NumConfig::new_float(None))
                        }
                        WindowOperation::Count => {
                            Candidates::Numeric(NumConfig::new_unsigned(None))
                        }
                    },
                )
            }
            ExpressionKind::Binary(op, ref lhs, ref rhs) => {
                let lhs_type = self.get_candidates(lhs);
                let rhs_type = self.get_candidates(rhs);
                let meet_type = lhs_type.meet(&rhs_type);
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow
                        if meet_type.is_numeric() =>
                    {
                        meet_type
                    }
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow
                        if !meet_type.is_numeric() =>
                    {
                        self.reg_error(TypeError::IncompatibleTypes(
                            e,
                            format!(
                                "Binary operator {} applied to types {} and {}.",
                                op, lhs, rhs
                            ),
                        ));
                        Candidates::Numeric(NumConfig::new_unsigned(None))
                    }
                    BinOp::And | BinOp::Or if meet_type.is_logic() => meet_type,
                    BinOp::And | BinOp::Or if meet_type.is_logic() => {
                        self.reg_error(TypeError::IncompatibleTypes(
                            e,
                            format!(
                                "Binary operator {} applied to types {} and {}.",
                                op, lhs, rhs
                            ),
                        ));
                        Candidates::Concrete(BuiltinType::Bool)
                    }
                    // As long as two types are compatible in some way, we return a bool after the comparison.
                    BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt
                        if !meet_type.is_none() =>
                    {
                        Candidates::Concrete(BuiltinType::Bool)
                    }
                    BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt
                        if meet_type.is_none() =>
                    {
                        self.reg_error(TypeError::IncompatibleTypes(
                            e,
                            format!(
                                "Binary operator {} applied to types {} and {}.",
                                op, lhs, rhs
                            ),
                        ));
                        Candidates::Concrete(BuiltinType::Bool)
                    }
                    _ => unreachable!(),
                }
            }
            ExpressionKind::Unary(ref operator, ref operand) => {
                let op_type = self.get_candidates(operand);
                let res_type = match operator {
                    UnOp::Neg if op_type.is_numeric() => Some(op_type.clone().into_signed()),
                    UnOp::Not if op_type.is_logic() => Some(op_type.clone()),
                    _ => None,
                };
                match res_type {
                    Some(cand) => self.reg_cand(*e.id(), cand),
                    None => {
                        self.reg_error(TypeError::IncompatibleTypes(
                            e,
                            format!(
                                "Unary operator {} cannot be applied to type {}.",
                                operator, op_type
                            ),
                        ));
                        Candidates::Concrete(BuiltinType::Bool)
                    }
                }
            }
            ExpressionKind::Ite(ref cond, ref cons, ref alt) => {
                let cond = self.get_candidates(cond);
                let cons = self.get_candidates(cons);
                let alt = self.get_candidates(alt);
                if !cond.is_logic() {
                    self.reg_error(TypeError::IncompatibleTypes(
                        e,
                        format!(
                            "Expected boolean value in condition of `if` expression but got {}.",
                            cond
                        ),
                    ));
                }
                let res_type = cons.meet(&alt);
                if res_type.is_none() {
                    self.reg_error(TypeError::IncompatibleTypes(
                        e,
                        format!(
                            "Arms of if` expression have incompatible types {} and {}.",
                            cond, alt
                        ),
                    ))
                } else {
                    self.reg_cand(*e.id(), res_type)
                }
            }
            ExpressionKind::ParenthesizedExpression(_, ref expr, _) => {
                let res = self.get_candidates(expr);
                self.reg_cand(*e.id(), res)
            }
            ExpressionKind::MissingExpression() => self.reg_error(TypeError::MissingExpression(e)),
            ExpressionKind::Tuple(ref exprs) => {
                let cands: Vec<Candidates> = exprs.iter().map(|e| self.get_candidates(e)).collect();
                self.reg_cand(*e.id(), Candidates::Tuple(cands))
            }
            ExpressionKind::Function(ref kind, ref args) => self.check_function(e, *kind, args),
        }
    }

    fn check_function(
        &mut self,
        e: &'a Expression,
        kind: FunctionKind,
        args: &'a Vec<Box<Expression>>,
    ) -> Candidates {
        let cands: Vec<Candidates> = args.iter().map(|a| self.get_candidates(a)).collect();
        let numeric_check = Box::new(|c: &Candidates| c.is_numeric());
        let integer_check = Box::new(|c: &Candidates| c.is_integer());
        let float_check = Box::new(|c: &Candidates| c.is_float());
        let args = args.iter().map(|a| a.as_ref()).collect();
        match kind {
            FunctionKind::NthRoot => {
                let expected: Vec<(Box<Fn(&Candidates) -> bool>, &str)> = vec![
                    (integer_check, "integer value"),
                    (numeric_check, "numeric value"),
                ];
                self.check_n_ary_fn(e, args, expected, &cands);
                self.reg_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None)))
            }
            FunctionKind::Sqrt => {
                let expected: Vec<(Box<Fn(&Candidates) -> bool>, &str)> =
                    vec![(numeric_check, "numeric value")];
                self.check_n_ary_fn(e, args, expected, &cands);
                self.reg_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None)))
            }
            FunctionKind::Projection => {
                if cands.len() < 2 {
                    self.reg_error(TypeError::inv_num_of_args(e, 2, cands.len() as u8));
                }
                if !cands[0].is_unsigned() {
                    self.reg_error(TypeError::inv_argument(e, args[0], format!("Projection requires an unsigned integer as first argument but found {}.", cands[0])));
                }
                match cands[1] {
                    Candidates::Tuple(ref v) => {
                        match TypeChecker::convert_to_constant(args[1]) {
                            Some(n) if n < v.len() =>
                            // No way to recover from here.
                            {
                                self.reg_error(TypeError::inv_argument(
                                    e,
                                    args[1],
                                    format!(
                                        "Projection index {} exceeds tuple dimension {}.",
                                        n,
                                        v.len()
                                    ),
                                ))
                            }
                            Some(n) => self.reg_cand(*e.id(), cands[n].clone()),
                            None => self.reg_error(TypeError::ConstantValueRequired(e, args[1])),
                        }
                    }
                    _ => self.reg_error(TypeError::inv_argument(
                        e,
                        args[1],
                        format!(
                            "Projection requires a tuple as second argument but found {}.",
                            cands[1]
                        ),
                    )),
                }
            }
            FunctionKind::Sin
            | FunctionKind::Cos
            | FunctionKind::Tan
            | FunctionKind::Arcsin
            | FunctionKind::Arccos
            | FunctionKind::Arctan => {
                let expected: Vec<(Box<Fn(&Candidates) -> bool>, &str)> =
                    vec![(numeric_check, "numeric value")];
                self.check_n_ary_fn(e, args, expected, &cands);
                self.reg_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None)))
            }
            FunctionKind::Exp => {
                let expected: Vec<(Box<Fn(&Candidates) -> bool>, &str)> =
                    vec![(numeric_check, "numeric value")];
                self.check_n_ary_fn(e, args, expected, &cands);
                self.reg_cand(*e.id(), Candidates::Numeric(NumConfig::new_float(None)))
            }
            FunctionKind::Floor | FunctionKind::Ceil => {
                let expected: Vec<(Box<Fn(&Candidates) -> bool>, &str)> =
                    vec![(float_check, "float value")];
                self.check_n_ary_fn(e, args, expected, &cands);
                self.reg_cand(
                    *e.id(),
                    Candidates::Numeric(NumConfig::new_signed(cands[0].width())),
                )
            }
        }
    }

    fn check_n_ary_fn(
        &mut self,
        call: &'a Expression,
        args: Vec<&'a Expression>,
        expected: Vec<(Box<Fn(&Candidates) -> bool>, &str)>,
        was: &Vec<Candidates>,
    ) {
        if was.len() != expected.len() {
            self.reg_error(TypeError::inv_num_of_args(
                call,
                expected.len() as u8,
                was.len() as u8,
            ));
            return;
        }
        for ((pred, desc), ty, arg) in izip!(expected, was, args) {
            if !pred(ty) {
                self.reg_error(TypeError::inv_argument(
                    call,
                    arg,
                    format!("Required {} but found {}.", desc, ty),
                ));
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
                self.reg_error(TypeError::IncompatibleTypes(
                    origin,
                    String::from("A discrete offset must be an integer value."),
                ));
            }
            (false, false, _) => {
                self.reg_error(TypeError::IncompatibleTypes(
                    origin,
                    String::from("A real-time offset must be a numeric value."),
                ));
            }
        }
    }

    fn check_stream_instance(
        &mut self,
        inst: &'a StreamInstance,
        origin: &'a Expression,
    ) -> Candidates {
        let inst_candidates = self.get_ty_from_decl(*inst.id(), origin);
        match self.declarations.get(inst.id()) {
            Some(Declaration::Out(ref o)) => {
                if o.params.len() != inst.arguments.len() {
                    let msg = format!(
                        "Expected {}, but got {}.",
                        o.params.len(),
                        inst.arguments.len()
                    );
                    self.reg_error(TypeError::UnexpectedNumberOfArguments(origin, msg));
                } else {
                    // Check the parameter/argument pairs.
                    // We can only perform these checks of there is the correct number of arguments. Otherwise skip until correct number of arguments is provided.
                    let params: Vec<Candidates> = o
                        .params
                        .iter()
                        .flat_map(|p| &p.ty)
                        .map(|t| self.type_to_cands(&t))
                        .collect(); // Collect to resolve closure dependencies.
                    let arguments: Vec<Candidates> = inst
                        .arguments
                        .iter()
                        .map(|e: &Box<Expression>| self.get_candidates(e))
                        .collect();

                    if arguments.len() == inst.arguments.len() {
                        // Otherwise we had a problem earlier, which is already reported.
                        for (exp, was) in params.iter().zip(arguments) {
                            let res = exp.meet(&was);
                            // If `exp` or `was` is `none`, `res` will be `none`, too.
                            // We already reported the error in the recursion, so no need to repeat ourselves; skip.
                            if res.is_none() && !exp.is_none() && !was.is_none() {
                                self.reg_error(TypeError::IncompatibleTypes(
                                    origin,
                                    String::from(""),
                                ));
                            }
                        }
                    }
                }
                // Independent of all checks, pretend everything worked fine.
                self.reg_cand(*origin.id(), inst_candidates)
            }
            Some(Declaration::In(i)) => self.reg_cand(*origin.id(), inst_candidates),
            _ => self.reg_error(TypeError::UnknownIdentifier(inst)), // Unknown output, return Candidates::None.
        }
    }

    fn type_to_cands(&mut self, ty: &'a Type) -> Candidates {
        match ty.kind {
            TypeKind::Tuple(ref v) => {
                let vals = v.iter().map(|t| self.type_to_cands(t)).collect();
                return Candidates::Tuple(vals);
            }
            TypeKind::Malformed(_) => unreachable!(),
            TypeKind::Simple(_) => self.get_ty_from_decl(*ty.id(), ty),
        }
    }

    fn typedecl_to_cands(&mut self, td: &'a TypeDeclaration) -> Candidates {
        let _subs: Vec<Candidates> = td
            .fields
            .iter()
            .map(|field| self.type_to_cands(&field.ty))
            .collect();
        unimplemented!()
    }

    fn get_ty_from_decl(&mut self, nid: NodeId, ident: &'a AstNode<'a>) -> Candidates {
        match self.declarations.get(&nid) {
            Some(Declaration::Const(ref c)) => {
                let cand = self.get_from_tt(*c.id());
                assert!(cand.is_some()); // Since we added all declarations in the beginning, this value needs to be present.
                self.reg_cand(nid, cand.unwrap())
            }
            Some(Declaration::In(ref i)) => {
                let cand = self.get_from_tt(*i.id());
                assert!(cand.is_some()); // Since we added all declarations in the beginning, this value needs to be present.
                self.reg_cand(nid, cand.unwrap())
            }
            Some(Declaration::Out(ref o)) => {
                let cand = self.get_from_tt(*o.id());
                assert!(cand.is_some()); // Since we added all declarations in the beginning, this value needs to be present.
                self.reg_cand(nid, cand.unwrap())
            }
            Some(Declaration::UserDefinedType(_td)) => unimplemented!(),
            Some(Declaration::BuiltinType(ref b)) => self.reg_cand(nid, Candidates::from(b)),
            Some(Declaration::Param(ref p)) => {
                let cands =
                    p.ty.as_ref()
                        .map(|t| self.type_to_cands(&t))
                        .unwrap_or(Candidates::Any);
                self.reg_cand(nid, cands)
            }
            None => self.reg_error(TypeError::UnknownIdentifier(ident)),
        }
    }

    fn get_from_tt(&self, nid: NodeId) -> Option<Candidates> {
        self.tt.get(&nid).map(|t| t.clone())
    }

    fn reg_error(&mut self, e: TypeError<'a>) -> Candidates {
        self.errors.push(Box::new(e));
        Candidates::None // In the error case, we return the contradiction.
    }

    fn reg_cand(&mut self, nid: NodeId, cand: Candidates) -> Candidates {
        let res = match self.tt.get(&nid) {
            Some(c) => c.meet(&cand),
            None => cand,
        };
        self.tt.insert(nid, res.clone());
        res
    }
}
