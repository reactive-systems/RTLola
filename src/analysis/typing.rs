//! Implementation of the Hindley-Milner type inference through unification
//!
//! Relevant references:
//! * Introduction to Hindley-Milner type inference https://eli.thegreenplace.net/2018/type-inference/
//! * Unification algorithm https://eli.thegreenplace.net/2018/unification/
//! * Unification in Rust http://smallcultfollowing.com/babysteps/blog/2017/03/25/unification-in-chalk-part-1/
//! * Ena (union-find package) https://crates.io/crates/ena

use super::naming::{Declaration, DeclarationTable};
use crate::ast::LolaSpec;
use crate::ast::{
    BinOp, Constant, Expression, Input, Literal, Offset, Output, Type, TypeKind, UnOp, ExpressionKind, LitKind
};
use crate::reporting::Handler;
use crate::reporting::LabeledSpan;
use crate::stdlib::{FuncDecl, MethodLookup, Parameter};
use crate::ty::{Ty, TypeConstraint};
use ast_node::NodeId;
use ast_node::Span;
use ena::unify::{EqUnifyValue, InPlaceUnificationTable, UnificationTable, UnifyKey, UnifyValue};
use log::{debug, trace};
use std::collections::HashMap;

pub(crate) struct TypeAnalysis<'a> {
    handler: &'a Handler,
    declarations: &'a DeclarationTable<'a>,
    method_lookup: MethodLookup,
    unifier: Unifier,
    /// maps `NodeId`'s to the variables used in `unifier`
    var_lookup: HashMap<NodeId, InferVar>,
}

impl<'a> TypeAnalysis<'a> {
    pub(crate) fn new(
        handler: &'a Handler,
        declarations: &'a DeclarationTable<'a>,
    ) -> TypeAnalysis<'a> {
        TypeAnalysis {
            handler,
            declarations,
            method_lookup: MethodLookup::new(),
            unifier: Unifier::new(),
            var_lookup: HashMap::new(),
        }
    }

    pub(crate) fn check(&mut self, spec: &'a LolaSpec) {
        self.infer_types(spec);
        if self.handler.contains_error() {
            return;
        }

        self.assign_types(spec);
    }

    fn infer_types(&mut self, spec: &'a LolaSpec) {
        trace!("infer types");
        for constant in &spec.constants {
            self.infer_constant(constant).unwrap_or_else(|_| {
                debug!("type inference failed for {}", constant);
            });
        }

        for input in &spec.inputs {
            self.infer_input(input).unwrap_or_else(|_| {
                debug!("type inference failed for {}", input);
            });
        }

        for output in &spec.outputs {
            self.infer_output(output).unwrap_or_else(|_| {
                debug!("type inference failed for {}", output);
            });
        }

        for output in &spec.outputs {
            self.infer_output_expression(output).unwrap_or_else(|_| {
                debug!("type inference failed for {}", output);
            });
        }
    }

    fn new_var(&mut self, node: NodeId) -> InferVar {
        assert!(!self.var_lookup.contains_key(&node));
        let var = self.unifier.new_var();
        self.var_lookup.insert(node, var);
        var
    }

    fn infer_constant(&mut self, constant: &'a Constant) -> Result<(), ()> {
        trace!("infer type for {}", constant);
        let var = self.new_var(constant._id);

        // generate constraint in case a type is annotated
        if let Some(type_name) = constant.ty.as_ref() {
            assert!(!self.var_lookup.contains_key(&type_name._id));

            if let Some(ty) = self.declarations.get(&type_name._id) {
                match ty {
                    Declaration::Type(ty) => {
                        self.unifier
                            .unify_ty_ty(Ty::Infer(var), (*ty).clone())
                            .map_err(|err| self.handle_error(err, type_name._span))?;
                    }
                    _ => unreachable!(),
                }
            }
        }
        // generate constraint from literal
        match self.get_constraint_for_literal(&constant.literal) {
            ConstraintOrType::Type(ty) => self
                .unifier
                .unify_var_ty(var, ty)
                .map_err(|err| self.handle_error(err, constant.literal._span))?,
            ConstraintOrType::Constraint(constr) => self
                .unifier
                .add_constraint(var, constr)
                .map_err(|err| self.handle_error(err, constant.literal._span))?,
        };
        Ok(())
    }

    fn infer_input(&mut self, input: &'a Input) -> Result<(), ()> {
        trace!("infer type for {}", input);
        let var = self.new_var(input._id);
        let ty_var = self.new_var(input.ty._id);

        match &input.ty.kind {
            TypeKind::Simple(_) => {
                let ty = self.declarations[&input.ty._id];
                match ty {
                    Declaration::Type(ty) => {
                        // constraints:
                        // - ?ty_var = `ty`
                        // - ?var = EventStream<?ty_var>
                        self.unifier
                            .unify_var_ty(ty_var, (*ty).clone())
                            .map_err(|err| self.handle_error(err, input.ty._span))?;
                    }
                    _ => unreachable!(),
                }
            }
            TypeKind::Tuple(tuple) => {
                let ty = self.get_tuple_type(tuple);
                // ?ty_var = `ty`
                self.unifier
                    .unify_var_ty(ty_var, ty)
                    .map_err(|err| self.handle_error(err, input.ty._span))?;
            }
            _ => unreachable!(),
        }
        self.unifier
            .unify_var_ty(var, Ty::EventStream(Box::new(Ty::Infer(ty_var))))
            .map_err(|err| self.handle_error(err, input.name.span))
    }

    fn infer_output(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("infer type for {}", output);
        let var = self.new_var(output._id);

        // generate constraint in case a type is annotated
        let ty_var = self.new_var(output.ty._id);
        match &output.ty.kind {
            TypeKind::Inferred => {}
            TypeKind::Simple(_) => {
                let ty = self.declarations[&output.ty._id];
                match ty {
                    // ?ty_var = `ty`
                    Declaration::Type(ty) => {
                        self.unifier
                            .unify_var_ty(ty_var, (*ty).clone())
                            .map_err(|err| self.handle_error(err, output.ty._span))?;
                    }
                    _ => unreachable!(),
                }
            }
            TypeKind::Tuple(tuple) => {
                let ty = self.get_tuple_type(tuple);
                // ?ty_var = `ty`
                self.unifier
                    .unify_var_ty(ty_var, ty)
                    .map_err(|err| self.handle_error(err, output.ty._span))?;
            }
            TypeKind::Malformed(_) => unreachable!(),
        }

        // ?var = EventStream<?ty_var>
        self.unifier
            .unify_var_ty(var, Ty::EventStream(Box::new(Ty::Infer(ty_var))))
            .map_err(|err| self.handle_error(err, output.name.span))
    }

    fn infer_output_expression(&mut self, output: &'a Output) -> Result<(), ()> {
        trace!("infer type for {}", output);
        let ty_var = self.var_lookup[&output.ty._id];

        // generate constraint for expression
        self.infer_expression(&output.expression)?;

        // match stream type with expression type
        // ?ty_var = ?expr_var
        self.unifier
            .unify_var_var(ty_var, self.var_lookup[&output.expression._id])
            .map_err(|err| self.handle_error(err, output.expression._span))
    }

    fn get_constraint_for_literal(&self, lit: &Literal) -> ConstraintOrType {
        use crate::ast::LitKind::*;
        match lit.kind {
            Str(_) | RawStr(_) => ConstraintOrType::Type(Ty::String),
            Bool(_) => ConstraintOrType::Type(Ty::Bool),
            Int(i) => ConstraintOrType::Constraint(if i < 0 {
                TypeConstraint::SignedInteger
            } else {
                TypeConstraint::Integer
            }),
            Float(_) => ConstraintOrType::Constraint(TypeConstraint::FloatingPoint),
        }
    }

    fn infer_expression(&mut self, expr: &'a Expression) -> Result<(), ()> {
        let var = self.new_var(expr._id);

        use crate::ast::ExpressionKind::*;
        match &expr.kind {
            Lit(l) => {
                // generate constraint from literal
                match self.get_constraint_for_literal(&l) {
                    ConstraintOrType::Type(ty) => self
                        .unifier
                        .unify_var_ty(var, ty)
                        .map_err(|err| self.handle_error(err, expr._span))?,
                    ConstraintOrType::Constraint(constr) => {
                        self.unifier
                            .add_constraint(var, constr)
                            .map_err(|err| self.handle_error(err, expr._span))?;
                    }
                };
            }
            Ident(_) => {
                let decl = self.declarations[&expr._id];

                match decl {
                    Declaration::Const(constant) => {
                        let const_var = self.var_lookup[&constant._id];
                        self.unifier
                            .unify_var_var(var, const_var)
                            .map_err(|err| self.handle_error(err, expr._span))?;
                    }
                    Declaration::In(input) => {
                        let in_var = self.var_lookup[&input._id];
                        self.unifier
                            .unify_var_var(var, in_var)
                            .map_err(|err| self.handle_error(err, expr._span))?;
                    }
                    Declaration::Out(output) => {
                        let out_var = self.var_lookup[&output._id];
                        self.unifier
                            .unify_var_var(var, out_var)
                            .map_err(|err| self.handle_error(err, expr._span))?;
                    }
                    _ => unreachable!(),
                }
            }
            Ite(cond, left, right) => {
                // recursion
                self.infer_expression(cond)?;
                self.infer_expression(left)?;
                self.infer_expression(right)?;

                // constraints
                // - cond = bool
                // - left = right
                // - expr = left
                let cond_var = self.var_lookup[&cond._id];
                let left_var = self.var_lookup[&left._id];
                let right_var = self.var_lookup[&right._id];
                self.unifier
                    .unify_var_ty(cond_var, Ty::Bool)
                    .map_err(|err| {
                        self.handle_error(err, cond._span);
                    })?;
                self.unifier
                    .unify_var_var(left_var, right_var)
                    .map_err(|err| {
                        self.handle_error(err, right._span);
                    })?;
                self.unifier.unify_var_var(var, left_var).map_err(|err| {
                    self.handle_error(err, expr._span);
                })?;
            }
            Unary(op, appl) => {
                // recursion
                self.infer_expression(appl)?;

                let appl_var = self.var_lookup[&appl._id];
                use self::UnOp::*;
                match op {
                    Not => {
                        // ?appl_var = Bool
                        // ?var = Bool
                        self.unifier
                            .unify_var_ty(appl_var, Ty::Bool)
                            .map_err(|err| {
                                self.handle_error(err, appl._span);
                            })?;
                        self.unifier.unify_var_ty(var, Ty::Bool).map_err(|err| {
                            self.handle_error(err, expr._span);
                        })?;
                    }
                    _ => unimplemented!("unary operator {}", op),
                }
            }
            Binary(op, left, right) => {
                // recursion
                self.infer_expression(left)?;
                self.infer_expression(right)?;

                let left_var = self.var_lookup[&left._id];
                let right_var = self.var_lookup[&right._id];
                use self::BinOp::*;
                match op {
                    Eq => {
                        // ?new_var :< Equality
                        // ?new_var = ?left_var
                        // ?new_var = ?right_var
                        // ?new_var = ?var
                        let new_var = self.unifier.new_var();
                        self.unifier
                            .add_constraint(new_var, TypeConstraint::Equality)
                            .expect("adding constraint cannot fail, new_var is a fresh variable");

                        self.unifier
                            .unify_var_var(new_var, left_var)
                            .map_err(|err| {
                                self.handle_error(err, left._span);
                            })?;
                        self.unifier
                            .unify_var_var(new_var, right_var)
                            .map_err(|err| {
                                self.handle_error(err, right._span);
                            })?;
                        self.unifier.unify_var_var(new_var, var).map_err(|err| {
                            self.handle_error(err, expr._span);
                        })?;
                    }
                    Add => {
                        // ?new_var :< Numeric
                        // ?new_var = ?left_var
                        // ?new_var = ?right_var
                        // ?new_var = ?var
                        let new_var = self.unifier.new_var();
                        self.unifier
                            .add_constraint(new_var, TypeConstraint::Numeric)
                            .expect("adding constraint cannot fail, new_var is a fresh variable");

                        self.unifier
                            .unify_var_var(new_var, left_var)
                            .map_err(|err| {
                                self.handle_error(err, left._span);
                            })?;
                        self.unifier
                            .unify_var_var(new_var, right_var)
                            .map_err(|err| {
                                self.handle_error(err, right._span);
                            })?;
                        self.unifier.unify_var_var(new_var, var).map_err(|err| {
                            self.handle_error(err, expr._span);
                        })?;
                    }
                    _ => {
                        println!("{}", op);
                        unimplemented!()
                    }
                }
            }
            // discrete offset
            Lookup(stream, Offset::DiscreteOffset(off_expr), None) => {
                // recursion
                self.infer_expression(off_expr)?;
                let off_var = self.var_lookup[&off_expr._id];

                let stream_var = self.new_var(stream._id);

                let negative_offset = match &off_expr.kind {
                    ExpressionKind::Lit(l) => {
                        match l.kind {
                            LitKind::Int(i) => {
                                i < 0
                            }
                            _ => {
                                unreachable!("offset expressions have to be integers")
                            }
                        }
                    },
                    _ => unreachable!("offset expressions have to be literal"),
                };

                // constraints
                // off_expr = {integer}
                // stream = EventStream<?1>
                // if negative_offset { expr = Option<?1> } else { expr = ?1 }
                self.unifier
                    .add_constraint(off_var, TypeConstraint::Integer)
                    .map_err(|err| {
                        self.handle_error(err, off_expr._span);
                    })?;
                // need to derive "inner" type `?1`
                let decl = self.declarations[&stream._id];
                let inner_var: InferVar = match decl {
                    Declaration::In(input) => self.var_lookup[&input.ty._id],
                    Declaration::Out(output) => self.var_lookup[&output.ty._id],
                    _ => unreachable!(),
                };
                self.unifier
                    .unify_var_ty(stream_var, Ty::EventStream(Box::new(Ty::Infer(inner_var))))
                    .map_err(|err| {
                        self.handle_error(err, stream._span);
                    })?;
                if negative_offset {
self.unifier
                    .unify_var_ty(var, Ty::Option(Box::new(Ty::Infer(inner_var))))
                    .map_err(|err| {
                        self.handle_error(err, expr._span);
                    })?;    
                } else {
                    self.unifier
                    .unify_var_var(var, inner_var)
                    .map_err(|err| {
                        self.handle_error(err, expr._span);
                    })?;
                }
                
            }
            Default(left, right) => {
                // recursion
                self.infer_expression(left)?;
                self.infer_expression(right)?;

                let left_var = self.var_lookup[&left._id];
                let right_var = self.var_lookup[&right._id];

                // constraints
                // left = Option<expr>
                // expr = right
                self.unifier
                    .unify_var_ty(left_var, Ty::Option(Box::new(Ty::Infer(var))))
                    .map_err(|err| self.handle_error(err, left._span))?;
                self.unifier
                    .unify_var_var(var, right_var)
                    .map_err(|err| self.handle_error(err, right._span))?;
            }
            Function(_, params) => {
                let decl = self.declarations[&expr._id];
                let fun_decl = match decl {
                    Declaration::Func(fun_decl) => fun_decl,
                    _ => unreachable!("expected function declaration"),
                };
                assert_eq!(params.len(), fun_decl.parameters.len());
                println!("{:?}", fun_decl);

                // recursion
                for param in params {
                    self.infer_expression(param)?;
                }

                let params: Vec<&Expression> = params.iter().map(|e| e.as_ref()).collect();

                self.infer_function_application(var, expr._span, fun_decl, params.as_slice())?;
            }
            Method(base, name, params) => {
                // recursion
                self.infer_expression(base)?;

                if let Some(infered) = self.unifier.get_type(self.var_lookup[&base._id]) {
                    debug!("{} {}", base, infered);
                    if let Some(fun_decl) = self.method_lookup.get(&infered, name.name.as_str()) {
                        debug!("{:?}", fun_decl);

                        // recursion
                        for param in params {
                            self.infer_expression(param)?;
                        }

                        let mut parameters = vec![base.as_ref()];
                        parameters.extend(params.iter().map(|e| e.as_ref()));

                        self.infer_function_application(
                            var,
                            expr._span,
                            &fun_decl,
                            parameters.as_slice(),
                        )?;
                    } else {
                        panic!("could not find `{}`", name);
                    }
                } else {
                    panic!("could not get type of `{}`", base);
                }
            }
            Tuple(expressions) => {
                // recursion
                for element in expressions {
                    self.infer_expression(element)?;
                }
                // ?var = Tuple(?expr1, ?expr2, ..)
                self.unifier
                    .unify_var_ty(
                        var,
                        Ty::Tuple(
                            expressions
                                .iter()
                                .map(|e| {
                                    let infer_var = self.var_lookup[&e._id];
                                    Ty::Infer(infer_var)
                                })
                                .collect(),
                        ),
                    )
                    .map_err(|err| self.handle_error(err, expr._span))?;
            }
            Field(base, ident) => {
                // recursion
                self.infer_expression(base)?;

                if let Some(infered) = self.unifier.get_type(self.var_lookup[&base._id]) {
                    debug!("{} {}", base, infered);
                    match infered {
                        Ty::Tuple(inner) => {
                            let num: usize = ident
                                .name
                                .parse::<usize>()
                                .expect("verify that this is checked earlier");
                            if num >= inner.len() {
                                self.handler.error_with_span(
                                    &format!("Try to access tuple at position {}", num),
                                    LabeledSpan::new(ident.span, "", true),
                                );
                                return Err(());
                            }
                            // ?var = inner[num]
                            self.unifier
                                .unify_var_ty(var, inner[num].clone())
                                .map_err(|err| self.handle_error(err, expr._span))?;
                        }
                        _ => {
                            self.handler.error_with_span(
                                &format!("Type `{}` has no field `{}`", infered, ident.name),
                                LabeledSpan::new(ident.span, "unknown field", true),
                            );
                            return Err(());
                        }
                    }
                }
            }
            t => unimplemented!("expression `{:?}`", t),
        }
        Ok(())
    }

    fn infer_function_application(
        &mut self,
        var: InferVar,
        span: Span,
        fun_decl: &FuncDecl,
        params: &[&'a Expression],
    ) -> Result<(), ()> {
        assert_eq!(params.len(), fun_decl.parameters.len());

        // build symbolic names for generic arguments
        let generics: Vec<InferVar> = fun_decl
            .generics
            .iter()
            .map(|gen| {
                let var = self.unifier.new_var();
                self.unifier
                    .add_constraint(var, gen.constraint)
                    .expect("cannot fail as var is freshly created");
                var
            })
            .collect();

        for (type_param, parameter) in fun_decl.parameters.iter().zip(params) {
            let param_var = self.var_lookup[&parameter._id];
            match type_param {
                Parameter::Generic(num) => {
                    // ?generic = ?type_var
                    self.unifier
                        .unify_var_var(generics[*num as usize], param_var)
                        .map_err(|err| self.handle_error(err, parameter._span))?;
                }
                Parameter::Type(ty) => {
                    // ?param_var = `ty`
                    self.unifier
                        .unify_var_ty(param_var, ty.clone())
                        .map_err(|err| self.handle_error(err, parameter._span))?;
                }
            }
        }
        // return type
        match fun_decl.return_type {
            Parameter::Generic(num) => {
                // ?generic = ?var
                self.unifier
                    .unify_var_var(generics[num as usize], var)
                    .map_err(|err| self.handle_error(err, span))?;
            }
            Parameter::Type(ref ty) => {
                // ?param_var = `ty`
                self.unifier
                    .unify_var_ty(var, ty.clone())
                    .map_err(|err| self.handle_error(err, span))?;
            }
        }
        Ok(())
    }

    fn get_tuple_type(&mut self, tuple: &[Box<Type>]) -> Ty {
        let mut inner = Vec::new();
        for t in tuple {
            match t.kind {
                TypeKind::Simple(_) => {
                    let ty = self.declarations[&t._id];
                    match ty {
                        Declaration::Type(ty) => inner.push(ty.clone()),
                        _ => unreachable!(),
                    }
                }
                TypeKind::Tuple(_) => unimplemented!(),
                _ => unreachable!(),
            }
        }
        Ty::Tuple(inner)
    }

    /// Assigns types as infered
    fn assign_types(&mut self, spec: &LolaSpec) {
        for constant in &spec.constants {
            debug!(
                "{} has type {}",
                constant,
                self.unifier.get_final_type(self.var_lookup[&constant._id])
            );
        }
        for input in &spec.inputs {
            debug!(
                "{} has type {}",
                input,
                self.unifier.get_final_type(self.var_lookup[&input._id])
            );
        }
        for output in &spec.outputs {
            debug!(
                "{} has type {}",
                output,
                self.unifier.get_final_type(self.var_lookup[&output._id])
            );
        }
    }

    fn handle_error(&self, err: InferError, span: Span) {
        match err {
            InferError::TypeMismatch(ty_l, ty_r) => {
                self.handler.error_with_span(
                    &format!("Type mismatch between `{}` and `{}`", ty_l, ty_r),
                    LabeledSpan::new(
                        span,
                        &format!("expected `{}`, found `{}`", ty_l, ty_r),
                        true,
                    ),
                );
            }
            InferError::UnsatisfiedConstraint(ty, constr) => {
                self.handler.error_with_span(
                    &format!("Type `{}` does not satisfy constraint `{}`", ty, constr),
                    LabeledSpan::new(
                        span,
                        &format!("expected `{}`, found `{}`", constr, ty),
                        true,
                    ),
                );
            }
            InferError::ConflictingConstraint(left, right) => {
                self.handler.error_with_span(
                    &format!("Conflicting constraints `{}` and `{}`", left, right),
                    LabeledSpan::new(
                        span,
                        &format!("no concrete type satisfies `{}` and `{}`", left, right),
                        true,
                    ),
                );
            }
            InferError::CyclicDependency(_, _) => {
                self.handler.error_with_span(
                    "Cannot infer type",
                    LabeledSpan::new(span, "consider using a type annotation", true),
                );
            }
        }
    }
}

enum ConstraintOrType {
    Type(Ty),
    Constraint(TypeConstraint),
}

/// We have two types of constraints, equality constraints and subtype constraints
struct Unifier {
    /// union-find data structure representing the current state of the unification
    table: InPlaceUnificationTable<InferVar>,
}

impl Unifier {
    fn new() -> Unifier {
        Unifier {
            table: UnificationTable::new(),
        }
    }

    fn new_var(&mut self) -> InferVar {
        self.table.new_key(InferVarVal::Unknown)
    }

    fn unify_ty_ty(&mut self, left: Ty, right: Ty) -> InferResult {
        debug!("unify ty ty {} {}", left, right);
        assert!(left != right);

        match (left, right) {
            (Ty::Infer(l), Ty::Infer(r)) => self.table.unify_var_var(l, r),
            (Ty::Infer(n), ty) => self.unify_var_ty(n, ty),
            (ty, Ty::Infer(n)) => self.unify_var_ty(n, ty),
            (_, _) => unreachable!(),
        }
    }

    fn unify_var_var(&mut self, left: InferVar, right: InferVar) -> InferResult {
        debug!("unify var var {} {}", left, right);
        match (self.table.probe_value(left), self.table.probe_value(right)) {
            (InferVarVal::Known(ty_l), InferVarVal::Known(ty_r)) => {
                // if both variables have values, we try to unify them recursively
                if self.types_equal_rec(&ty_l, &ty_r) {
                    return Ok(());
                } else {
                    return Err(InferError::TypeMismatch(ty_l, ty_r));
                }
            }
            _ => {}
        }
        self.table.unify_var_var(left, right)
    }

    fn unify_var_ty(&mut self, var: InferVar, ty: Ty) -> InferResult {
        debug!("unify var ty {} {}", var, ty);
        if let Ty::Infer(other) = ty {
            return self.unify_var_var(var, other);
        }
        match &ty {
            Ty::Infer(_) => unreachable!(),
            _ => {}
        }
        if self.check_occurrence(var, &ty) {
            return Err(InferError::CyclicDependency(var, ty));
        }
        if let InferVarVal::Known(val) = self.table.probe_value(var) {
            if self.types_equal_rec(&val, &ty) {
                self.table.unify_var_value(var, InferVarVal::Known(ty))
            } else {
                Err(InferError::TypeMismatch(val, ty))
            }
        } else {
            self.table.unify_var_value(var, InferVarVal::Known(ty))
        }
    }

    /// Checks if `var` occurs in `ty`
    fn check_occurrence(&mut self, var: InferVar, ty: &Ty) -> bool {
        trace!("check occurrence {} {}", var, ty);
        match ty {
            Ty::Infer(t) => self.table.unioned(var, *t),
            Ty::EventStream(ty) => self.check_occurrence(var, &ty),
            Ty::TimedStream(_) => unimplemented!(),
            Ty::Tuple(t) => t.iter().any(|e| self.check_occurrence(var, e)),
            _ => false,
        }
    }

    /// Checks recursively if types are equal. Tries to unify type parameters if possible.
    fn types_equal_rec(&mut self, left: &Ty, right: &Ty) -> bool {
        debug!("comp {} {}", left, right);
        match (left, right) {
            (&Ty::Infer(l), &Ty::Infer(r)) => {
                if self.table.unioned(l, r) {
                    true
                } else {
                    // try to unify values
                    self.unify_var_var(l, r).is_ok()
                }
            }
            (&Ty::Infer(var), ty) => {
                // try to unify
                self.unify_var_ty(var, ty.clone()).is_ok()
            }
            (ty, &Ty::Infer(var)) => {
                // try to unify
                self.unify_var_ty(var, ty.clone()).is_ok()
            }
            (Ty::Constr(constr_l), &Ty::Constr(constr_r)) => {
                constr_l.conjunction(constr_r).is_some()
            }
            (&Ty::Constr(constr), other) => other.satisfies(constr),
            (other, &Ty::Constr(constr)) => other.satisfies(constr),
            (Ty::Option(l), Ty::Option(r)) => self.types_equal_rec(l, r),
            (Ty::EventStream(l), Ty::EventStream(r)) => self.types_equal_rec(l, r),
            (Ty::Tuple(l), Ty::Tuple(r)) => {
                if l.len() != r.len() {
                    return false;
                }
                l.iter()
                    .zip(r)
                    .all(|(e_l, e_r)| self.types_equal_rec(e_l, e_r))
            }
            (Ty::TimedStream(_), Ty::TimedStream(_)) => unimplemented!(),
            (l, r) => l == r,
        }
    }

    fn add_constraint(&mut self, var: InferVar, constr: TypeConstraint) -> InferResult {
        debug!("add constraint {} {}", var, constr);
        self.table
            .unify_var_value(var, InferVarVal::Known(Ty::Constr(constr)))
    }

    /// Returns type where every inference variable is substituted
    fn get_final_type(&mut self, var: InferVar) -> Ty {
        use self::InferVarVal::*;
        match self.table.probe_value(var) {
            Unknown => Ty::Error,
            Known(Ty::Infer(other)) => self.get_final_type(other),
            Known(Ty::Constr(constr)) => {
                if let Some(ty) = constr.has_default() {
                    // TODO: emit warning that default type has been used
                    ty
                } else {
                    Ty::Error
                }
            }
            Known(Ty::EventStream(ty)) => {
                let inner = match *ty {
                    Ty::Infer(other) => self.get_final_type(other),
                    ref t => t.clone(),
                };
                Ty::EventStream(Box::new(inner))
            }
            Known(t) => t.clone(),
        }
    }

    /// returns current value of inference variable
    fn get_type(&mut self, var: InferVar) -> Option<Ty> {
        if let InferVarVal::Known(ty) = self.table.probe_value(var) {
            Some(ty)
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct InferVar(u32);

// used in UnifcationTable
impl UnifyKey for InferVar {
    type Value = InferVarVal;
    fn index(&self) -> u32 {
        self.0
    }
    fn from_index(u: u32) -> InferVar {
        InferVar(u)
    }
    fn tag() -> &'static str {
        "InferVar"
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum InferVarVal {
    Known(Ty),
    Unknown,
}

impl UnifyValue for InferVarVal {
    type Error = InferError;

    /// the idea of the unification is to merge two types and always taking the more concrete value
    fn unify_values(left: &Self, right: &Self) -> Result<Self, InferError> {
        use self::InferVarVal::*;
        match (left, right) {
            (Known(ty_l), Known(ty_r)) => Ok(Known(ty_l.unify(ty_r)?)),
            (Known(_), Unknown) => Ok(left.clone()),
            (Unknown, Known(_)) => Ok(right.clone()),
            (Unknown, Unknown) => Ok(Unknown),
        }
    }
}

impl Ty {
    fn unify(&self, other: &Ty) -> Result<Ty, InferError> {
        match (self, other) {
            (&Ty::Infer(l), &Ty::Infer(r)) => Ok(self.clone()),
            (&Ty::Infer(_), _) => Ok(other.clone()),
            (_, &Ty::Infer(_)) => Ok(self.clone()),
            (&Ty::Constr(constr_l), &Ty::Constr(constr_r)) => {
                if let Some(combined) = constr_l.conjunction(constr_r) {
                    Ok(Ty::Constr(combined))
                } else {
                    Err(InferError::ConflictingConstraint(constr_l, constr_r))
                }
            }
            (&Ty::Constr(constr), other) => {
                if other.satisfies(constr) {
                    Ok(other.clone())
                } else {
                    Err(InferError::TypeMismatch(self.clone(), other.clone()))
                }
            }
            (other, &Ty::Constr(constr)) => {
                if other.satisfies(constr) {
                    Ok(other.clone())
                } else {
                    Err(InferError::TypeMismatch(self.clone(), other.clone()))
                }
            }
            (Ty::Option(l), Ty::Option(r)) => Ok(Ty::Option(Box::new(l.unify(r)?))),
            (Ty::EventStream(l), Ty::EventStream(r)) => Ok(Ty::EventStream(Box::new(l.unify(r)?))),
            (Ty::Tuple(l), Ty::Tuple(r)) => {
                if l.len() != r.len() {
                    return Err(InferError::TypeMismatch(self.clone(), other.clone()));
                }
                let mut inner = Vec::new();
                for (e_l, e_r) in l.iter().zip(r) {
                    inner.push(e_l.unify(e_r)?);
                }
                Ok(Ty::Tuple(inner))
            }
            (Ty::TimedStream(_), Ty::TimedStream(_)) => unimplemented!(),
            (l, r) => {
                if l == r {
                    Ok(self.clone())
                } else {
                    Err(InferError::TypeMismatch(self.clone(), other.clone()))
                }
            }
        }
    }
}

impl EqUnifyValue for Ty {}

impl std::fmt::Display for InferVar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

type InferResult = Result<(), InferError>;

#[derive(Debug)]
pub enum InferError {
    TypeMismatch(Ty, Ty),
    UnsatisfiedConstraint(Ty, TypeConstraint),
    ConflictingConstraint(TypeConstraint, TypeConstraint),
    CyclicDependency(InferVar, Ty),
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::analysis::id_assignment::*;
    use crate::analysis::naming::*;
    use crate::parse::*;
    use crate::reporting::Handler;
    use std::path::PathBuf;

    fn num_type_errors(spec: &str) -> usize {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));

        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new(&handler);
        na.check(&spec);
        assert!(
            !handler.contains_error(),
            "Spec produces errors in naming analysis."
        );
        let mut type_analysis = TypeAnalysis::new(&handler, &na.result);
        type_analysis.check(&spec);
        handler.emitted_errors()
    }

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn method_call() {
        let spec = "output count := count.offset(-1).default(0) + 1\n";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn simple_const_float() {
        let spec = "constant c: Float32 := 2.0";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn simple_const_int() {
        let spec = "constant c: Int8 := 3";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn simple_const_faulty() {
        let spec = "constant c: Int8 := true";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_signedness() {
        let spec = "constant c: UInt8 := -2";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_incorrect_float() {
        let spec = "constant c: UInt8 := 2.3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn simple_output() {
        let spec = "output o: Int8 := 3";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn simple_binary() {
        let spec = "output o: Int8 := 3 + 5";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn simple_unary() {
        let spec = "output o: Bool := !false";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn simple_unary_faulty() {
        // The negation should return a bool even if the underlying expression is wrong.
        // Thus, there is only one error here.
        let spec = "output o: Bool := !3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn simple_binary_faulty() {
        let spec = "output o: Float32 := false + 2.5";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn simple_ite() {
        let spec = "output o: Int8 := if false then 1 else 2";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn underspecified_ite_type() {
        let spec = "output o := if false then 1.3 else -2.0";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_ite_condition_faulty() {
        let spec = "output o: UInt8 := if 3 then 1 else 1";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_ite_arms_incompatible() {
        let spec = "output o: UInt8 := if true then 1 else false";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_underspecified_type() {
        let spec = "output o: Float32 := 2";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_trigonometric() {
        let spec = "import math\noutput o: Float32 := sin(2)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_trigonometric_faulty() {
        let spec = "import math\noutput o: UInt8 := cos(1)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_trigonometric_faulty_2() {
        let spec = "import math\noutput o: Float64 := cos(true)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_input_lookup() {
        let spec = "input a: UInt8\n output b: UInt8 := a";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_input_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_stream_lookup() {
        let spec = "output a: UInt8 := 3\n output b: UInt8 := a[0]";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_stream_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_stream_lookup_dft() {
        let spec = "output a: UInt8 := 3\n output b: UInt8 := a[0] ? 3";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_stream_lookup_dft_fault() {
        let spec = "output a: UInt8 := 3\n output b: Bool := a[0] ? false";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_invoke_type() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    #[ignore]
    fn test_invoke_type_faulty() {
        let spec = "input in: Bool\n output a<p1: Int8>: Int8 { invoke in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_extend_type() {
        let spec = "input in: Bool\n output a: Int8 { extend in } := 3";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    #[ignore]
    fn test_extend_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { extend in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_terminate_type() {
        let spec = "input in: Bool\n output a: Int8 { terminate in } := 3";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    #[ignore]
    fn test_terminate_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { terminate in } := 3";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_param_spec() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(3)[-2] ? 1";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_param_spec_faulty() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(true)[-2] ? 1";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_lookup_incomp() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: UInt8 := a(3)[2] ? 1";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_tuple() {
        let spec = "output out: (Int8, Bool) := (14, false)";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_tuple_faulty() {
        let spec = "output out: (Int8, Bool) := (14, 3)";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_tuple_access() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.1";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_tuple_access_faulty_type() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.0";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_tuple_access_faulty_len() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.2";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3]";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_window_widening() {
        let spec = "input in: Int8\n output out: Int64 {extend @5Hz}:= in[3s, Σ] ? 0";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 {extend @5Hz} := in[3s, Σ] ? 0";
        assert_eq!(0, num_type_errors(spec));
    }

    #[test]
    fn test_window_untimed() {
        let spec = "input in: Int8\n output out: Int16 := in[3s, Σ] ? 5";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_window_faulty() {
        let spec = "input in: Int8\n output out: Bool {extend @5Hz} := in[3s, Σ] ? true";
        assert_eq!(1, num_type_errors(spec));
    }

    #[test]
    fn test_involved() {
        let spec =
            "input velo: Float32\n output avg: Float64 {extend @5Hz} := velo[1h, avg] ? 10000";
        assert_eq!(0, num_type_errors(spec));
    }

}
