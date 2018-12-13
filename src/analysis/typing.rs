//! Implementation of the Hindley-Milner type inference through unification
//!
//! Relevant references:
//! * https://eli.thegreenplace.net/2018/type-inference/
//! * https://eli.thegreenplace.net/2018/unification/

use super::naming::{Declaration, DeclarationTable};
use crate::ast::LolaSpec;
use crate::ast::{
    BinOp, Constant, Expression, ExpressionKind, Input, Literal, Offset, Output, TypeKind,
};
use crate::reporting::Handler;
use crate::stdlib::{MethodLookup, Parameter};
use crate::ty::{GenericTypeConstraint, Ty};
use ast_node::NodeId;
use ena::unify::{EqUnifyValue, InPlaceUnificationTable, UnificationTable, UnifyKey};
use log::{debug, error, trace};
use std::collections::HashMap;

pub(crate) struct TypeAnalysis<'a> {
    handler: &'a Handler,
    declarations: &'a DeclarationTable<'a>,
    /// unresolved method calls
    method_calls: Vec<&'a Expression>,
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
            method_calls: Vec::new(),
            method_lookup: MethodLookup::new(),
            unifier: Unifier::new(),
            var_lookup: HashMap::new(),
        }
    }

    pub(crate) fn check(&mut self, spec: &'a LolaSpec) {
        self.infer_types(spec);

        for fun in &self.method_calls {
            let (id, name, args) = match &fun.kind {
                ExpressionKind::Method(base, ident, args) => (base._id, ident.name.as_str(), args),
                _ => unreachable!(),
            };
            let infered = self.unifier.get_type(self.var_lookup[&id]);
            println!("{} {} {}", fun, infered, infered.is_error());
            if infered.is_error() {
                continue;
            }
            if let Some(fun_decl) = self.method_lookup.get(&infered, name) {
                println!("{:?}", fun_decl);
            }
        }

        assert!(self.method_calls.is_empty());

        self.assign_types(spec);

        unimplemented!();
    }

    fn infer_types(&mut self, spec: &'a LolaSpec) {
        for constant in &spec.constants {
            self.infer_constant(constant).unwrap_or_else(|e| {
                panic!("type inference failed: {}", e);
            });
        }

        for input in &spec.inputs {
            self.infer_input(input).unwrap_or_else(|e| {
                panic!("type inference failed: {}", e);
            });
        }

        for output in &spec.outputs {
            self.infer_output(output).unwrap_or_else(|e| {
                panic!("type inference failed: {}", e);
            });
        }

        for output in &spec.outputs {
            self.infer_output_expression(output).unwrap_or_else(|e| {
                panic!("type inference failed: {}", e);
            });
        }
    }

    fn new_var(&mut self, node: NodeId) -> InferVar {
        assert!(!self.var_lookup.contains_key(&node));
        let var = self.unifier.new_var();
        self.var_lookup.insert(node, var);
        var
    }

    fn infer_constant(&mut self, constant: &'a Constant) -> Result<(), String> {
        trace!("infer type for {}", constant);
        let var = self.new_var(constant._id);

        // generate constraint in case a type is annotated
        if let Some(type_name) = constant.ty.as_ref() {
            assert!(!self.var_lookup.contains_key(&type_name._id));

            if let Some(ty) = self.declarations.get(&type_name._id) {
                match ty {
                    Declaration::Type(ty) => {
                        self.unifier.add_equality(Ty::Infer(var), (*ty).clone())?
                    }
                    _ => unreachable!(),
                }
            }
        }
        // generate constraint from literal
        match self.get_constraint_for_literal(&constant.literal) {
            ConstraintOrType::Type(ty) => self.unifier.add_equality(Ty::Infer(var), ty)?,
            ConstraintOrType::Constraint(constr) => self.unifier.add_constraint(var, constr)?,
        };
        Ok(())
    }

    fn infer_input(&mut self, input: &'a Input) -> Result<(), String> {
        trace!("infer type for {}", input);
        let var = self.new_var(input._id);
        let ty_var = self.new_var(input.ty._id);

        if let Some(ty) = self.declarations.get(&input.ty._id) {
            match ty {
                Declaration::Type(ty) => {
                    // constraints:
                    // - ?ty_var = `ty`
                    // - ?var = EventStream<?ty_var>
                    self.unifier
                        .add_equality(Ty::Infer(ty_var), (*ty).clone())?;
                    self.unifier
                        .add_equality(Ty::Infer(var), Ty::EventStream(Box::new(Ty::Infer(ty_var))))
                }
                _ => unreachable!(),
            }
        } else {
            unreachable!();
        }
    }

    fn infer_output(&mut self, output: &'a Output) -> Result<(), String> {
        trace!("infer type for {}", output);
        let var = self.new_var(output._id);

        // generate constraint in case a type is annotated
        let ty_var = self.new_var(output.ty._id);
        match output.ty.kind {
            TypeKind::Inferred => {}
            _ => {
                if let Some(ty) = self.declarations.get(&output.ty._id) {
                    match ty {
                        // ?ty_var = `ty`
                        Declaration::Type(ty) => self
                            .unifier
                            .add_equality(Ty::Infer(ty_var), (*ty).clone())?,
                        _ => unreachable!(),
                    }
                } else {
                    unreachable!();
                }
            }
        }

        // ?var = EventStream<?ty_var>
        self.unifier
            .add_equality(Ty::Infer(var), Ty::EventStream(Box::new(Ty::Infer(ty_var))))
    }

    fn infer_output_expression(&mut self, output: &'a Output) -> Result<(), String> {
        trace!("infer type for {}", output);
        let ty_var = self.var_lookup[&output.ty._id];

        // generate constraint for expression
        self.infer_expression(&output.expression)?;

        // match stream type with expression type
        // ?ty_var = ?expr_var
        self.unifier.add_equality(
            Ty::Infer(ty_var),
            Ty::Infer(self.var_lookup[&output.expression._id]),
        )
    }

    fn get_constraint_for_literal(&self, lit: &Literal) -> ConstraintOrType {
        use crate::ast::LitKind::*;
        match lit.kind {
            Str(_) | RawStr(_) => ConstraintOrType::Type(Ty::String),
            Bool(_) => ConstraintOrType::Type(Ty::Bool),
            Int(_) => ConstraintOrType::Constraint(GenericTypeConstraint::Integer),
            Float(_) => ConstraintOrType::Constraint(GenericTypeConstraint::FloatingPoint),
        }
    }

    fn infer_expression(&mut self, expr: &'a Expression) -> InferResult {
        let var = self.new_var(expr._id);

        use crate::ast::ExpressionKind::*;
        match &expr.kind {
            Lit(l) => {
                // generate constraint from literal
                match self.get_constraint_for_literal(&l) {
                    ConstraintOrType::Type(ty) => self.unifier.add_equality(Ty::Infer(var), ty)?,
                    ConstraintOrType::Constraint(constr) => {
                        self.unifier.add_constraint(var, constr)?
                    }
                };
            }
            Ident(_) => {
                let decl = self.declarations[&expr._id];

                match decl {
                    Declaration::Const(constant) => {
                        let const_var = self.var_lookup[&constant._id];
                        self.unifier
                            .add_equality(Ty::Infer(var), Ty::Infer(const_var))?;
                    }
                    Declaration::In(input) => {
                        let in_var = self.var_lookup[&input._id];
                        self.unifier
                            .add_equality(Ty::Infer(var), Ty::Infer(in_var))?;
                    }
                    Declaration::Out(output) => {
                        let out_var = self.var_lookup[&output._id];
                        self.unifier
                            .add_equality(Ty::Infer(var), Ty::Infer(out_var))?;
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
                self.unifier.add_equality(Ty::Infer(cond_var), Ty::Bool)?;
                self.unifier
                    .add_equality(Ty::Infer(left_var), Ty::Infer(right_var))?;
                self.unifier
                    .add_equality(Ty::Infer(var), Ty::Infer(left_var))?;
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
                        self.unifier
                            .add_equality(Ty::Infer(left_var), Ty::Infer(right_var))?;
                        self.unifier
                            .add_constraint(left_var, GenericTypeConstraint::Equality)?;
                    }
                    Add => {
                        self.unifier
                            .add_equality(Ty::Infer(left_var), Ty::Infer(right_var))?;
                        self.unifier
                            .add_constraint(left_var, GenericTypeConstraint::Numeric)?;
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

                // constraints
                // off_expr = {integer}
                // stream = EventStream<?1>
                // expr = Option<?1>
                self.unifier
                    .add_constraint(off_var, GenericTypeConstraint::Integer)?;
                // need to derive "inner" type `?1`
                let decl = self.declarations[&stream._id];
                let inner_var: InferVar = match decl {
                    Declaration::In(input) => self.var_lookup[&input.ty._id],
                    Declaration::Out(_) => unimplemented!(),
                    Declaration::Const(_) => unimplemented!(),
                    _ => unreachable!(),
                };
                self.unifier.add_equality(
                    Ty::Infer(stream_var),
                    Ty::EventStream(Box::new(Ty::Infer(inner_var))),
                )?;
                self.unifier
                    .add_equality(Ty::Infer(var), Ty::Option(Box::new(Ty::Infer(inner_var))))?;
            }
            Default(left, right) => {
                // recursion
                self.infer_expression(left);
                self.infer_expression(right);

                let left_var = self.var_lookup[&left._id];
                let right_var = self.var_lookup[&right._id];

                // constraints
                // left = Option<expr>
                // expr = right
                self.unifier
                    .add_equality(Ty::Infer(left_var), Ty::Option(Box::new(Ty::Infer(var))))?;
                self.unifier
                    .add_equality(Ty::Infer(var), Ty::Infer(right_var))?;
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
                    self.infer_expression(param);
                }

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
                            self.unifier.add_equality(
                                Ty::Infer(generics[*num as usize]),
                                Ty::Infer(param_var),
                            )?;
                        }
                        Parameter::Type(ty) => {
                            // ?param_var = `ty`
                            self.unifier
                                .add_equality(Ty::Infer(param_var), ty.clone())?;
                        }
                    }
                }
                // return type
                match fun_decl.return_type {
                    Parameter::Generic(num) => {
                        // ?generic = ?var
                        self.unifier
                            .add_equality(Ty::Infer(generics[num as usize]), Ty::Infer(var))?;
                    }
                    Parameter::Type(ref ty) => {
                        // ?param_var = `ty`
                        self.unifier.add_equality(Ty::Infer(var), ty.clone())?;
                    }
                }
            }
            Method(base, name, params) => {
                // recursion
                self.infer_expression(base)?;

                let infered = self.unifier.get_type(self.var_lookup[&base._id]);
                println!("{} {} {}", base, infered, infered.is_error());
                if !infered.is_error() {
                    if let Some(fun_decl) = self.method_lookup.get(&infered, name.name.as_str()) {
                        println!("{:?}", fun_decl);
                    }
                }

                for param in params {
                    self.infer_expression(param)?;
                }

                // save for later inspection
                self.method_calls.push(expr);
            }
            _ => unimplemented!(),
        }
        Ok(())
    }

    /// Assigns types as infered
    fn assign_types(&mut self, spec: &LolaSpec) {
        for constant in &spec.constants {
            debug!(
                "{} has type {}",
                constant,
                self.unifier.get_type(self.var_lookup[&constant._id])
            );
        }
        for input in &spec.inputs {
            debug!(
                "{} has type {}",
                input,
                self.unifier.get_type(self.var_lookup[&input._id])
            );
        }
        for output in &spec.outputs {
            debug!(
                "{} has type {}",
                output,
                self.unifier.get_type(self.var_lookup[&output._id])
            );
        }
    }
}

enum ConstraintOrType {
    Type(Ty),
    Constraint(GenericTypeConstraint),
}

/// We have two types of constraints, equality constraints and subtype constraints
struct Unifier {
    /// union-find data structure representing the current state of the unification
    table: InPlaceUnificationTable<InferVar>,
    /// constraints associated with inference variables, if any
    constraints: HashMap<InferVar, GenericTypeConstraint>,
}

impl Unifier {
    fn new() -> Unifier {
        Unifier {
            table: UnificationTable::new(),
            constraints: HashMap::new(),
        }
    }

    fn new_var(&mut self) -> InferVar {
        self.table.new_key(None)
    }

    fn add_equality(&mut self, left: Ty, right: Ty) -> InferResult {
        debug!("unify ty ty {} {}", left, right);
        assert!(left != right);

        match (left, right) {
            (Ty::Infer(l), Ty::Infer(r)) => self
                .table
                .unify_var_var(l, r)
                .map_err(|(ty_l, ty_r)| format!("cannot merge types {} and {}", ty_l, ty_r)),
            (Ty::Infer(n), ty) => self.unify_var_type(n, ty),
            (ty, Ty::Infer(n)) => self.unify_var_type(n, ty),
            (_, _) => unreachable!(),
        }
    }

    fn unify_var_type(&mut self, var: InferVar, ty: Ty) -> InferResult {
        debug!("unify var ty {} {}", var, ty);
        match &ty {
            Ty::Infer(_) => unreachable!(),
            _ => {}
        }
        if self.check_occurrence(var, &ty) {
            return Err(format!("unification impossible, `var` occurs in `ty`"));
        }
        match self.table.unify_var_value(var, Some(ty)) {
            Ok(_) => Ok(()),
            Err((ty_l, ty_r)) => {
                if self.types_equal_rec(&ty_l, &ty_r) {
                    Ok(())
                } else {
                    Err(format!("cannot merge types {} and {}", ty_l, ty_r))
                }
            }
        }
    }

    /// Checks if `var` occurs in `ty`
    fn check_occurrence(&mut self, var: InferVar, ty: &Ty) -> bool {
        trace!("check occurrence {} {}", var, ty);
        match ty {
            Ty::Infer(t) => self.table.unioned(var, *t),
            Ty::EventStream(ty) => self.check_occurrence(var, &ty),
            Ty::TimedStream(_) => unimplemented!(),
            Ty::Tuple(_) => unimplemented!(),
            _ => false,
        }
    }

    fn types_equal_rec(&mut self, left: &Ty, right: &Ty) -> bool {
        println!("comp {} {}", left, right);
        match (left, right) {
            (&Ty::Infer(l), &Ty::Infer(r)) => {
                if self.table.unioned(l, r) {
                    true
                } else {
                    // try to unify values
                    self.table.unify_var_var(l, r).is_ok()
                }
            }
            (Ty::Option(l), Ty::Option(r)) => self.types_equal_rec(l, r),
            (Ty::EventStream(l), Ty::EventStream(r)) => self.types_equal_rec(l, r),
            (l, r) => l == r,
        }
    }

    fn add_constraint(&mut self, var: InferVar, constr: GenericTypeConstraint) -> InferResult {
        debug!("constraint {} {}", var, constr);
        if let Some(&other) = self.constraints.get(&var) {
            if other != constr {
                return Err(format!(
                    "unsatisfiable conjunction of constraint {} and {}",
                    other, constr
                ));
            }
        } else {
            self.constraints.insert(var, constr);
        }
        Ok(())
    }

    fn get_type(&mut self, var: InferVar) -> Ty {
        match self.table.probe_value(var) {
            None => Ty::Error,
            Some(Ty::Infer(other)) => self.get_type(other),
            Some(Ty::EventStream(ty)) => {
                let inner = match *ty {
                    Ty::Infer(other) => self.get_type(other),
                    ref t => t.clone(),
                };
                Ty::EventStream(Box::new(inner))
            }
            Some(t) => t.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct InferVar(u32);

// used in UnifcationTable
impl UnifyKey for InferVar {
    type Value = Option<Ty>;
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

impl EqUnifyValue for Ty {}

impl std::fmt::Display for InferVar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

type InferResult = Result<(), String>;

#[cfg(test)]
mod tests {

    use super::*;
    use analysis::id_assignment::*;
    use analysis::naming::*;
    use analysis::reporting::Handler;
    use parse::*;
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
    fn simple_constant() {
        let spec = "constant c: Float32 := 2.0";
        assert_eq!(0, num_type_errors(spec));
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
}
