use analysis::naming::{Declaration, DeclarationTable};
use analysis::reporting::Handler;
use ast_node::NodeId;
use crate::ast::LolaSpec;
use crate::ty::{GenricTypeConstraint, Ty};
use std::collections::HashMap;

pub(crate) struct TypeAnalysis<'a> {
    handler: &'a Handler,
    declarations: &'a DeclarationTable<'a>,
    equations: Vec<TypeEquation>,
}

impl<'a> TypeAnalysis<'a> {
    pub(crate) fn new(
        handler: &'a Handler,
        declarations: &'a DeclarationTable<'a>,
    ) -> TypeAnalysis<'a> {
        TypeAnalysis {
            handler,
            declarations,
            equations: Vec::new(),
        }
    }

    pub(crate) fn check(&mut self, spec: &LolaSpec) {
        self.add_equations(spec);

        debug!("Equations:");
        for equation in &self.equations {
            debug!("{}", equation);
        }

        let subst = self.unify_equations();
        debug!("Substitutions:\n{}", subst);

        unimplemented!();
    }

    fn add_equations(&mut self, spec: &LolaSpec) {
        for constant in &spec.constants {
            // generate constraint in case a type is annotated
            if let Some(type_name) = constant.ty.as_ref() {
                if let Some(ty) = self.declarations.get(&type_name._id) {
                    match ty {
                        Declaration::Type(ty) => self.equations.push(TypeEquation::new_concrete(
                            constant._id,
                            (*ty).clone(),
                            constant._id,
                        )),
                        _ => unreachable!(),
                    }
                }
            }
            // generate constraint from literal
            use ast::LitKind::*;
            match constant.literal.kind {
                Str(_) | RawStr(_) => self.equations.push(TypeEquation::new_concrete(
                    constant._id,
                    Ty::String,
                    constant.literal._id,
                )),
                Bool(_) => self.equations.push(TypeEquation::new_concrete(
                    constant._id,
                    Ty::Bool,
                    constant.literal._id,
                )),
                Int(_) => self.equations.push(TypeEquation::new_constraint(
                    constant._id,
                    GenricTypeConstraint::Integer,
                    constant.literal._id,
                )),
                Float(_) => self.equations.push(TypeEquation::new_constraint(
                    constant._id,
                    GenricTypeConstraint::FloatingPoint,
                    constant.literal._id,
                )),
            }
        }
        for input in &spec.inputs {
            if let Some(ty) = self.declarations.get(&input.ty._id) {
                match ty {
                    Declaration::Type(ty) => self.equations.push(TypeEquation::new_concrete(
                        input._id,
                        Ty::EventStream(Box::new((*ty).clone())),
                        input._id,
                    )),
                    _ => unreachable!(),
                }
            }
        }

        for output in &spec.outputs {
            // generate constraint in case a type is annotated
            if let Some(type_name) = output.ty.as_ref() {
                if let Some(ty) = self.declarations.get(&type_name._id) {
                    match ty {
                        Declaration::Type(ty) => self.equations.push(TypeEquation::new_concrete(
                            output._id,
                            Ty::EventStream(Box::new((*ty).clone())),
                            output._id,
                        )),
                        _ => unreachable!(),
                    }
                }
            }
            // generate constraint for expression
            unimplemented!();
        }
    }

    /// Computes the most general unifier.
    fn unify_equations(&mut self) -> Substitution {
        let mut substitution = Substitution::new();
        for equation in &self.equations {
            substitution.unify(&equation.lhs, &equation.rhs);
        }
        substitution
    }
}

struct Substitution {
    map: HashMap<NodeId, TypeRef>,
}

impl Substitution {
    fn new() -> Substitution {
        Substitution {
            map: HashMap::new(),
        }
    }

    /// Unify to types under the given substitution.
    /// Returns false if there is no substitution.
    fn unify(&mut self, left: &TypeRef, right: &TypeRef) -> bool {
        if left == right {
            return true;
        }
        use self::TypeRef::*;
        match (left, right) {
            (Symbolic(n), _) => self.unify_variable(n, right),
            (_, Symbolic(n)) => self.unify_variable(n, left),
            (Concrete(ty), Constraint(c)) => ty.satisfies(*c),
            _ => {
                println!("{} {}", left, right);
                unimplemented!()
            }
        };
        false
    }

    /// Unifies variable n with type def using current substitutions.
    /// Returns false if there is no substitution.
    fn unify_variable(&mut self, n: &NodeId, def: &TypeRef) -> bool {
        if self.map.contains_key(n) {
            let val = &self.map[n].clone();
            return self.unify(val, def);
        }
        if let TypeRef::Symbolic(t) = def {
            if self.map.contains_key(t) {
                let val = &self.map[t].clone();
                return self.unify(&TypeRef::Symbolic(*n), val);
            }
        }
        if self.occurs_check(n, def) {
            false
        } else {
            // update substitution
            self.map.insert(*n, def.clone());
            true
        }
    }

    /// Does the variable `n` occur anywhere inside def?
    fn occurs_check(&self, n: &NodeId, def: &TypeRef) -> bool {
        match def {
            TypeRef::Symbolic(t) if t == n => true,
            TypeRef::Symbolic(t) if self.map.contains_key(t) => {
                assert!(t != n);
                // recursion
                self.occurs_check(n, &self.map[t])
            }
            TypeRef::Concrete(Ty::EventStream(ty)) => {
                self.occurs_check(n, &TypeRef::Concrete(*ty.clone()))
            }
            TypeRef::Concrete(Ty::TimedStream(_)) => unimplemented!(),
            TypeRef::Concrete(Ty::Tuple(_)) => unimplemented!(),
            _ => false,
        }
    }
}

impl std::fmt::Display for Substitution {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for (id, ty) in self.map.iter() {
            writeln!(f, "{} -> {}", id, ty)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TypeRef {
    Symbolic(NodeId),
    Concrete(Ty),
    Constraint(GenricTypeConstraint),
}

/// Asserts that `lhs` == `rhs` due to `orig`
#[derive(Debug)]
struct TypeEquation {
    lhs: TypeRef,
    rhs: TypeRef,
    orig: NodeId,
}

impl std::fmt::Display for TypeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TypeRef::Symbolic(n) => write!(f, "t{}", n),
            TypeRef::Concrete(ty) => write!(f, "{}", ty),
            TypeRef::Constraint(c) => write!(f, "{{{}}}", c),
        }
    }
}

impl std::fmt::Display for TypeEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}\t{}\t{}", self.lhs, self.rhs, self.orig)
    }
}

impl TypeEquation {
    fn new_symbolic(lhs: NodeId, rhs: NodeId, orig: NodeId) -> TypeEquation {
        TypeEquation {
            lhs: TypeRef::Symbolic(lhs),
            rhs: TypeRef::Symbolic(rhs),
            orig,
        }
    }

    fn new_concrete(lhs: NodeId, rhs: Ty, orig: NodeId) -> TypeEquation {
        TypeEquation {
            lhs: TypeRef::Symbolic(lhs),
            rhs: TypeRef::Concrete(rhs),
            orig,
        }
    }

    fn new_constraint(lhs: NodeId, rhs: GenricTypeConstraint, orig: NodeId) -> TypeEquation {
        TypeEquation {
            lhs: TypeRef::Symbolic(lhs),
            rhs: TypeRef::Constraint(rhs),
            orig,
        }
    }
}

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
    fn simple_input() {
        let spec = "input i: Int8";
        assert_eq!(0, num_type_errors(spec));
    }
}
