use super::super::common::BuiltinType;
use super::super::naming::{Declaration, DeclarationTable};
use super::super::AnalysisError;
use ast::*;
use std::collections::HashMap;
extern crate ast_node;
use super::candidates::*;
use super::type_error::*;
use super::TypeCheckResult;
use ast_node::{AstNode, NodeId, Span};

// TODO: Remove?
#[derive(Debug)]
pub struct TypeTable {
    map: HashMap<NodeId, Candidates>,
}

impl TypeTable {
    fn new() -> Self {
        TypeTable {
            map: HashMap::new(),
        }
    }

    fn meet(&mut self, target: NodeId, candidates: &Candidates) -> Candidates {
        let res = {
            let current = self.map.get(&target).unwrap_or(&Candidates::Any);
            current.meet(candidates)
        }; // Drops `current`, such that there no burrow to `self.map` anymore.
        self.map.insert(target, res.clone());
        res
    }

    fn get(&self, target: NodeId) -> Option<&Candidates> {
        self.map.get(&target)
    }
}

pub(crate) struct TypeChecker<'a> {
    declarations: &'a DeclarationTable<'a>,
    spec: &'a LolaSpec,
    tt: TypeTable,
    errors: Vec<Box<AnalysisError<'a> + 'a>>,
}

impl<'a> TypeChecker<'a> {
    pub(crate) fn new(dt: &'a DeclarationTable, spec: &'a LolaSpec) -> TypeChecker<'a> {
        TypeChecker {
            declarations: dt,
            spec,
            tt: TypeTable::new(),
            errors: Vec::new(),
        }
    }

    pub(crate) fn check(self) -> TypeCheckResult<'a> {
        unimplemented!()
    }

    fn get_candidates(&mut self, e: &'a Expression) -> Option<Candidates> {
        match e.kind {
            ExpressionKind::Lit(ref lit) => self.reg_cand(*e.id(), Candidates::from(lit)),
            ExpressionKind::Ident(ref i) => self
                .get_ty_from_decl(*e.id(), e)
                .and_then(|c| self.reg_cand(*e.id(), c)),
            ExpressionKind::Default(ref expr, ref dft) => {
                let res = self.get_candidates(expr)?.meet(&self.get_candidates(dft)?);
                if res.is_none() {
                    self.reg_error(TypeError::IncompatibleTypes(e, String::from("A potentially undefined expression and its default need matching types.")))
                } else {
                    self.reg_cand(*e.id(), res)
                }
            }
            ExpressionKind::Lookup(ref inst, ref offset, ref op) => {
                self.check_offset(offset, e); // Return value does not matter.
                let target_stream_type = self.check_stream_instance(inst, e)?;

                if op.is_none() {
                    return self.reg_cand(*e.id(), target_stream_type);
                }
                let op = op.unwrap();
                // For all window operations, the source stream needs to be numeric.
                if target_stream_type.is_numeric() {
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
                let lhs_type = self.get_candidates(lhs)?;
                let rhs_type = self.get_candidates(rhs)?;
                let meet_type = lhs_type.meet(&rhs_type);
                let res = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow
                        if meet_type.is_numeric() =>
                    {
                        Some(meet_type)
                    }
                    BinOp::And | BinOp::Or if meet_type.is_logic() => Some(meet_type),
                    BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt
                        if meet_type.is_none() =>
                    {
                        Some(Candidates::Concrete(BuiltinType::Bool))
                    }
                    _ => None,
                };
                match res {
                    Some(cand) => self.reg_cand(*e.id(), cand),
                    None => self.reg_error(TypeError::IncompatibleTypes(
                        e,
                        format!(
                            "Binary operator {} applied to types {} and {}.",
                            op, lhs, rhs
                        ),
                    )),
                }
            }
            ExpressionKind::Unary(ref operator, ref operand) => unimplemented!(),
            ExpressionKind::Ite(ref cond, ref cons, ref alt) => unimplemented!(),
            ExpressionKind::ParenthesizedExpression(_, ref expr, _) => unimplemented!(),
            ExpressionKind::MissingExpression() => unimplemented!(),
            ExpressionKind::Tuple(ref exprs) => unimplemented!(),
            ExpressionKind::Function(ref kind, ref args) => unimplemented!(),
        }
    }

    fn check_offset(&mut self, offset: &'a Offset, origin: &'a Expression) -> bool {
        let (expr, is_discrete) = match offset {
            Offset::DiscreteOffset(e) => (e, true),
            Offset::RealTimeOffset(e, _) => (e, false),
        };

        let cand = self.get_candidates(expr);
        if cand.is_none() {
            return true;
        } // Error is already reported, so pretend everything is dandy.
        let cand = cand.unwrap();

        let err = match (is_discrete, cand.is_numeric(), cand.is_integer()) {
            (true, _, true) => return true,
            (true, _, false) => TypeError::IncompatibleTypes(
                origin,
                String::from("A discrete offset must be an integer value."),
            ),
            (false, true, _) => return true,
            (false, false, _) => TypeError::IncompatibleTypes(
                origin,
                String::from("A real-time offset must be a numeric value."),
            ),
        };
        self.reg_error(err);
        false
    }

    fn check_stream_instance(
        &mut self,
        inst: &'a StreamInstance,
        origin: &'a Expression,
    ) -> Option<Candidates> {
        let inst_candidates = self.get_ty_from_decl(*inst.id(), origin)?;
        if let Some(Declaration::Out(ref o)) = self.declarations.get(inst.id()) {
            if o.params.len() != inst.arguments.len() {
                let msg = format!(
                    "Expected {}, but got {}.",
                    o.params.len(),
                    inst.arguments.len()
                );
                return self.reg_error(TypeError::UnexpectedNumberOfArguments(origin, msg));
            }
            let params = o.params.iter().map(|p| Candidates::from(&p.ty));
            let arguments: Vec<Candidates> = inst
                .arguments
                .iter()
                .flat_map(|e: &Box<Expression>| self.get_candidates(e))
                .collect();

            if arguments.len() < inst.arguments.len() {
                for (exp, was) in params.zip(arguments) {
                    let res = exp.meet(&was);
                    if res.is_none() && !exp.is_none() && !was.is_none() {
                        self.reg_error(TypeError::IncompatibleTypes(origin, String::from("")));
                    }
                }
            } // The error was already reported, so just go on as if everything is dandy.
            self.reg_cand(*origin.id(), inst_candidates)
        } else {
            self.reg_error(TypeError::UnknownIdentifier(inst))
        }
    }

    fn get_ty_from_decl(&mut self, nid: NodeId, ident: &'a Expression) -> Option<Candidates> {
        // Note that nested self calls do not work (, yet? See https://internals.rust-lang.org/t/accepting-nested-method-calls-with-an-mut-self-receiver/4588).
        // https://stackoverflow.com/questions/37986640/cannot-obtain-a-mutable-reference-when-iterating-a-recursive-structure-cannot-b
        // This solution is at least accepted by the borrow checker and not entirely unreadable, so ¯\_(ツ)_/¯.
        match self.declarations.get(&nid) {
            Some(Declaration::Const(ref c)) => {
                let cand = self.get_from_tt(*c.id());
                self.reg_cand(nid, cand)
            }
            Some(Declaration::In(ref i)) => {
                let cand = self.get_from_tt(*i.id());
                self.reg_cand(nid, cand)
            }
            Some(Declaration::Out(ref o)) => {
                let cand = self.get_from_tt(*o.id());
                self.reg_cand(nid, cand)
            }
            Some(Declaration::UserDefinedType(td)) => {
                let cand = self.get_from_tt(*td.id());
                self.reg_cand(nid, cand)
            }
            Some(Declaration::BuiltinType(ref b)) => self.reg_cand(nid, Candidates::from(b)),
            Some(Declaration::Param(ref p)) => self.reg_cand(nid, Candidates::from(&p.ty)),
            None => self.reg_error(TypeError::UnknownIdentifier(ident)),
        }
    }

    fn get_from_tt(&self, nid: NodeId) -> Candidates {
        self.tt
            .get(nid)
            .map(|t| t.clone())
            .unwrap_or(Candidates::Any)
    }

    fn reg_error(&mut self, e: TypeError<'a>) -> Option<Candidates> {
        self.errors.push(Box::new(e));
        None
    }

    fn reg_cand(&mut self, nid: NodeId, cand: Candidates) -> Option<Candidates> {
        Some(self.tt.meet(nid, &cand))
    }
}
