use crate::ast::*;
use crate::parse::NodeId;

pub(crate) fn assign_ids(spec: &mut LolaAst) {
    let mut free_id = 0;
    let mut next_id = || {
        let res = free_id;
        free_id += 1;
        NodeId::from_u32(res)
    };

    for td in &mut spec.type_declarations {
        assert_eq!(td.id, NodeId::DUMMY, "Ids already assigned.");
        td.id = next_id();
        for field in &mut td.fields {
            field.id = next_id();
            assign_ids_type(&mut field.ty, &mut next_id);
        }
    }
    for c in &mut spec.constants {
        assert_eq!(c.id, NodeId::DUMMY, "Ids already assigned.");
        c.id = next_id();
        if let Some(ref mut t) = c.ty {
            assign_ids_type(t, &mut next_id);
        }
        assign_ids_literal(&mut c.literal, &mut next_id);
    }
    for i in &mut spec.inputs {
        assert_eq!(i.id, NodeId::DUMMY, "Ids already assigned.");
        i.id = next_id();
        assign_ids_type(&mut i.ty, &mut next_id);
        for param in &mut i.params {
            assign_ids_parameter(param, &mut next_id);
        }
    }
    for o in &mut spec.outputs {
        assert_eq!(o.id, NodeId::DUMMY, "Ids already assigned.");
        o.id = next_id();
        assign_ids_type(&mut o.ty, &mut next_id);
        assign_ids_extend(&mut o.extend, &mut next_id);

        for param in &mut o.params {
            assign_ids_parameter(param, &mut next_id);
        }
        if let Some(ref mut ts) = o.template_spec {
            assign_ids_template_spec(ts, &mut next_id);
        }
        if let Some(ref mut ts) = o.termination {
            assign_ids_expr(ts, &mut next_id);
        }
        assign_ids_expr(&mut o.expression, &mut next_id);
    }
    for t in &mut spec.trigger {
        assert_eq!(t.id, NodeId::DUMMY, "Ids already assigned.");
        t.id = next_id();
        assign_ids_expr(&mut t.expression, &mut next_id);
    }
}

fn assign_ids_invoke_spec<E>(ts: &mut InvokeSpec, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    ts.id = next_id();
    assign_ids_expr(&mut ts.target, next_id);
    if let Some(ref mut cond) = ts.condition {
        assign_ids_expr(cond, next_id);
    }
}
fn assign_ids_extend_spec<E>(ts: &mut ExtendSpec, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    ts.id = next_id();
    assign_ids_expr(&mut ts.target, next_id);
}
fn assign_ids_terminate_spec<E>(ts: &mut TerminateSpec, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    ts.id = next_id();
    assign_ids_expr(&mut ts.target, next_id);
}

fn assign_ids_template_spec<E>(ts: &mut TemplateSpec, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    ts.id = next_id();
    if let Some(ref mut inv) = ts.inv {
        assign_ids_invoke_spec(inv, next_id);
    }
    if let Some(ref mut ext) = ts.ext {
        assign_ids_extend_spec(ext, next_id);
    }
    if let Some(ref mut ter) = ts.ter {
        assign_ids_terminate_spec(ter, next_id);
    }
}

fn assign_ids_parameter<E>(param: &mut Parameter, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    param.id = next_id();
    assign_ids_type(&mut param.ty, next_id);
}

fn assign_ids_literal<E>(lit: &mut Literal, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    lit.id = next_id();
}

fn assign_ids_type<E>(ty: &mut Type, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    ty.id = next_id();
    if let TypeKind::Tuple(ref mut elements) = ty.kind {
        for element in elements.iter_mut() {
            assign_ids_type(element, next_id);
        }
    }
}

fn assign_ids_extend<E>(extend: &mut ActivationCondition, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    extend.id = next_id();
    if let Some(expr) = extend.expr.as_mut() {
        assign_ids_expr(expr, next_id);
    }
}

fn assign_ids_expr<E>(exp: &mut Expression, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    exp.id = next_id();
    match &mut exp.kind {
        ExpressionKind::Lit(lit) => {
            assign_ids_literal(lit, next_id);
        }
        ExpressionKind::Ident(_) | ExpressionKind::MissingExpression => {}
        ExpressionKind::StreamAccess(expr, _) => {
            assign_ids_expr(expr, next_id);
        }
        ExpressionKind::Default(lhs, rhs) | ExpressionKind::Binary(_, lhs, rhs) => {
            assign_ids_expr(lhs, next_id);
            assign_ids_expr(rhs, next_id);
        }
        ExpressionKind::Offset(expr, _) => {
            assign_ids_expr(expr, next_id);
        }
        ExpressionKind::SlidingWindowAggregation { expr, duration, .. } => {
            assign_ids_expr(expr, next_id);
            assign_ids_expr(duration, next_id);
        }
        ExpressionKind::Unary(_, operand) => assign_ids_expr(operand, next_id),
        ExpressionKind::Ite(cond, cons, alt) => {
            assign_ids_expr(cond, next_id);
            assign_ids_expr(cons, next_id);
            assign_ids_expr(alt, next_id)
        }
        ExpressionKind::ParenthesizedExpression(open, e, close) => {
            assign_ids_expr(e, next_id);
            if let Some(ref mut paren) = open {
                paren.id = next_id();
            }
            if let Some(ref mut paren) = close {
                paren.id = next_id();
            }
        }
        ExpressionKind::Tuple(exprs) => exprs.iter_mut().for_each(|e| assign_ids_expr(e, next_id)),
        ExpressionKind::Function(_, types, args) => {
            types.iter_mut().for_each(|ty| assign_ids_type(ty, next_id));
            args.iter_mut().for_each(|e| assign_ids_expr(e, next_id));
        }
        ExpressionKind::Field(expr, _) => {
            assign_ids_expr(expr, next_id);
        }
        ExpressionKind::Method(expr, _, types, args) => {
            assign_ids_expr(expr, next_id);
            types.iter_mut().for_each(|ty| assign_ids_type(ty, next_id));
            args.iter_mut().for_each(|e| assign_ids_expr(e, next_id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Input;
    use crate::parse::{Ident, Span};

    fn get_id_o(s: Option<&Output>) -> NodeId {
        if let Some(o) = s {
            o.ty.id
        } else {
            panic!("Assigning ids must not remove streams!")
        }
    }
    fn get_id_c(s: Option<&Constant>) -> NodeId {
        if let Some(o) = s {
            if let Some(ref ty) = o.ty {
                ty.id
            } else {
                panic!("Assigning ids must not remove types!")
            }
        } else {
            panic!("Assigning ids must not remove streams!")
        }
    }

    fn span() -> Span {
        Span { start: 0, end: 0 }
    }
    fn ty() -> Type {
        Type::new_simple(String::from("something"), span())
    }
    fn ident() -> Ident {
        Ident::new(String::from("Something"), span())
    }
    fn input() -> Input {
        Input {
            id: NodeId::DUMMY,
            name: Ident::new(String::from("Something"), span()),
            params: Vec::new(),
            ty: Type::new_simple(String::from("something"), span()),
            span: span(),
        }
    }
    fn constant() -> Constant {
        Constant {
            id: NodeId::DUMMY,
            name: ident(),
            ty: Some(ty()),
            literal: Literal::new_bool(false, span()),
            span: span(),
        }
    }
    fn output(expr: Expression) -> Output {
        Output {
            id: NodeId::DUMMY,
            name: ident(),
            ty: ty(),
            extend: ActivationCondition { expr: None, id: NodeId::DUMMY, span: span() },
            params: Vec::new(),
            template_spec: None,
            termination: None,
            expression: expr,
            span: span(),
        }
    }

    #[test]
    fn assign_atomic() {
        let mut spec = LolaAst::new();
        spec.inputs.push(input());
        assign_ids(&mut spec);
        assert_ne!(spec.inputs[0].id, NodeId::DUMMY);
        assert_ne!(spec.inputs[0].ty.id, NodeId::DUMMY);
    }

    #[test]
    fn assign_different_one_stream() {
        let mut spec = LolaAst::new();
        spec.inputs.push(input());
        assign_ids(&mut spec);
        assert_ne!(spec.inputs[0].ty.id, spec.inputs[0].id);
    }

    #[test]
    fn assign_different_several_streams() {
        let mut spec = LolaAst::new();
        spec.inputs.push(input());
        spec.inputs.push(input());
        spec.constants.push(constant());
        let expr = Expression::new(ExpressionKind::Ident(ident()), span());
        spec.outputs.push(output(expr));
        assign_ids(&mut spec);
        let mut v = vec![
            spec.inputs[0].ty.id,
            spec.inputs[0].id,
            spec.inputs[1].ty.id,
            spec.inputs[1].id,
            get_id_c(spec.constants.get(0)),
            spec.constants[0].id,
            get_id_o(spec.outputs.get(0)),
            spec.outputs[0].id,
            spec.outputs[0].expression.id,
        ];
        v.dedup();
        assert_eq!(v.len(), 9, "Some ids occur multiple times.");
        assert!(v.iter().all(|id| *id != NodeId::DUMMY), "No node should have a dummy id anymore.");
    }

    #[test]
    #[should_panic]
    fn already_assigned() {
        let mut spec = LolaAst::new();
        let mut input = input();
        input.id = NodeId::from_u32(42);
        spec.inputs.push(input);
        // Should panic:
        assign_ids(&mut spec);
    }

    #[test]
    fn assign_expr() {
        let mut spec = LolaAst::new();
        let lhs = Expression::new(ExpressionKind::Ident(ident()), span());
        let rhs = Expression::new(ExpressionKind::Ident(ident()), span());
        let expr = Expression::new(ExpressionKind::Binary(BinOp::Div, Box::new(lhs), Box::new(rhs)), span());
        spec.outputs.push(output(expr));
        assign_ids(&mut spec);
        let mut v = vec![get_id_o(spec.outputs.get(0)), spec.outputs[0].id, spec.outputs[0].expression.id];
        if let ExpressionKind::Binary(BinOp::Div, ref lhs, ref rhs) = spec.outputs[0].expression.kind {
            v.push(rhs.id);
            v.push(lhs.id);
        } else {
            panic!("Assigning ids must not change the ast in any other way.")
        }
        v.dedup();

        assert_eq!(v.len(), 5, "Some ids occur multiple times.");
        assert!(v.iter().all(|id| *id != NodeId::DUMMY), "No node should have a dummy id anymore.");
    }
}
