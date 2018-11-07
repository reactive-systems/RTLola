
use super::super::LolaSpec;
use super::super::ast::*;

pub(crate) fn assign_ids(spec: &mut LolaSpec) {
    let mut free_id = 0;
    let mut next_id = || {
        let res = free_id;
        free_id += 1;
        NodeId::from_u32(res)
    };

    for mut c in &mut spec.constants {
        assert_eq!(c.id, NodeId::DUMMY, "Ids already assigned.");
        c.id = next_id();
        if let Some(ref mut t) = c.ty {
            t.id = next_id();
        }
    }
    for mut i in &mut spec.inputs {
        assert_eq!(i.id, NodeId::DUMMY, "Ids already assigned.");
        i.id = next_id();
        i.ty.id = next_id();
    }
    for mut o in &mut spec.outputs {
        assert_eq!(o.id, NodeId::DUMMY, "Ids already assigned.");
        o.id = next_id();
        if let Some(ref mut t) = o.ty {
            t.id = next_id();
        }
        assign_ids_expr(&mut o.expression, &mut next_id);
    }
    for mut t in &mut spec.trigger {
        assert_eq!(t.id, NodeId::DUMMY, "Ids already assigned.");
        t.id = next_id();
        assign_ids_expr(&mut t.expression, &mut next_id);
    }
}

fn assign_ids_expr<E>(exp: &mut Expression, next_id: &mut E) where E: FnMut() -> NodeId {
    exp.id = next_id();
    match &mut exp.kind {
        ExpressionKind::Lit(_) => {},
        ExpressionKind::Ident(_) => {},
        ExpressionKind::Default(lhs, rhs) => {
            assign_ids_expr(lhs, next_id);
            assign_ids_expr(rhs, next_id);
        },
        ExpressionKind::Lookup(inst, offset, _winop) => {
            inst.arguments.iter_mut().for_each(|e| assign_ids_expr(e, next_id));
            match offset {
                Offset::DiscreteOffset(expr) => assign_ids_expr(expr, next_id),
                Offset::RealTimeOffset(expr, _) => assign_ids_expr(expr, next_id),
            }
        }
        ExpressionKind::Binary(_, lhs, rhs) => {
            assign_ids_expr(lhs, next_id);
            assign_ids_expr(rhs, next_id)
        },
        ExpressionKind::Unary(_, operand) => assign_ids_expr(operand, next_id),
        ExpressionKind::Ite(cond, cons, alt) => {
            assign_ids_expr(cond, next_id);
            assign_ids_expr(cons, next_id);
            assign_ids_expr(alt, next_id)
        }
        ExpressionKind::ParenthesizedExpression(_, e, _) => {
            assign_ids_expr(e, next_id)
        }
        ExpressionKind::MissingExpression() => {},
        ExpressionKind::Tuple(exprs) =>
            exprs.iter_mut().for_each(|e| assign_ids_expr(e, next_id)),
        ExpressionKind::Function(_, args) =>
            args.iter_mut().for_each(|e| assign_ids_expr(e, next_id)),
    }
}

