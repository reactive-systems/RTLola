use super::super::ast::*;
use super::super::LolaSpec;
use ast_node::NodeId;

pub(crate) fn assign_ids(spec: &mut LolaSpec) {
    let mut free_id = 0;
    let mut next_id = || {
        let res = free_id;
        free_id += 1;
        NodeId::from_u32(res)
    };

    for mut c in &mut spec.constants {
        assert_eq!(c._id, NodeId::DUMMY, "Ids already assigned.");
        c._id = next_id();
        if let Some(ref mut t) = c.ty {
            t._id = next_id();
        }
    }
    for mut i in &mut spec.inputs {
        assert_eq!(i._id, NodeId::DUMMY, "Ids already assigned.");
        i._id = next_id();
        i.ty._id = next_id();
    }
    for mut o in &mut spec.outputs {
        assert_eq!(o._id, NodeId::DUMMY, "Ids already assigned.");
        o._id = next_id();
        if let Some(ref mut t) = o.ty {
            t._id = next_id();
        }
        assign_ids_expr(&mut o.expression, &mut next_id);
    }
    for mut t in &mut spec.trigger {
        assert_eq!(t._id, NodeId::DUMMY, "Ids already assigned.");
        t._id = next_id();
        assign_ids_expr(&mut t.expression, &mut next_id);
    }
}

fn assign_ids_expr<E>(exp: &mut Expression, next_id: &mut E)
where
    E: FnMut() -> NodeId,
{
    exp._id = next_id();
    match &mut exp.kind {
        ExpressionKind::Lit(_) => {}
        ExpressionKind::Ident(_) => {}
        ExpressionKind::Default(lhs, rhs) => {
            assign_ids_expr(lhs, next_id);
            assign_ids_expr(rhs, next_id);
        }
        ExpressionKind::Lookup(inst, offset, _winop) => {
            inst.arguments
                .iter_mut()
                .for_each(|e| assign_ids_expr(e, next_id));
            match offset {
                Offset::DiscreteOffset(expr) => assign_ids_expr(expr, next_id),
                Offset::RealTimeOffset(expr, _) => assign_ids_expr(expr, next_id),
            }
        }
        ExpressionKind::Binary(_, lhs, rhs) => {
            assign_ids_expr(lhs, next_id);
            assign_ids_expr(rhs, next_id)
        }
        ExpressionKind::Unary(_, operand) => assign_ids_expr(operand, next_id),
        ExpressionKind::Ite(cond, cons, alt) => {
            assign_ids_expr(cond, next_id);
            assign_ids_expr(cons, next_id);
            assign_ids_expr(alt, next_id)
        }
        ExpressionKind::ParenthesizedExpression(_, e, _) => assign_ids_expr(e, next_id),
        ExpressionKind::MissingExpression() => {}
        ExpressionKind::Tuple(exprs) => exprs.iter_mut().for_each(|e| assign_ids_expr(e, next_id)),
        ExpressionKind::Function(_, args) => {
            args.iter_mut().for_each(|e| assign_ids_expr(e, next_id))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::ast::Input;
    use super::super::super::parse::Ident;
    use super::*;
    use ast_node::Span;

    fn get_id_o(s: Option<&Output>) -> NodeId {
        if let Some(o) = s {
            if let Some(ref ty) = o.ty {
                ty._id
            } else {
                panic!("Assigning ids must not remove types!")
            }
        } else {
            panic!("Assigning ids must not remove streams!")
        }
    }
    fn get_id_c(s: Option<&Constant>) -> NodeId {
        if let Some(o) = s {
            if let Some(ref ty) = o.ty {
                ty._id
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
            _id: NodeId::DUMMY,
            name: Ident::new(String::from("Something"), span()),
            ty: Type::new_simple(String::from("something"), span()),
            _span: span(),
        }
    }
    fn constant() -> Constant {
        Constant {
            _id: NodeId::DUMMY,
            name: ident(),
            ty: Some(ty()),
            literal: Literal::new_bool(false, span()),
            _span: span(),
        }
    }
    fn output(expr: Expression) -> Output {
        Output {
            _id: NodeId::DUMMY,
            name: ident(),
            ty: Some(ty()),
            params: Vec::new(),
            template_spec: None,
            expression: expr,
            _span: span(),
        }
    }

    #[test]
    fn assign_atomic() {
        let mut spec = LolaSpec::new();
        spec.inputs.push(input());
        assign_ids(&mut spec);
        assert_ne!(spec.inputs[0]._id, NodeId::DUMMY);
        assert_ne!(spec.inputs[0].ty._id, NodeId::DUMMY);
    }

    #[test]
    fn assign_different_one_stream() {
        let mut spec = LolaSpec::new();
        spec.inputs.push(input());
        assign_ids(&mut spec);
        assert_ne!(spec.inputs[0].ty._id, spec.inputs[0]._id);
    }

    #[test]
    fn assign_different_several_streams() {
        let mut spec = LolaSpec::new();
        spec.inputs.push(input());
        spec.inputs.push(input());
        spec.constants.push(constant());
        let expr = Expression::new(ExpressionKind::Ident(ident()), span());
        spec.outputs.push(output(expr));
        assign_ids(&mut spec);
        let mut v = vec![
            spec.inputs[0].ty._id,
            spec.inputs[0]._id,
            spec.inputs[1].ty._id,
            spec.inputs[1]._id,
            get_id_c(spec.constants.get(0)),
            spec.constants[0]._id,
            get_id_o(spec.outputs.get(0)),
            spec.outputs[0]._id,
            spec.outputs[0].expression._id,
        ];
        v.dedup();
        assert_eq!(v.len(), 9, "Some ids occur multiple times.");
        assert!(
            v.iter().all(|id| *id != NodeId::DUMMY),
            "No node should have a dummy id anymore."
        );
    }

    #[test]
    #[should_panic]
    fn already_assigned() {
        let mut spec = LolaSpec::new();
        let mut input = input();
        input._id = NodeId::from_u32(42);
        spec.inputs.push(input);
        // Should panic:
        assign_ids(&mut spec);
    }

    #[test]
    fn assign_expr() {
        let mut spec = LolaSpec::new();
        let lhs = Expression::new(ExpressionKind::Ident(ident()), span());
        let rhs = Expression::new(ExpressionKind::Ident(ident()), span());
        let expr = Expression::new(
            ExpressionKind::Binary(BinOp::Div, Box::new(lhs), Box::new(rhs)),
            span(),
        );
        spec.outputs.push(output(expr));
        assign_ids(&mut spec);
        let mut v = vec![
            get_id_o(spec.outputs.get(0)),
            spec.outputs[0]._id,
            spec.outputs[0].expression._id,
        ];
        if let ExpressionKind::Binary(BinOp::Div, ref lhs, ref rhs) =
            spec.outputs[0].expression.kind
        {
            v.push(rhs._id);
            v.push(lhs._id);
        } else {
            panic!("Assigning ids must not change the ast in any other way.")
        }
        v.dedup();

        assert_eq!(v.len(), 5, "Some ids occur multiple times.");
        assert!(
            v.iter().all(|id| *id != NodeId::DUMMY),
            "No node should have a dummy id anymore."
        );
    }
}
