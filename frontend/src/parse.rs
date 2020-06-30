//! This module contains the parser for the Lola Language.

use super::ast::*;
use crate::reporting::{Handler, LabeledSpan};
use crate::FrontendConfig;
use lazy_static::lazy_static;
use pest::iterators::{Pair, Pairs};
use pest::prec_climber::{Assoc, Operator, PrecClimber};
use pest::Parser;
use pest_derive::Parser;
use std::{cell::RefCell, path::PathBuf, rc::Rc};

#[derive(Parser)]
#[grammar = "lola.pest"]
pub(crate) struct LolaParser;

#[derive(Debug)]
pub(crate) struct RTLolaParser<'a, 'b> {
    content: &'a str,
    spec: RTLolaAst,
    handler: &'b Handler,
    config: FrontendConfig,
    node_id: RefCell<NodeId>,
}

lazy_static! {
    // precedence taken from C/C++: https://en.wikipedia.org/wiki/Operators_in_C_and_C++
    // Precedence climber can be used to build the AST, see https://pest-parser.github.io/book/ for more details
    static ref PREC_CLIMBER: PrecClimber<Rule> = {
        use self::Assoc::*;
        use self::Rule::*;

        PrecClimber::new(vec![
            Operator::new(Or, Left),
            Operator::new(And, Left),
            Operator::new(BitOr, Left),
            Operator::new(BitXor, Left),
            Operator::new(BitAnd, Left),
            Operator::new(Equal, Left) | Operator::new(NotEqual, Left),
            Operator::new(LessThan, Left) | Operator::new(LessThanOrEqual, Left) | Operator::new(MoreThan, Left) | Operator::new(MoreThanOrEqual, Left),
            Operator::new(ShiftLeft, Left) | Operator::new(ShiftRight, Left),
            Operator::new(Add, Left) | Operator::new(Subtract, Left),
            Operator::new(Multiply, Left) | Operator::new(Divide, Left) | Operator::new(Mod, Left),
            Operator::new(Power, Right),
            Operator::new(Dot, Left),
            Operator::new(OpeningBracket, Left),
        ])
    };
}

impl<'a, 'b> RTLolaParser<'a, 'b> {
    pub(crate) fn new(content: &'a str, handler: &'b Handler, config: FrontendConfig) -> Self {
        RTLolaParser { content, spec: RTLolaAst::new(), handler, config, node_id: RefCell::new(NodeId::new(0)) }
    }

    fn next_id(&self) -> NodeId {
        let res = *self.node_id.borrow();
        self.node_id.borrow_mut().0 += 1;
        res
    }

    pub(crate) fn parse(mut self) -> Result<RTLolaAst, pest::error::Error<Rule>> {
        let mut pairs = LolaParser::parse(Rule::Spec, self.content)?;
        assert!(pairs.clone().count() == 1, "Spec must not be empty.");
        let spec_pair = pairs.next().unwrap();
        assert!(spec_pair.as_rule() == Rule::Spec);
        for pair in spec_pair.into_inner() {
            match pair.as_rule() {
                Rule::ImportStmt => {
                    let import = self.parse_import(pair);
                    self.spec.imports.push(import);
                }
                Rule::ConstantStream => {
                    let constant = self.parse_constant(pair);
                    self.spec.constants.push(Rc::new(constant));
                }
                Rule::InputStream => {
                    let inputs = self.parse_inputs(pair);
                    self.spec.inputs.extend(inputs.into_iter().map(Rc::new));
                }
                Rule::OutputStream => {
                    let output = self.parse_output(pair);
                    self.spec.outputs.push(Rc::new(output));
                }
                Rule::Trigger => {
                    let trigger = self.parse_trigger(pair);
                    self.spec.trigger.push(Rc::new(trigger));
                }
                Rule::TypeDecl => {
                    let type_decl = self.parse_type_declaration(pair);
                    self.spec.type_declarations.push(type_decl);
                }
                Rule::EOI => {}
                _ => unreachable!(),
            }
        }
        Ok(self.spec)
    }

    fn parse_import(&self, pair: Pair<Rule>) -> Import {
        assert_eq!(pair.as_rule(), Rule::ImportStmt);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
        Import { name, id: self.next_id(), span }
    }

    /**
     * Transforms a `Rule::ConstantStream` into `Constant` AST node.
     * Panics if input is not `Rule::ConstantStream`.
     * The constant rule consists of the following tokens:
     * - `Rule::Ident`
     * - `Rule::Type`
     * - `Rule::Literal`
     */
    fn parse_constant(&self, pair: Pair<'_, Rule>) -> Constant {
        assert_eq!(pair.as_rule(), Rule::ConstantStream);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
        let ty = self.parse_type(pairs.next().expect("mismatch between grammar and AST"));
        let literal = self.parse_literal(pairs.next().expect("mismatch between grammar and AST"));
        Constant { id: self.next_id(), name, ty: Some(ty), literal, span }
    }

    /**
     * Transforms a `Rule::InputStream` into `Input` AST node.
     * Panics if input is not `Rule::InputStream`.
     * The input rule consists of non-empty sequences of following tokens:
     * - `Rule::Ident`
     * - (`Rule::ParamList`)?
     * - `Rule::Type`
     */
    fn parse_inputs(&self, pair: Pair<'_, Rule>) -> Vec<Input> {
        assert_eq!(pair.as_rule(), Rule::InputStream);
        let mut inputs = Vec::new();
        let mut pairs = pair.into_inner();
        while let Some(pair) = pairs.next() {
            let start = pair.as_span().start();
            let name = self.parse_ident(&pair);

            let mut pair = pairs.next().expect("mismatch between grammar and AST");
            let params = if let Rule::ParamList = pair.as_rule() {
                let res = self.parse_parameter_list(pair.into_inner());
                pair = pairs.next().expect("mismatch between grammar and AST");

                if !self.config.allow_parameters {
                    self.handler.error_with_span(
                        "Parameterization is disabled",
                        LabeledSpan::new(res[0].span, "found parameter", true),
                    )
                }

                res
            } else {
                Vec::new()
            };
            let end = pair.as_span().end();
            let ty = self.parse_type(pair);
            inputs.push(Input {
                id: self.next_id(),
                name,
                params: params.into_iter().map(Rc::new).collect(),
                ty,
                span: Span { start, end },
            })
        }

        assert!(!inputs.is_empty());
        inputs
    }

    /**
     * Transforms a `Rule::OutputStream` into `Output` AST node.
     * Panics if input is not `Rule::OutputStream`.
     * The output rule consists of the following tokens:
     * - `Rule::Ident`
     * - `Rule::Type`
     * - `Rule::Expr`
     */
    fn parse_output(&self, pair: Pair<'_, Rule>) -> Output {
        assert_eq!(pair.as_rule(), Rule::OutputStream);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));

        let mut pair = pairs.next().expect("mismatch between grammar and AST");
        let params = if let Rule::ParamList = pair.as_rule() {
            let res = self.parse_parameter_list(pair.into_inner());
            pair = pairs.next().expect("mismatch between grammar and AST");

            if !self.config.allow_parameters {
                self.handler.error_with_span(
                    "Parameterization is disabled",
                    LabeledSpan::new(res[0].span, "found parameter", true),
                )
            }

            res
        } else {
            Vec::new()
        };

        let ty = if let Rule::Type = pair.as_rule() {
            let ty = self.parse_type(pair);
            pair = pairs.next().expect("mismatch between grammar and AST");
            ty
        } else {
            Type::new_inferred(self.next_id())
        };

        // Parse the `@ [Expr]` part of output declaration
        let extend = if let Rule::ActivationCondition = pair.as_rule() {
            let span: Span = pair.as_span().into();
            let expr = self.build_expression_ast(pair.into_inner());
            pair = pairs.next().expect("mismatch between grammar and AST");
            ActivationCondition { expr: Some(expr), id: self.next_id(), span }
        } else {
            ActivationCondition { expr: None, id: self.next_id(), span: Span::unknown() }
        };

        let mut tspec = None;
        if let Rule::TemplateSpec = pair.as_rule() {
            tspec = Some(self.parse_template_spec(pair));
            pair = pairs.next().expect("mismatch between grammar and AST");
        };

        // Parse termination condition `close EXPRESSION`
        let termination = if let Rule::TerminateDecl = pair.as_rule() {
            let expr = pair.into_inner().next().expect("mismatch between grammar and AST");
            let expr = self.build_expression_ast(expr.into_inner());
            pair = pairs.next().expect("mismatch between grammar and AST");

            if !self.config.allow_parameters {
                self.handler.error_with_span(
                    "Parameterization is disabled",
                    LabeledSpan::new(expr.span, "found termination condition", true),
                )
            }
            if params.is_empty() {
                self.handler.error_with_span(
                    "Termination condition is only allowed for parameterized streams",
                    LabeledSpan::new(expr.span, "found termination condition", true),
                )
            }

            Some(expr)
        } else {
            None
        };

        // Parse expression
        let expression = self.build_expression_ast(pair.into_inner());
        Output {
            id: self.next_id(),
            name,
            ty,
            extend,
            params: params.into_iter().map(Rc::new).collect(),
            template_spec: tspec,
            termination,
            expression,
            span,
        }
    }

    fn parse_parameter_list(&self, param_list: Pairs<'_, Rule>) -> Vec<Parameter> {
        let mut params = Vec::new();
        for param_decl in param_list {
            assert_eq!(Rule::ParameterDecl, param_decl.as_rule());
            let span = param_decl.as_span().into();
            let mut decl = param_decl.into_inner();
            let name = self.parse_ident(&decl.next().expect("mismatch between grammar and AST"));
            let ty = if let Some(type_pair) = decl.next() {
                assert_eq!(Rule::Type, type_pair.as_rule());
                self.parse_type(type_pair)
            } else {
                Type::new_inferred(self.next_id())
            };
            params.push(Parameter { name, ty, id: self.next_id(), span });
        }
        params
    }

    fn parse_template_spec(&self, pair: Pair<'_, Rule>) -> TemplateSpec {
        let span = pair.as_span().into();
        let mut decls = pair.into_inner();
        let mut pair = decls.next();
        let mut rule = pair.as_ref().map(Pair::as_rule);

        let mut inv_spec = None;
        if let Some(Rule::InvokeDecl) = rule {
            inv_spec = Some(self.parse_inv_spec(pair.unwrap()));
            pair = decls.next();
            rule = pair.as_ref().map(Pair::as_rule);
        }
        let mut ext_spec = None;
        if let Some(Rule::ExtendDecl) = rule {
            ext_spec = Some(self.parse_ext_spec(pair.unwrap()));
            pair = decls.next();
            rule = pair.as_ref().map(Pair::as_rule);
        }
        let mut ter_spec = None;
        if let Some(Rule::TerminateDecl) = rule {
            let exp = pair.unwrap();
            let span_ter = exp.as_span().into();
            let expr = exp.into_inner().next().expect("mismatch between grammar and AST");
            let expr = self.build_expression_ast(expr.into_inner());
            ter_spec = Some(TerminateSpec { target: expr, id: self.next_id(), span: span_ter });
        }
        TemplateSpec { inv: inv_spec, ext: ext_spec, ter: ter_spec, id: self.next_id(), span }
    }

    fn parse_ext_spec(&self, ext_pair: Pair<'_, Rule>) -> ExtendSpec {
        let span_ext = ext_pair.as_span().into();
        let mut children = ext_pair.into_inner();

        let first_child = children.next().expect("mismatch between grammar and ast");
        let target = match first_child.as_rule() {
            Rule::Expr => self.build_expression_ast(first_child.into_inner()),
            _ => unreachable!(),
        };
        ExtendSpec { target, id: self.next_id(), span: span_ext }
    }

    fn parse_inv_spec(&self, inv_pair: Pair<'_, Rule>) -> InvokeSpec {
        let span_inv = inv_pair.as_span().into();
        let mut inv_children = inv_pair.into_inner();
        let expr_pair = inv_children.next().expect("mismatch between grammar and AST");
        let inv_target = self.build_expression_ast(expr_pair.into_inner());
        // Compute invocation condition:
        let mut is_if = false;
        let mut cond_expr = None;
        if let Some(inv_cond_pair) = inv_children.next() {
            is_if = match inv_cond_pair.as_rule() {
                Rule::InvokeIf => true,
                Rule::InvokeUnless => false,
                _ => unreachable!(),
            };
            let condition = inv_cond_pair.into_inner().next().expect("mismatch between grammar and AST");
            cond_expr = Some(self.build_expression_ast(condition.into_inner()))
        }
        InvokeSpec { condition: cond_expr, is_if, target: inv_target, id: self.next_id(), span: span_inv }
    }

    /**
     * Transforms a `Rule::Trigger` into `Trigger` AST node.
     * Panics if input is not `Rule::Trigger`.
     * The output rule consists of the following tokens:
     * - (`Rule::Ident`)?
     * - `Rule::Expr`
     * - (`Rule::StringLiteral`)?
     */
    fn parse_trigger(&self, pair: Pair<'_, Rule>) -> Trigger {
        assert_eq!(pair.as_rule(), Rule::Trigger);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();

        let mut name = None;
        let mut message = None;

        let mut pair = pairs.next().expect("mismatch between grammar and AST");
        // first token is either expression or identifier
        if let Rule::Ident = pair.as_rule() {
            name = Some(self.parse_ident(&pair));
            pair = pairs.next().expect("mismatch between grammar and AST");
        }
        let expression = self.build_expression_ast(pair.into_inner());

        if let Some(pair) = pairs.next() {
            assert_eq!(pair.as_rule(), Rule::String);
            message = Some(pair.as_str().to_string());
        }

        Trigger { id: self.next_id(), name, expression, message, span }
    }

    /**
     * Transforms a `Rule::Ident` into `Ident` AST node.
     * Panics if input is not `Rule::Ident`.
     */
    fn parse_ident(&self, pair: &Pair<'_, Rule>) -> Ident {
        assert_eq!(pair.as_rule(), Rule::Ident);
        let name = pair.as_str().to_string();
        Ident::new(name, pair.as_span().into())
    }

    /**
     * Transforms a `Rule::TypeDecl` into `TypeDeclaration` AST node.
     * Panics if input is not `Rule::TypeDecl`.
     */
    fn parse_type_declaration(&self, pair: Pair<'_, Rule>) -> TypeDeclaration {
        assert_eq!(pair.as_rule(), Rule::TypeDecl);
        let span = pair.as_span().into();
        let mut pairs = pair.into_inner();
        let name = self.parse_ident(&pairs.next().expect("mismatch between grammar and AST"));
        let mut fields = Vec::new();
        while let Some(pair) = pairs.next() {
            let field_name = pair.as_str().to_string();
            let ty = self.parse_type(pairs.next().expect("mismatch between grammar and AST"));
            fields.push(Box::new(TypeDeclField {
                name: field_name,
                ty,
                id: self.next_id(),
                span: pair.as_span().into(),
            }));
        }

        TypeDeclaration { name: Some(name), span, id: self.next_id(), fields }
    }

    /**
     * Transforms a `Rule::Type` into `Type` AST node.
     * Panics if input is not `Rule::Type`.
     */
    fn parse_type(&self, pair: Pair<'_, Rule>) -> Type {
        assert_eq!(pair.as_rule(), Rule::Type);
        let span = pair.as_span();
        let mut tuple = Vec::new();
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Ident => {
                    return Type::new_simple(self.next_id(), pair.as_str().to_string(), pair.as_span().into());
                }
                Rule::Type => tuple.push(self.parse_type(pair)),
                Rule::Optional => {
                    let span = pair.as_span();
                    let inner =
                        pair.into_inner().next().expect("mismatch between grammar and AST: first argument is a type");
                    let inner_ty = Type::new_simple(self.next_id(), inner.as_str().to_string(), inner.as_span().into());
                    return Type::new_optional(self.next_id(), inner_ty, span.into());
                }
                _ => unreachable!("{:?} is not a type, ensured by grammar", pair.as_rule()),
            }
        }
        Type::new_tuple(self.next_id(), tuple, span.into())
    }

    /**
     * Transforms a `Rule::Literal` into `Literal` AST node.
     * Panics if input is not `Rule::Literal`.
     */
    fn parse_literal(&self, pair: Pair<'_, Rule>) -> Literal {
        assert_eq!(pair.as_rule(), Rule::Literal);
        let inner = pair.into_inner().next().expect("Rule::Literal has exactly one child");
        match inner.as_rule() {
            Rule::String => {
                let str_rep = inner.as_str();
                Literal::new_str(self.next_id(), str_rep, inner.as_span().into())
            }
            Rule::RawString => {
                let str_rep = inner.as_str();
                Literal::new_raw_str(self.next_id(), str_rep, inner.as_span().into())
            }
            Rule::NumberLiteral => {
                let span = inner.as_span();
                let mut pairs = inner.into_inner();
                let value = pairs.next().expect("Mismatch between AST and grammar");

                let str_rep: &str = value.as_str();
                let unit = match pairs.next() {
                    None => None,
                    Some(unit) => Some(unit.as_str().to_string()),
                };

                Literal::new_numeric(self.next_id(), str_rep, unit, span.into())
            }
            Rule::True => Literal::new_bool(self.next_id(), true, inner.as_span().into()),
            Rule::False => Literal::new_bool(self.next_id(), false, inner.as_span().into()),
            _ => unreachable!(),
        }
    }

    #[allow(clippy::vec_box)]
    fn parse_vec_of_expressions(&self, pairs: Pairs<'_, Rule>) -> Vec<Box<Expression>> {
        pairs.map(|expr| self.build_expression_ast(expr.into_inner())).map(Box::new).collect()
    }

    fn parse_vec_of_types(&self, pairs: Pairs<'_, Rule>) -> Vec<Type> {
        pairs.map(|p| self.parse_type(p)).collect()
    }

    fn build_function_expression(&self, pair: Pair<'_, Rule>, span: Span) -> Expression {
        let mut children = pair.into_inner();
        let fun_name = self.parse_ident(&children.next().unwrap());
        let mut next = children.next().expect("Mismatch between AST and parser");
        let type_params = match next.as_rule() {
            Rule::GenericParam => {
                let params = self.parse_vec_of_types(next.into_inner());
                next = children.next().expect("Mismatch between AST and parser");
                params
            }
            Rule::FunctionArgs => Vec::new(),
            _ => unreachable!(),
        };
        assert_eq!(next.as_rule(), Rule::FunctionArgs);
        let mut args = Vec::new();
        let mut arg_names = Vec::new();
        for pair in next.into_inner() {
            assert_eq!(pair.as_rule(), Rule::FunctionArg);
            let mut pairs = pair.into_inner();
            let mut pair = pairs.next().expect("Mismatch between AST and parser");
            if pair.as_rule() == Rule::Ident {
                // named argument
                arg_names.push(Some(self.parse_ident(&pair)));
                pair = pairs.next().expect("Mismatch between AST and parser");
            } else {
                arg_names.push(None);
            }
            args.push(self.build_expression_ast(pair.into_inner()).into());
        }
        let name = FunctionName { name: fun_name, arg_names };
        Expression::new(self.next_id(), ExpressionKind::Function(name, type_params, args), span)
    }

    /**
     * Builds the Expr AST.
     */
    fn build_expression_ast(&self, pairs: Pairs<'_, Rule>) -> Expression {
        PREC_CLIMBER.climb(
            pairs,
            |pair: Pair<'_, Rule>| self.build_term_ast(pair),
            |lhs: Expression, op: Pair<'_, Rule>, rhs: Expression| {
                // Reduce function combining `Expression`s to `Expression`s with the correct precs
                let span = Span { start: lhs.span.start, end: rhs.span.end };
                let op = match op.as_rule() {
                    // Arithmetic
                    Rule::Add => BinOp::Add,
                    Rule::Subtract => BinOp::Sub,
                    Rule::Multiply => BinOp::Mul,
                    Rule::Divide => BinOp::Div,
                    Rule::Mod => BinOp::Rem,
                    Rule::Power => BinOp::Pow,
                    // Logical
                    Rule::And => BinOp::And,
                    Rule::Or => BinOp::Or,
                    // Comparison
                    Rule::LessThan => BinOp::Lt,
                    Rule::LessThanOrEqual => BinOp::Le,
                    Rule::MoreThan => BinOp::Gt,
                    Rule::MoreThanOrEqual => BinOp::Ge,
                    Rule::Equal => BinOp::Eq,
                    Rule::NotEqual => BinOp::Ne,
                    // Bitwise
                    Rule::BitAnd => BinOp::BitAnd,
                    Rule::BitOr => BinOp::BitOr,
                    Rule::BitXor => BinOp::BitXor,
                    Rule::ShiftLeft => BinOp::Shl,
                    Rule::ShiftRight => BinOp::Shr,
                    // bubble up the unary operator on the lhs (if it exists) to fix precedence
                    Rule::Dot => {
                        let (unop, binop_span, inner) = match lhs.kind {
                            ExpressionKind::Unary(unop, inner) => {
                                (Some(unop), Span { start: inner.span.start, end: rhs.span.end }, inner)
                            }
                            _ => (None, span, Box::new(lhs)),
                        };
                        match rhs.kind {
                            // access to a tuple
                            ExpressionKind::Lit(l) => {
                                let ident = match l.kind {
                                    LitKind::Numeric(val, unit) => {
                                        assert!(unit.is_none());
                                        Ident::new(val, l.span)
                                    }
                                    _ => {
                                        self.handler.error_with_span(
                                            &format!("expected unsigned integer, found {}", l),
                                            LabeledSpan::new(rhs.span, "unexpected", true),
                                        );
                                        std::process::exit(1);
                                    }
                                };
                                let binop_expr =
                                    Expression::new(self.next_id(), ExpressionKind::Field(inner, ident), binop_span);
                                match unop {
                                    None => return binop_expr,
                                    Some(unop) => {
                                        return Expression::new(
                                            self.next_id(),
                                            ExpressionKind::Unary(unop, Box::new(binop_expr)),
                                            span,
                                        )
                                    }
                                }
                            }
                            ExpressionKind::Function(name, types, args) => {
                                // match for builtin function names and transform them into appropriate AST nodes
                                let signature = name.as_string();
                                let kind = match signature.as_str() {
                                    "defaults(to:)" => {
                                        assert_eq!(args.len(), 1);
                                        ExpressionKind::Default(inner, args[0].clone())
                                    }
                                    "offset(by:)" => {
                                        assert_eq!(args.len(), 1);
                                        let offset_expr = &args[0];
                                        let offset = match offset_expr.parse_offset() {
                                            Ok(offset) => offset,
                                            Err(reason) => {
                                                self.handler.error_with_span(
                                                    "failed to parse offset",
                                                    LabeledSpan::new(rhs.span, &reason, true),
                                                );
                                                std::process::exit(1);
                                            }
                                        };
                                        ExpressionKind::Offset(inner, offset)
                                    }
                                    "hold()" => {
                                        assert_eq!(args.len(), 0);
                                        ExpressionKind::StreamAccess(inner, StreamAccessKind::Hold)
                                    }
                                    "hold(or:)" => {
                                        assert_eq!(args.len(), 1);
                                        let lhs = Expression::new(
                                            self.next_id(),
                                            ExpressionKind::StreamAccess(inner, StreamAccessKind::Hold),
                                            span,
                                        );
                                        ExpressionKind::Default(Box::new(lhs), args[0].clone())
                                    }
                                    "get()" => {
                                        assert_eq!(args.len(), 0);
                                        ExpressionKind::StreamAccess(inner, StreamAccessKind::Optional)
                                    }
                                    "aggregate(over:using:)" | "aggregate(over_exactly:using:)" => {
                                        assert_eq!(args.len(), 2);
                                        let window_op = match &args[1].kind {
                                            ExpressionKind::Ident(i) => match i.name.as_str() {
                                                "Σ" | "sum" => WindowOperation::Sum,
                                                "#" | "count" => WindowOperation::Count,
                                                //"Π" | "prod" | "product" => WindowOperation::Product,
                                                "∫" | "integral" => WindowOperation::Integral,
                                                "avg" | "average" => WindowOperation::Average,
                                                "min" => WindowOperation::Min,
                                                "max" => WindowOperation::Max,
                                                "∃" | "disjunction" | "∨" | "exists" => {
                                                    WindowOperation::Disjunction
                                                }
                                                "∀" | "conjunction" | "∧" | "forall" => {
                                                    WindowOperation::Conjunction
                                                }
                                                fun => {
                                                    self.handler.error_with_span(
                                                        &format!("unknown aggregation function {}", fun),
                                                        LabeledSpan::new(
                                                            i.span,
                                                            "available: count, min, max, sum, average, integral",
                                                            true,
                                                        ),
                                                    );
                                                    std::process::exit(1);
                                                }
                                            },
                                            _ => {
                                                self.handler.error_with_span(
                                                    "expected aggregation function",
                                                    LabeledSpan::new(
                                                        args[1].span,
                                                        "available: count, min, max, sum, average, integral",
                                                        true,
                                                    ),
                                                );
                                                std::process::exit(1);
                                            }
                                        };
                                        ExpressionKind::SlidingWindowAggregation {
                                            expr: inner,
                                            duration: args[0].clone(),
                                            wait: signature.contains("over_exactly"),
                                            aggregation: window_op,
                                        }
                                    }
                                    _ => ExpressionKind::Method(inner, name, types, args),
                                };
                                let binop_expr = Expression::new(self.next_id(), kind, binop_span);
                                match unop {
                                    None => return binop_expr,
                                    Some(unop) => {
                                        return Expression::new(
                                            self.next_id(),
                                            ExpressionKind::Unary(unop, Box::new(binop_expr)),
                                            span,
                                        )
                                    }
                                }
                            }
                            _ => {
                                self.handler.error_with_span(
                                    &format!("expected method call or tuple access, found {}", rhs),
                                    LabeledSpan::new(rhs.span, "unexpected", true),
                                );
                                std::process::exit(1);
                            }
                        }
                    }
                    Rule::OpeningBracket => {
                        let offset = match rhs.parse_offset() {
                            Ok(offset) => offset,
                            Err(reason) => {
                                self.handler.error_with_span(
                                    "failed to parse offset expression",
                                    LabeledSpan::new(rhs.span, &reason, true),
                                );
                                std::process::exit(1);
                            }
                        };
                        match lhs.kind {
                            ExpressionKind::Unary(unop, inner) => {
                                let inner_span = Span { start: inner.span.start, end: rhs.span.end };
                                let new_inner =
                                    Expression::new(self.next_id(), ExpressionKind::Offset(inner, offset), inner_span);
                                return Expression::new(
                                    self.next_id(),
                                    ExpressionKind::Unary(unop, Box::new(new_inner)),
                                    span,
                                );
                            }
                            _ => {
                                return Expression::new(
                                    self.next_id(),
                                    ExpressionKind::Offset(lhs.into(), offset),
                                    span,
                                )
                            }
                        }
                    }
                    _ => unreachable!(),
                };
                Expression::new(self.next_id(), ExpressionKind::Binary(op, Box::new(lhs), Box::new(rhs)), span)
            },
        )
    }

    fn build_term_ast(&self, pair: Pair<'_, Rule>) -> Expression {
        let span = pair.as_span();
        match pair.as_rule() {
            // Map function from `Pair` to AST data structure `Expression`
            Rule::Literal => {
                Expression::new(self.next_id(), ExpressionKind::Lit(self.parse_literal(pair)), span.into())
            }
            Rule::Ident => Expression::new(self.next_id(), ExpressionKind::Ident(self.parse_ident(&pair)), span.into()),
            Rule::ParenthesizedExpression => {
                let mut inner = pair.into_inner();
                let opp = inner.next().expect(
                    "Rule::ParenthesizedExpression has a token for the (potentialy missing) opening parenthesis",
                );
                let opening_parenthesis = if let Rule::OpeningParenthesis = opp.as_rule() {
                    Some(Box::new(Parenthesis::new(self.next_id(), opp.as_span().into())))
                } else {
                    None
                };

                let inner_expression =
                    inner.next().expect("Rule::ParenthesizedExpression has a token for the contained expression");

                let closing = inner.next().expect(
                    "Rule::ParenthesizedExpression has a token for the (potentialy missing) closing parenthesis",
                );
                let closing_parenthesis = if let Rule::ClosingParenthesis = closing.as_rule() {
                    Some(Box::new(Parenthesis::new(self.next_id(), closing.as_span().into())))
                } else {
                    None
                };

                Expression::new(
                    self.next_id(),
                    ExpressionKind::ParenthesizedExpression(
                        opening_parenthesis,
                        Box::new(self.build_expression_ast(inner_expression.into_inner())),
                        closing_parenthesis,
                    ),
                    span.into(),
                )
            }
            Rule::UnaryExpr => {
                // First child is the operator, second the operand.
                let mut children = pair.into_inner();
                let pest_operator = children.next().expect("Unary expressions need to have an operator.");
                let operand = children.next().expect("Unary expressions need to have an operand.");
                let operand = self.build_term_ast(operand);
                let operator = match pest_operator.as_rule() {
                    Rule::Add => return operand, // Discard unary plus because it is semantically null.
                    Rule::Subtract => UnOp::Neg,
                    Rule::Neg => UnOp::Not,
                    Rule::BitNot => UnOp::BitNot,
                    _ => unreachable!(),
                };
                Expression::new(self.next_id(), ExpressionKind::Unary(operator, Box::new(operand)), span.into())
            }
            Rule::TernaryExpr => {
                let mut children = self.parse_vec_of_expressions(pair.into_inner());
                assert_eq!(children.len(), 3, "A ternary expression needs exactly three children.");
                Expression::new(
                    self.next_id(),
                    ExpressionKind::Ite(children.remove(0), children.remove(0), children.remove(0)),
                    span.into(),
                )
            }
            Rule::Tuple => {
                let elements = self.parse_vec_of_expressions(pair.into_inner());
                assert!(elements.len() != 1, "Tuples may not have exactly one element.");
                Expression::new(self.next_id(), ExpressionKind::Tuple(elements), span.into())
            }
            Rule::Expr => self.build_expression_ast(pair.into_inner()),
            Rule::FunctionExpr => self.build_function_expression(pair, span.into()),
            Rule::IntegerLiteral => {
                let span = span.into();
                Expression::new(
                    self.next_id(),
                    ExpressionKind::Lit(Literal::new_numeric(self.next_id(), pair.as_str(), None, span)),
                    span,
                )
            }
            Rule::MissingExpression => {
                let span = span.into();
                Expression::new(self.next_id(), ExpressionKind::MissingExpression, span)
            }
            _ => unreachable!("Unexpected rule when parsing expression ast: {:?}", pair.as_rule()),
        }
    }
}

/**
 * Transforms a textual representation of a Lola specification into
 * an AST representation.
 */
pub(crate) fn parse<'a, 'b>(
    content: &'a str,
    handler: &'b Handler,
    config: FrontendConfig,
) -> Result<RTLolaAst, pest::error::Error<Rule>> {
    RTLolaParser::new(content, handler, config).parse()
}

#[derive(Debug, Clone, Eq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub(crate) fn new(name: String, span: Span) -> Ident {
        Ident { name, span }
    }
}

/// In the equality definition of `Ident`, we only compare the string values
/// and ignore the `Span` info
impl PartialEq for Ident {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// Every node in the AST gets a unique id, represented by a 32bit unsiged integer.
/// They are used in the later analysis phases to store information about AST nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

impl NodeId {
    pub(crate) fn new(x: usize) -> NodeId {
        assert!(x < (u32::max_value() as usize));
        NodeId(x as u32)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A span marks a range in a file.
/// Start and end positions are *byte* offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    // TODO Do we need this here or do we want to keep a mapping from byte positions to lines in the LSP part.
    // line: usize,
    // /// The LSP uses UTF-16 code units (2 bytes) as their unit for offsets.
    // lineOffsetLSP: usize,
}

impl Span {
    pub(crate) fn unknown() -> Span {
        use std::usize;
        Span { start: usize::max_value(), end: usize::max_value() }
    }
}

impl<'a> From<pest::Span<'a>> for Span {
    fn from(span: pest::Span<'a>) -> Self {
        Span { start: span.start(), end: span.end() }
    }
}

/// A mapper from `Span` to actual source code
#[derive(Debug)]
pub(crate) struct SourceMapper {
    path: PathBuf,
    content: String,
}

#[derive(Debug, Eq, Ord)]
pub(crate) struct CodeLine {
    pub(crate) path: PathBuf,
    pub(crate) line_number: usize,
    pub(crate) column_number: usize,
    pub(crate) line: String,
    pub(crate) highlight: CharSpan,
}

impl PartialEq for CodeLine {
    /// the equality on code lines is given by the equality of the tuples `(path, line_number, column_number)`
    fn eq(&self, rhs: &CodeLine) -> bool {
        (&self.path, self.line_number, self.column_number).eq(&(&rhs.path, rhs.line_number, rhs.column_number))
    }
}

impl PartialOrd for CodeLine {
    /// the partial order on code lines is given by the lexicographic order of `(path, line_number, column_number)`
    fn partial_cmp(&self, rhs: &CodeLine) -> Option<std::cmp::Ordering> {
        (&self.path, self.line_number, self.column_number).partial_cmp(&(&rhs.path, rhs.line_number, rhs.column_number))
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct CharSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

impl SourceMapper {
    pub(crate) fn new(path: PathBuf, content: &str) -> SourceMapper {
        SourceMapper { path, content: content.to_string() }
    }

    #[allow(dead_code)]
    pub(crate) fn get_line(&self, span: Span) -> Option<CodeLine> {
        let mut byte_offset = 0;
        for (num, line) in self.content.split('\n').enumerate() {
            assert!(byte_offset <= span.start);
            let line_end = byte_offset + line.len() + 1; // +1 as it is excluding newline character

            if span.start < line_end {
                if span.end > line_end {
                    // not a single line
                    return None;
                } else {
                    // get column
                    let mut column = 0;
                    let mut start: Option<usize> = None;
                    let mut end: Option<usize> = None;
                    let mut i = 0;
                    for (index, _) in line.char_indices() {
                        i = index;
                        if index < span.start - byte_offset {
                            column += 1;
                        } else if index == span.start - byte_offset {
                            start = Some(index);
                        } else if index == span.end - byte_offset {
                            end = Some(index);
                            break;
                        }
                    }
                    let (start, end) = (
                        start.unwrap_or(i), // we might hit the end of the line/EOI
                        end.unwrap_or(i + 1),
                    );

                    return Some(CodeLine {
                        path: self.path.clone(),
                        line_number: num + 1,
                        column_number: column + 1,
                        line: line.to_string(),
                        highlight: CharSpan { start, end },
                    });
                }
            }
            byte_offset = line_end;
        }
        unreachable!();
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use pest::{consumes_to, parses_to};

    fn cmp_ast_spec(ast: &RTLolaAst, spec: &str) -> bool {
        // Todo: Make more robust, e.g. against changes in whitespace.
        assert_eq!(format!("{}", ast), spec);
        true
    }

    #[test]
    fn parse_simple() {
        let _ = LolaParser::parse(Rule::Spec, "input in: Int\noutput out: Int := in\ntrigger in ≠ out")
            .unwrap_or_else(|e| panic!("{}", e));
    }

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn parse_constant() {
        parses_to! {
            parser: LolaParser,
            input:  "constant five : Int := 5",
            rule:   Rule::ConstantStream,
            tokens: [
                ConstantStream(0, 24, [
                    Ident(9, 13, []),
                    Type(16, 19, [
                        Ident(16, 19, []),
                    ]),
                    Literal(23, 24, [
                        NumberLiteral(23, 24, [
                            NumberLiteralValue(23, 24, [])
                        ]),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_constant_ast() {
        let spec = "constant five : Int := 5";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let parser = RTLolaParser::new(spec, &handler, FrontendConfig::default());
        let pair = LolaParser::parse(Rule::ConstantStream, spec).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let ast = parser.parse_constant(pair);
        assert_eq!(format!("{}", ast), "constant five: Int := 5")
    }

    #[test]
    fn parse_constant_double() {
        let spec = "constant fiveoh: Double := 5.0";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let parser = RTLolaParser::new(spec, &handler, FrontendConfig::default());
        let pair = LolaParser::parse(Rule::ConstantStream, spec).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let ast = parser.parse_constant(pair);
        assert_eq!(format!("{}", ast), "constant fiveoh: Double := 5.0")
    }

    #[test]
    fn parse_input() {
        parses_to! {
            parser: LolaParser,
            input:  "input in: Int",
            rule:   Rule::InputStream,
            tokens: [
                InputStream(0, 13, [
                    Ident(6, 8, []),
                    Type(10, 13, [
                        Ident(10, 13, []),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_input_ast() {
        let spec = "input a: Int, b: Int, c: Bool";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let parser = RTLolaParser::new(spec, &handler, FrontendConfig::default());
        let pair = LolaParser::parse(Rule::InputStream, spec).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let inputs = parser.parse_inputs(pair);
        assert_eq!(inputs.len(), 3);
        assert_eq!(format!("{}", inputs[0]), "input a: Int");
        assert_eq!(format!("{}", inputs[1]), "input b: Int");
        assert_eq!(format!("{}", inputs[2]), "input c: Bool");
    }

    #[test]
    fn build_ast_parameterized_input() {
        let spec = "input in (ab: Int8): Int8\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn parse_output() {
        parses_to! {
            parser: LolaParser,
            input:  "output out: Int := in + 1",
            rule:   Rule::OutputStream,
            tokens: [
                OutputStream(0, 25, [
                    Ident(7, 10, []),
                    Type(12, 15, [
                        Ident(12, 15, []),
                    ]),
                    Expr(19, 25, [
                        Ident(19, 21, []),
                        Add(22, 23, []),
                        Literal(24, 25, [
                            NumberLiteral(24, 25, [
                                NumberLiteralValue(24, 25, [])
                            ]),
                        ]),
                    ]),
                ]),
            ]
        };
    }

    #[test]
    fn parse_output_ast() {
        let spec = "output out: Int := in + 1";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let parser = RTLolaParser::new(spec, &handler, FrontendConfig::default());
        let pair = LolaParser::parse(Rule::OutputStream, spec).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let ast = parser.parse_output(pair);
        assert_eq!(format!("{}", ast), spec)
    }

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn parse_trigger() {
        parses_to! {
            parser: LolaParser,
            input:  "trigger in != out \"some message\"",
            rule:   Rule::Trigger,
            tokens: [
                Trigger(0, 32, [
                    Expr(8, 17, [
                        Ident(8, 10, []),
                        NotEqual(11, 13, []),
                        Ident(14, 17, []),
                    ]),
                    String(19, 31, []),
                ]),
            ]
        };
    }

    #[test]
    fn parse_trigger_ast() {
        let spec = "trigger in ≠ out \"some message\"";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let parser = RTLolaParser::new(spec, &handler, FrontendConfig::default());
        let pair = LolaParser::parse(Rule::Trigger, spec).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let ast = parser.parse_trigger(pair);
        assert_eq!(format!("{}", ast), "trigger in ≠ out \"some message\"")
    }

    #[test]
    fn parse_expression() {
        let content = "in + 1";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let expr = LolaParser::parse(Rule::Expr, content).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let parser = RTLolaParser::new(content, &handler, FrontendConfig::default());
        let ast = parser.build_expression_ast(expr.into_inner());
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn parse_expression_precedence() {
        let content = "(a ∨ b ∧ c)";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let parser = RTLolaParser::new(content, &handler, FrontendConfig::default());
        let expr = LolaParser::parse(Rule::Expr, content).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let ast = parser.build_expression_ast(expr.into_inner());
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn parse_missing_closing_parenthesis() {
        let content = "(a ∨ b ∧ c";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), content));
        let parser = RTLolaParser::new(content, &handler, FrontendConfig::default());
        let expr = LolaParser::parse(Rule::Expr, content).unwrap_or_else(|e| panic!("{}", e)).next().unwrap();
        let ast = parser.build_expression_ast(expr.into_inner());
        assert_eq!(format!("{}", ast), content)
    }

    #[test]
    fn build_simple_ast() {
        let spec = "input in: Int\noutput out: Int := in\ntrigger in ≠ out\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_ast_input() {
        let spec = "input in: Int\ninput in2: Int\ninput in3: (Int, Bool)\ninput in4: Bool\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_parenthesized_expression() {
        let spec = "output s: Bool := (true ∨ true)\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_optional_type() {
        let spec = "output s: Bool? := (false ∨ true)\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_default() {
        let spec = "output s: Int := s.offset(by: -1).defaults(to: (3 * 4))\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_lookup_expression_hold() {
        let spec = "output s: Int := s.offset(by: -1).hold().defaults(to: 3 * 4)\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_ternary_expression() {
        let spec = "input in: Int\noutput s: Int := if in = 3 then 4 else in + 2\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_function_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int := nroot(1, sin(1, in))\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_trigger() {
        let spec = "input in: Int\ntrigger in > 5\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_complex_expression() {
        let spec =
            "output s: Double := if !((s.offset(by: -1).defaults(to: (3 * 4)) + -4) = 12) ∨ true = false then 2.0 else 4.1\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_type_declaration() {
        let spec = "type VerifiedUser { name: String }\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_parameter_list() {
        let spec = "output s (a: B, c: D): E := 3\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_termination_spec() {
        let spec = "output s (a: Int): Int close s > 10 := 3\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn build_tuple_expression() {
        let spec = "input in: (Int, Bool)\noutput s: Int := (1, in.0).1\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_string() {
        let spec = r#"constant s: String := "a string with \n newline"
"#;
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_raw_string() {
        let spec = r##"constant s: String := r#"a raw \ string that " needs padding"#
"##;
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_import() {
        let spec = "import math\ninput in: UInt8\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_max() {
        let spec = "import math\ninput a: Int32\ninput b: Int32\noutput maxres: Int32 := max<Int32>(a, b)\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call() {
        let spec = "output count := count.offset(-1).default(0) + 1\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_method_call_with_param() {
        let spec = "output count := count.offset<Int8>(-1).default(0) + 1\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_realtime_offset() {
        let spec = "output a := b.offset(by: -1s)\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_future_offset() {
        let spec = "output a := b.offset(by: 1)\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[test]
    fn parse_function_argument_name() {
        let spec = "output a := b.hold().defaults(to: 0)\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, spec);
    }

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn parse_precedence_not_regression() {
        parses_to! {
            parser: LolaParser,
            input:  "!(fast || false) && fast",
            rule:   Rule::Expr,
            tokens: [
                Expr(0, 24, [
                    UnaryExpr(0, 16, [
                        Neg(0, 1, []),
                        ParenthesizedExpression(1, 16, [
                            OpeningParenthesis(1, 2, []),
                            Expr(2, 15, [
                                Ident(2, 6, []),
                                Or(7, 9, []),
                                Literal(10, 15, [
                                    False(10, 15, [])
                                ])
                            ]),
                            ClosingParenthesis(15, 16, [])
                        ])
                    ]),
                    And(17, 19, []),
                    Ident(20, 24, [])
                ]),
            ]
        };
    }

    #[test]
    fn handle_bom() {
        let spec = "\u{feff}input a: Bool\n";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, "input a: Bool\n");
    }

    #[test]
    fn regression71() {
        let spec = "output outputstream := 42 output c := outputstream";
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(|e| panic!("{}", e));
        cmp_ast_spec(&ast, "output outputstream := 42\noutput c := outputstream\n");
    }

    #[test]
    fn parse_bitwise() {
        let spec = "output x := 1 ^ 0 & 23123 | 111\n";
        let throw = |e| panic!("{}", e);
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let ast = parse(spec, &handler, FrontendConfig::default()).unwrap_or_else(throw);
        cmp_ast_spec(&ast, spec);
    }
}
