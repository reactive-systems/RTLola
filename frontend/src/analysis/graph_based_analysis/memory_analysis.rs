use crate::analysis::graph_based_analysis::space_requirements::TrackingRequirements;
use crate::analysis::graph_based_analysis::{get_byte_size, MemoryBound, TrackingRequirement};
use crate::analysis::graph_based_analysis::{SpaceRequirements, StorageRequirement};
use crate::analysis::naming::Declaration;
use crate::analysis::DeclarationTable;
use crate::analysis::TypeTable;
use crate::ast;
use crate::ast::{ExpressionKind, RTLolaAst, WindowOperation};
use crate::ty::StreamTy;
use num::rational::Rational64 as Rational;
use num::ToPrimitive;
use std::cmp::min;
use uom::si::frequency::hertz;
use uom::si::time::second;

fn is_efficient_operator(op: WindowOperation) -> bool {
    match op {
        WindowOperation::Count
        | WindowOperation::Min
        | WindowOperation::Max
        | WindowOperation::Sum
        | WindowOperation::Product
        | WindowOperation::Average
        | WindowOperation::Disjunction
        | WindowOperation::Conjunction
        | WindowOperation::Integral => true,
    }
}

fn determine_needed_window_memory(type_size: u128, number_of_element: u128, op: WindowOperation) -> u128 {
    match op {
        WindowOperation::Count => number_of_element * 8,
        WindowOperation::Min | WindowOperation::Max => number_of_element * type_size,
        WindowOperation::Sum | WindowOperation::Product => number_of_element * type_size,
        WindowOperation::Average => number_of_element * (8 + type_size),
        WindowOperation::Conjunction | WindowOperation::Disjunction => number_of_element * type_size,
        WindowOperation::Integral => number_of_element * (4 * 8 + 1 + 8),
    }
}

fn add_sliding_windows(
    expr: &ast::Expression,
    type_table: &TypeTable,
    declaration_table: &DeclarationTable,
) -> MemoryBound {
    let mut required_memory: u128 = 0;
    let mut unknown_size = false;

    use ExpressionKind::*;
    match &expr.kind {
        Lit(_) | Ident(_) => {}
        Binary(_, left, right) | Default(left, right) => {
            match add_sliding_windows(left, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
            match add_sliding_windows(right, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
        }
        MissingExpression => return MemoryBound::Unknown,
        Unary(_, inner) | ParenthesizedExpression(_, inner, _) | StreamAccess(inner, _) | Offset(inner, _) => {
            match add_sliding_windows(inner, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
        }
        Ite(condition, ifcase, elsecase) => {
            match add_sliding_windows(condition, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
            match add_sliding_windows(ifcase, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
            match add_sliding_windows(elsecase, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
        }
        Function(_, _, elements) | Tuple(elements) => {
            for expr in elements {
                match add_sliding_windows(expr, type_table, declaration_table) {
                    MemoryBound::Bounded(u) => required_memory += u,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => unknown_size = true,
                };
            }
        }
        Method(inner, _, _, params) => {
            match add_sliding_windows(inner, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
            for expr in params {
                match add_sliding_windows(expr, type_table, declaration_table) {
                    MemoryBound::Bounded(u) => required_memory += u,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => unknown_size = true,
                };
            }
        }
        Field(expr, ident) => {
            let num: usize = ident.name.parse::<usize>().expect("checked in AST verifier");
            if let Some(inner) = expr.get_expr_from_tuple(num) {
                match add_sliding_windows(inner, type_table, declaration_table) {
                    MemoryBound::Bounded(u) => required_memory += u,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => unknown_size = true,
                };
            } else {
                match add_sliding_windows(expr, type_table, declaration_table) {
                    MemoryBound::Bounded(u) => required_memory += u,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => unknown_size = true,
                };
            }
        }
        SlidingWindowAggregation { expr, duration, aggregation, .. } => {
            if let Ident(_) = &expr.kind {
            } else {
                unreachable!("checked in AST verification");
            }

            let node_id = match declaration_table
                .get(&expr.id)
                .expect("We expect the the declaration-table to contain information about every stream access")
            {
                Declaration::In(input) => input.id,
                Declaration::Out(output) => output.id,
                _ => unimplemented!(),
            };

            let stream_ty = &type_table.get_stream_type(node_id);
            let value_type = type_table.get_value_type(node_id);
            let value_type_size = match get_byte_size(value_type) {
                MemoryBound::Bounded(i) => i,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => {
                    unknown_size = true;
                    0
                }
            };
            let efficient_operator: bool = is_efficient_operator(*aggregation);
            match (stream_ty, efficient_operator) {
                (StreamTy::Event(_), false) => {
                    return MemoryBound::Unbounded;
                }
                (StreamTy::RealTime(freq), false) => {
                    let window_size = duration.parse_duration().expect("durations have been checked before");
                    let number_of_full_periods_in_window: Rational =
                        window_size.get::<second>() / freq.freq.get::<hertz>();
                    required_memory += number_of_full_periods_in_window
                        .to_integer()
                        .to_u128()
                        .expect("Number of complete periods does not fit in u128")
                        * value_type_size;
                }
                (StreamTy::Event(_), true) => {
                    let number_of_panes = 64;
                    required_memory += determine_needed_window_memory(value_type_size, number_of_panes, *aggregation);
                }
                (StreamTy::RealTime(freq), true) => {
                    let number_of_panes = 64;
                    let window_size = duration.parse_duration().expect("durations have been checked before");
                    let number_of_full_periods_in_window: Rational =
                        window_size.get::<second>() / freq.freq.get::<hertz>();
                    let number_of_elements = min(
                        number_of_full_periods_in_window
                            .to_integer()
                            .to_u128()
                            .expect("Number of complete periods does not fit in u128"),
                        number_of_panes,
                    );
                    required_memory +=
                        determine_needed_window_memory(value_type_size, number_of_elements, *aggregation);
                }
                _ => unreachable!("checked in type checking"),
            }
        }
    }
    if unknown_size {
        MemoryBound::Unknown
    } else {
        MemoryBound::Bounded(required_memory)
    }
}

pub(crate) fn determine_worst_case_memory_consumption(
    spec: &RTLolaAst,
    buffer_requirements: &SpaceRequirements,
    tracking_requirements: &TrackingRequirements,
    type_table: &TypeTable,
    declaration_table: &DeclarationTable,
) -> MemoryBound {
    //----------------------
    // fixed shared overhead
    let mut required_memory: u128 = 10_000;
    let mut unknown_size = false;

    for input in &spec.inputs {
        if input.params.is_empty() {
            if let Some(storage_requirement) = buffer_requirements.get(&input.id) {
                let buffer_size_per_instance = match storage_requirement {
                    StorageRequirement::Finite(size) => u128::from(*size),
                    StorageRequirement::FutureRef(_) => unreachable!(),
                    StorageRequirement::Unbounded => return MemoryBound::Unbounded,
                };
                let value_type = type_table.get_value_type(input.id);
                let value_type_size = match get_byte_size(value_type) {
                    MemoryBound::Bounded(i) => i,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => {
                        unknown_size = true;
                        0
                    }
                };

                required_memory += value_type_size * buffer_size_per_instance;

                //----------------------
                // fixed overhead per stream
                required_memory += 100;
            }
        } else {
            // TODO also add the space for the parametrization of the streams
            unimplemented!()
        }
    }
    for output in &spec.outputs {
        if output.params.is_empty() {
            if let Some(storage_requirement) = buffer_requirements.get(&output.id) {
                let buffer_size_per_instance = match storage_requirement {
                    StorageRequirement::Finite(size) => u128::from(*size),
                    StorageRequirement::FutureRef(_) => 0u128,
                    StorageRequirement::Unbounded => return MemoryBound::Unbounded,
                };
                let value_type = type_table.get_value_type(output.id);
                let value_type_size = match get_byte_size(value_type) {
                    MemoryBound::Bounded(i) => i,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => {
                        unknown_size = true;
                        0
                    }
                };

                required_memory += value_type_size * buffer_size_per_instance;

                //----------------------
                // tracking

                let tracking_requirement = tracking_requirements.get(&output.id).expect(
                    "We should have determined the tracking requirements for all streams that did not get pruned!",
                );
                let mut tracking_per_instance: u128 = 0_u128;
                for (node_id, tracking) in tracking_requirement {
                    let value_type = type_table.get_value_type(*node_id);
                    let value_type_size = if let MemoryBound::Bounded(i) = get_byte_size(value_type) {
                        i
                    } else {
                        return MemoryBound::Unbounded;
                    };

                    match tracking {
                        TrackingRequirement::Finite(size) => {
                            tracking_per_instance += u128::from(*size) * value_type_size
                        }
                        // TODO What about accessing a future dependent stream?
                        TrackingRequirement::Future => unimplemented!(),
                        TrackingRequirement::Unbounded => unimplemented!(),
                    }
                }

                required_memory += tracking_per_instance;

                //----------------------
                // windows
                let windows_space = match add_sliding_windows(&output.expression, type_table, declaration_table) {
                    MemoryBound::Bounded(i) => i,
                    MemoryBound::Unknown => {
                        unknown_size = true;
                        0
                    }
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                };
                required_memory += windows_space;

                //----------------------
                // fixed overhead per stream
                required_memory += 500;
            }
        } else {
            // TODO also add the space for the parametrization of the streams
            unimplemented!()
        }
    }
    for trigger in &spec.trigger {
        //----------------------
        // tracking
        if let Some(tracking_requirement) = tracking_requirements.get(&trigger.id) {
            let mut tracking_per_instance: u128 = 0_u128;
            for (node_id, tracking) in tracking_requirement {
                let value_type = type_table.get_value_type(*node_id);
                let value_type_size = match get_byte_size(value_type) {
                    MemoryBound::Bounded(i) => i,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => {
                        unknown_size = true;
                        0
                    }
                };

                match tracking {
                    TrackingRequirement::Finite(size) => tracking_per_instance += u128::from(*size) * value_type_size,
                    // TODO What about accessing a future dependent stream?
                    TrackingRequirement::Future => unimplemented!(),
                    TrackingRequirement::Unbounded => unimplemented!(),
                }
            }

            required_memory += tracking_per_instance;

            //----------------------
            // windows
            let windows_space = match add_sliding_windows(&trigger.expression, type_table, declaration_table) {
                MemoryBound::Bounded(i) => i,
                MemoryBound::Unknown => {
                    unknown_size = true;
                    0
                }
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
            };
            required_memory += windows_space;

            //----------------------
            // fixed overhead per stream
            required_memory += 500;
        }
    }
    if unknown_size {
        MemoryBound::Unknown
    } else {
        MemoryBound::Bounded(required_memory)
    }
}
