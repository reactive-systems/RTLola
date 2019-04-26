use crate::analysis::graph_based_analysis::get_byte_size;
use crate::analysis::graph_based_analysis::space_requirements::TrackingRequirements;
use crate::analysis::graph_based_analysis::MemoryBound;
use crate::analysis::graph_based_analysis::TrackingRequirement;
use crate::analysis::graph_based_analysis::{SpaceRequirements, StorageRequirement};
use crate::analysis::naming::Declaration;
use crate::analysis::DeclarationTable;
use crate::analysis::TypeTable;
use crate::ast;
use crate::ast::{ExpressionKind, LolaSpec, Offset, StreamAccessKind, WindowOperation};
use crate::ty::StreamTy;
use num::{BigRational, ToPrimitive};
use std::cmp::min;

fn is_efficient_operator(op: WindowOperation) -> bool {
    match op {
        WindowOperation::Average
        | WindowOperation::Count
        | WindowOperation::Sum
        | WindowOperation::Product
        | WindowOperation::Integral => true,
    }
}

fn determine_needed_window_memory(type_size: u128, number_of_element: u128, op: WindowOperation) -> u128 {
    match op {
        WindowOperation::Product | WindowOperation::Sum => type_size * number_of_element,
        WindowOperation::Count => number_of_element * 8,
        WindowOperation::Integral => number_of_element * (4 * 8 + 1 + 8),
        WindowOperation::Average => number_of_element * (8 + type_size),
    }
}

fn add_sliding_windows<'a>(
    expr: &ast::Expression,
    type_table: &TypeTable,
    declaration_table: &DeclarationTable<'a>,
) -> MemoryBound {
    let mut required_memory: u128 = 0;
    let mut unknown_size = false;

    match &expr.kind {
        ExpressionKind::Lit(_) | ExpressionKind::Ident(_) => {}
        ExpressionKind::Binary(_op, left, right) => {
            match add_sliding_windows(&*left, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
            match add_sliding_windows(&*right, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
        }
        ExpressionKind::MissingExpression => return MemoryBound::Unknown,
        ExpressionKind::StreamAccess(e, access_type) => match access_type {
            StreamAccessKind::Hold => {
                unimplemented!();
                /*if let ExpressionKind::Lookup(_, _, _) = &e.kind {
                    match add_sliding_windows(&*e, type_table, declaration_table) {
                        MemoryBound::Bounded(u) => required_memory += u,
                        MemoryBound::Unbounded => return MemoryBound::Unbounded,
                        MemoryBound::Unknown => unknown_size = true,
                    };
                } else {
                    // A "stray" sample and hold expression such as `5.hold()` is valid, but a no-op.
                    // Thus, print a warning. Evaluating the expression is necessary, the dft can be skipped.
                    match add_sliding_windows(&*e, type_table, declaration_table) {
                        MemoryBound::Bounded(u) => required_memory += u,
                        MemoryBound::Unbounded => return MemoryBound::Unbounded,
                        MemoryBound::Unknown => unknown_size = true,
                    };
                }*/
            }
            StreamAccessKind::Optional => {
                unimplemented!();
            }
        },
        ExpressionKind::Default(e, dft) => {
            unimplemented!();
            /*if let ExpressionKind::Lookup(_, _, _) = &e.kind {
                match add_sliding_windows(&*e, type_table, declaration_table) {
                    MemoryBound::Bounded(u) => required_memory += u,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => unknown_size = true,
                };
                match add_sliding_windows(&*dft, type_table, declaration_table) {
                    MemoryBound::Bounded(u) => required_memory += u,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => unknown_size = true,
                };
            } else {
                // A "stray" sample and hold expression such as `5 ! 3` is valid, but a no-op.
                // Thus, print a warning. Evaluating the expression is necessary, the dft can be skipped.
                match add_sliding_windows(&*e, type_table, declaration_table) {
                    MemoryBound::Bounded(u) => required_memory += u,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => unknown_size = true,
                };
            }*/
        }
        ExpressionKind::Offset(_, _) => unimplemented!(),
        ExpressionKind::SlidingWindowAggregation { expr: _expr, duration: _duration, aggregation: _aggregation } => {
            unimplemented!()
        }
        ExpressionKind::Unary(_, inner) | ExpressionKind::ParenthesizedExpression(_, inner, _) => {
            match add_sliding_windows(&*inner, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
        }
        ExpressionKind::Ite(condition, ifcase, elsecase) => {
            match add_sliding_windows(&*condition, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
            match add_sliding_windows(&*ifcase, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
            match add_sliding_windows(&*elsecase, type_table, declaration_table) {
                MemoryBound::Bounded(u) => required_memory += u,
                MemoryBound::Unbounded => return MemoryBound::Unbounded,
                MemoryBound::Unknown => unknown_size = true,
            };
        }
        ExpressionKind::Function(_, _, elements) | ExpressionKind::Tuple(elements) => {
            for expr in elements {
                match add_sliding_windows(&*expr, type_table, declaration_table) {
                    MemoryBound::Bounded(u) => required_memory += u,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                    MemoryBound::Unknown => unknown_size = true,
                };
            }
        }
        ExpressionKind::Field(_, _) | ExpressionKind::Method(_, _, _, _) => {
            unimplemented!()
        } /*ExpressionKind::Lookup(instance, offset, op) => match op {
              None => {}
              Some(wop) => {
                  let node_id = match declaration_table
                      .get(&instance.id)
                      .expect("We expect the the declaration-table to contain information about every stream access")
                  {
                      Declaration::In(input) => input.id,
                      Declaration::Out(output) => output.id,
                      _ => unimplemented!(),
                  };

                  let timing = &type_table.get_stream_type(node_id).timing;
                  let value_type = type_table.get_value_type(node_id);
                  let value_type_size = match get_byte_size(value_type) {
                      MemoryBound::Bounded(i) => i,
                      MemoryBound::Unbounded => return MemoryBound::Unbounded,
                      MemoryBound::Unknown => {
                          unknown_size = true;
                          0
                      }
                  };
                  let efficient_operator: bool = is_efficient_operator(*wop);
                  match (timing, efficient_operator) {
                      (TimingInfo::Event, false) => {
                          return MemoryBound::Unbounded;
                      }
                      (TimingInfo::RealTime(freq), false) => {
                          let window_size = match offset {
                              Offset::DiscreteOffset(_) => unreachable!(),
                              Offset::RealTimeOffset(time_spec) => &time_spec.exact_period,
                          };
                          let number_of_full_periods_in_window: BigRational = window_size / &freq.ns;
                          required_memory += number_of_full_periods_in_window
                              .to_integer()
                              .to_u128()
                              .expect("Number of complete periods does not fit in u128")
                              * value_type_size;
                      }
                      (TimingInfo::Event, true) => {
                          let number_of_panes = 64;
                          required_memory += determine_needed_window_memory(value_type_size, number_of_panes, *wop);
                      }
                      (TimingInfo::RealTime(freq), true) => {
                          let number_of_panes = 64;
                          let window_size = match offset {
                              Offset::DiscreteOffset(_) => unreachable!(),
                              Offset::RealTimeOffset(time_spec) => &time_spec.exact_period,
                          };
                          let number_of_full_periods_in_window: BigRational = window_size / &freq.ns;
                          let number_of_elements = min(
                              number_of_full_periods_in_window
                                  .to_integer()
                                  .to_u128()
                                  .expect("Number of complete periods does not fit in u128"),
                              number_of_panes,
                          );
                          required_memory += determine_needed_window_memory(value_type_size, number_of_elements, *wop);
                      }
                  }
              }
          },*/
    }
    if unknown_size {
        MemoryBound::Unknown
    } else {
        MemoryBound::Bounded(required_memory)
    }
}

pub(crate) fn determine_worst_case_memory_consumption<'a>(
    spec: &LolaSpec,
    buffer_requirements: &SpaceRequirements,
    tracking_requirements: &TrackingRequirements,
    type_table: &TypeTable,
    declaration_table: &DeclarationTable<'a>,
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
                    StorageRequirement::Unbounded => unimplemented!(),
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
                    StorageRequirement::FutureRef(_) => unimplemented!(),
                    StorageRequirement::Unbounded => unimplemented!(),
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
                let mut it = tracking_requirement.iter();
                let mut tracking_per_instance: u128 = 0_u128;
                while let Some((node_id, tracking)) = it.next() {
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
            let mut it = tracking_requirement.iter();
            let mut tracking_per_instance: u128 = 0_u128;
            while let Some((node_id, tracking)) = it.next() {
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
