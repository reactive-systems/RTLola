use crate::analysis::graph_based_analysis::space_requirements::TrackingRequirements;
use crate::analysis::graph_based_analysis::MemoryBound;
use crate::analysis::graph_based_analysis::TrackingRequirement;
use crate::analysis::graph_based_analysis::{SpaceRequirements, StorageRequirement};
use crate::analysis::TypeTable;
use crate::ast::LolaSpec;
use crate::ty::ValueTy;
use crate::{FloatTy, IntTy, UIntTy};

fn get_byte_size(value_ty: &ValueTy) -> MemoryBound {
    match value_ty {
        ValueTy::Bool => MemoryBound::Bounded(1),
        ValueTy::Int(int_ty) => MemoryBound::Bounded(match int_ty {
            IntTy::I8 => 1,
            IntTy::I16 => 2,
            IntTy::I32 => 4,
            IntTy::I64 => 8,
        }),
        ValueTy::UInt(uint_ty) => MemoryBound::Bounded(match uint_ty {
            UIntTy::U8 => 1,
            UIntTy::U16 => 2,
            UIntTy::U32 => 4,
            UIntTy::U64 => 8,
        }),
        ValueTy::Float(float_ty) => MemoryBound::Bounded(match float_ty {
            FloatTy::F32 => 4,
            FloatTy::F64 => 8,
        }),
        // an abstract data type, e.g., structs, enums, etc.
        //ValueTy::Adt(AdtDef),
        ValueTy::String => MemoryBound::Unbounded,
        ValueTy::Tuple(elements) => {
            let mut elements = elements.iter().map(get_byte_size);
            let mut accu = 0u128;
            while let Some(element) = elements.next() {
                match element {
                    MemoryBound::Bounded(i) => accu += i,
                    MemoryBound::Unbounded => return MemoryBound::Unbounded,
                };
            }
            MemoryBound::Bounded(accu)
        }
        // an optional value type, e.g., resulting from accessing a stream with offset -1
        ValueTy::Option(inner) => get_byte_size(inner),
        // Used during type inference
        ValueTy::Infer(_value_var) => unreachable!(),
        ValueTy::Constr(_type_constraint) => unreachable!(),
        // A reference to a generic parameter in a function declaration, e.g. `T` in `a<T>(x:T) -> T`
        ValueTy::Param(_, _) => MemoryBound::Bounded(0),
        ValueTy::Error => unreachable!(),
    }
}

pub(crate) fn determine_worst_case_memory_consumption(
    spec: &LolaSpec,
    buffer_requirements: &SpaceRequirements,
    tracking_requirements: &TrackingRequirements,
    type_table: &TypeTable,
) -> MemoryBound {
    let mut required_memory: u128 = 0;

    for input in &spec.inputs {
        if input.params.is_empty() {
            if let Some(storage_requirement) = buffer_requirements.get(&input.id) {
                let buffer_size_per_instance = match storage_requirement {
                    StorageRequirement::Finite(size) => *size as u128,
                    StorageRequirement::FutureRef(_) => unreachable!(),
                    StorageRequirement::Unbounded => unimplemented!(),
                };
                let value_type = type_table.get_value_type(input.id);
                let value_type_size = if let MemoryBound::Bounded(i) = get_byte_size(value_type) {
                    i
                } else {
                    return MemoryBound::Unbounded;
                };

                required_memory += value_type_size * buffer_size_per_instance;
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
                    StorageRequirement::Finite(size) => *size as u128,
                    StorageRequirement::FutureRef(_) => unimplemented!(),
                    StorageRequirement::Unbounded => unimplemented!(),
                };
                let value_type = type_table.get_value_type(output.id);
                let value_type_size = if let MemoryBound::Bounded(i) = get_byte_size(value_type) {
                    i
                } else {
                    return MemoryBound::Unbounded;
                };

                required_memory += value_type_size * buffer_size_per_instance;

                //----------------------
                // tracking

                let tracking_requirement = tracking_requirements.get(&output.id)
                    .expect("We should have determined the tracking requirements for all streams that did not get pruned!");
                let mut it = tracking_requirement.iter();
                let mut tracking_per_instance: u128 = 0u128;
                while let Some((node_id, tracking)) = it.next() {
                    let value_type = type_table.get_value_type(*node_id);
                    let value_type_size = if let MemoryBound::Bounded(i) = get_byte_size(value_type)
                    {
                        i
                    } else {
                        return MemoryBound::Unbounded;
                    };

                    match tracking {
                        TrackingRequirement::Finite(size) => tracking_per_instance += (*size as u128)*value_type_size,
                        // TODO What about accessing a future dependent stream?
                        TrackingRequirement::Future => unimplemented!(),
                        TrackingRequirement::Unbounded => unimplemented!(),
                    }
                }

                required_memory += tracking_per_instance;
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
            let mut tracking_per_instance: u128 = 0u128;
            while let Some((node_id, tracking)) = it.next() {
                let value_type = type_table.get_value_type(*node_id);
                let value_type_size = if let MemoryBound::Bounded(i) = get_byte_size(value_type) {
                    i
                } else {
                    return MemoryBound::Unbounded;
                };

                match tracking {
                    TrackingRequirement::Finite(size) => tracking_per_instance += (*size as u128)*value_type_size,
                    // TODO What about accessing a future dependent stream?
                    TrackingRequirement::Future => unimplemented!(),
                    TrackingRequirement::Unbounded => unimplemented!(),
                }
            }

            required_memory += tracking_per_instance;
        }
    }

    return MemoryBound::Bounded(required_memory);
}
