use super::naming::DeclarationTable;
use ast::LolaSpec;

mod candidates;
mod checker;
mod type_error;

use super::common::Type;
use super::type_checker::checker::*;
use super::type_checker::type_error::TypeError;
use ast_node::NodeId;

use std::collections::HashMap;

type TypeTable = HashMap<NodeId, Type>;

#[derive(Debug)]
pub(crate) struct TypeCheckResult<'a> {
    type_table: TypeTable,
    errors: Vec<Box<TypeError<'a>>>,
}

pub(crate) fn type_check<'a>(dt: &'a DeclarationTable, spec: &'a LolaSpec) -> TypeCheckResult<'a> {
    let mut tc = TypeChecker::new(dt, spec);
    let tc_res = tc.check();
    tc_res.errors.iter().for_each(|e| println!("{:?}", e)); // TODO: pretty print.
    tc_res
}

#[cfg(test)]
mod tests {
    use analysis::id_assignment::*;
    use analysis::naming::*;
    use analysis::type_checker::type_check;
    use analysis::type_checker::type_error::TypeError;
    use parse::*;

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert!(res.errors.is_empty(), "There should not be a typing error.");
    }

    #[test]
    fn simple_const() {
        let spec = "constant c: Int8 := 3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert!(res.errors.is_empty(), "There should not be a typing error.");
    }

    #[test]
    fn simple_const_faulty() {
        let spec = "constant c: Int8 := true";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should not be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_signedness() {
        let spec = "constant c: UInt8 := -2";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_incorrect_float() {
        let spec = "constant c: UInt8 := 2.3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should not be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn simple_output() {
        let spec = "output o: Int8 := 3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn simple_binary() {
        let spec = "output o: Int8 := 3 + 5";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn simple_unary() {
        let spec = "output o: Bool := !false";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn simple_unary_faulty() {
        // The negation should return a bool even if the underlying expression is wrong.
        // Thus, there is only one error here.
        let spec = "output o: Bool := !3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn simple_binary_faulty() {
        let spec = "output o: Int8 := false + 2.5";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn simple_ite() {
        let spec = "output o: Int8 := if false then 1 else 2";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn underspecified_ite_type() {
        let spec = "output o: Float64 := if false then 1.3 else -2";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_ite_condition_faulty() {
        let spec = "output o: UInt8 := if 3 then 1 else 1";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_ite_arms_incompatible() {
        let spec = "output o: UInt8 := if true then 1 else false";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            2,
            "There should be exactly one typing error."
        );
    }

    #[test]
    fn test_underspecified_type() {
        let spec = "output o: Float32 := 2";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_trigonometric() {
        let spec = "output o: Float32 := sin(2)";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_trigonometric_faulty() {
        let spec = "output o: UInt8 := cos(1)";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_trigonometric_faulty_2() {
        let spec = "output o: Float64 := cos(true)";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::InvalidArgument { .. } => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_input_lookup() {
        let spec = "input a: UInt8\n output b: UInt8 := a";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_input_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_stream_lookup() {
        let spec = "output a: UInt8 := 3\n output b: UInt8 := a[0]";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_stream_lookup_fault() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_stream_lookup_dft() {
        let spec = "output a: UInt8 := 3\n output b: UInt8 := a[0] ? 3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_stream_lookup_dft_fault() {
        let spec = "output a: UInt8 := 3\n output b: Bool := a[0] ? false";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_invoke_type() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    #[ignore]
    fn test_invoke_type_faulty() {
        let spec = "input in: Bool\n output a<p1: Int8>: Int8 { invoke in } := 3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_extend_type() {
        let spec = "input in: Bool\n output a: Int8 { extend in } := 3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    #[ignore]
    fn test_extend_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { extend in } := 3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_terminate_type() {
        let spec = "input in: Bool\n output a: Int8 { terminate in } := 3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    #[ignore]
    fn test_terminate_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { terminate in } := 3";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_param_spec() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(3)[-2] ? 1";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_param_spec_faulty() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(true)[-2] ? 1";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_lookup_incomp() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: UInt8 := a(3)[2] ? 1";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_tuple() {
        let spec = "output out: (Int8, Bool) := (14, false)";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_tuple_faulty() {
        let spec = "output out: (Int8, Bool) := (14, 3)";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3]";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_window_widening() {
        let spec = "input in: Int8\n output out: Int64 := in[3s, Σ] ? 0";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 := in[3s, Σ] ? 0";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

    #[test]
    fn test_window_faulty() {
        let spec = "input in: Int8\n output out: Bool := in[3s, Σ] ? true";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(
            res.errors.len(),
            1,
            "There should be exactly one typing error."
        );
        match *res.errors[0] {
            TypeError::IncompatibleTypes(_, _) => {}
            _ => assert!(false, "Incompatible types were not recognized as such."),
        }
    }

    #[test]
    fn test_involved() {
        let spec = "input velo: Float32\n output avg: Float32 := velo[1h, avg] ? 10000\n trigger avg < 500";
        let mut spec = match parse(spec) {
            Err(e) => panic!("Spec {} cannot be parsed: {}.", spec, e),
            Ok(s) => s,
        };
        assign_ids(&mut spec);
        let mut na = NamingAnalysis::new();
        na.check(&spec);
        let res = type_check(&na.result, &spec);
        assert_eq!(res.errors.len(), 0, "There should not be a typing error.");
    }

}
