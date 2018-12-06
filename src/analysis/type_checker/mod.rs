use super::naming::DeclarationTable;
use ast::LolaSpec;

mod candidates;
mod checker;

use super::common::Type;
use super::type_checker::checker::*;
use crate::reporting::Handler;
use ast_node::NodeId;

use std::collections::HashMap;

type TypeTable = HashMap<NodeId, Type>;

#[derive(Debug)]
pub(crate) struct TypeCheckResult {
    type_table: TypeTable,
}

pub(crate) fn type_check<'a>(
    dt: &'a DeclarationTable,
    spec: &'a LolaSpec,
    handler: &'a Handler,
) -> TypeCheckResult {
    TypeChecker::new(dt, spec, handler).check_spec()
}

#[cfg(test)]
mod tests {
    use analysis::id_assignment::*;
    use analysis::naming::*;
    use analysis::type_checker::type_check;
    use analysis::type_checker::TypeCheckResult;
    use parse::*;
    use reporting::Handler;
    use std::path::PathBuf;

    fn setup(spec: &str) -> (TypeCheckResult, Handler) {
        let handler = Handler::new(SourceMapper::new(PathBuf::new(), spec));
        let tcr;
        {
            // Resolve `Cannot move out of `handler` because it is borrowed`- error.
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
            tcr = type_check(&na.result, &spec, &handler);
        }
        (tcr, handler)
    }

    #[test]
    fn simple_input() {
        let spec = "input i: Int8";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn simple_const() {
        let spec = "constant c: Int8 := 3";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn simple_const_faulty() {
        let spec = "constant c: Int8 := true";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_signedness() {
        let spec = "constant c: UInt8 := -2";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_incorrect_float() {
        let spec = "constant c: UInt8 := 2.3";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn simple_output() {
        let spec = "output o: Int8 := 3";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn simple_binary() {
        let spec = "output o: Int8 := 3 + 5";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn simple_unary() {
        let spec = "output o: Bool := !false";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn simple_unary_faulty() {
        // The negation should return a bool even if the underlying expression is wrong.
        // Thus, there is only one error here.
        let spec = "output o: Bool := !3";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn simple_binary_faulty() {
        let spec = "output o: Float32 := false + 2.5";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn simple_ite() {
        let spec = "output o: Int8 := if false then 1 else 2";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn underspecified_ite_type() {
        let spec = "output o: Float64 := if false then 1.3 else -2";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_ite_condition_faulty() {
        let spec = "output o: UInt8 := if 3 then 1 else 1";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_ite_arms_incompatible() {
        let spec = "output o: UInt8 := if true then 1 else false";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_underspecified_type() {
        let spec = "output o: Float32 := 2";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_trigonometric() {
        let spec = "output o: Float32 := sin(2)";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_trigonometric_faulty() {
        let spec = "output o: UInt8 := cos(1)";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_trigonometric_faulty_2() {
        let spec = "output o: Float64 := cos(true)";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_input_lookup() {
        let spec = "input a: UInt8\n output b: UInt8 := a";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_input_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_stream_lookup() {
        let spec = "output a: UInt8 := 3\n output b: UInt8 := a[0]";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_stream_lookup_faulty() {
        let spec = "input a: UInt8\n output b: Float64 := a";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_stream_lookup_dft() {
        let spec = "output a: UInt8 := 3\n output b: UInt8 := a[0] ? 3";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_stream_lookup_dft_fault() {
        let spec = "output a: UInt8 := 3\n output b: Bool := a[0] ? false";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_invoke_type() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    #[ignore]
    fn test_invoke_type_faulty() {
        let spec = "input in: Bool\n output a<p1: Int8>: Int8 { invoke in } := 3";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_extend_type() {
        let spec = "input in: Bool\n output a: Int8 { extend in } := 3";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    #[ignore]
    fn test_extend_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { extend in } := 3";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_terminate_type() {
        let spec = "input in: Bool\n output a: Int8 { terminate in } := 3";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    #[ignore]
    fn test_terminate_type_faulty() {
        let spec = "input in: Int8\n output a: Int8 { terminate in } := 3";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_param_spec() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(3)[-2] ? 1";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_param_spec_faulty() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: Int8 := a(true)[-2] ? 1";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_lookup_incomp() {
        let spec = "input in: Int8\n output a<p1: Int8>: Int8 { invoke in } := 3\n output b: UInt8 := a(3)[2] ? 1";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_tuple() {
        let spec = "output out: (Int8, Bool) := (14, false)";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_tuple_faulty() {
        let spec = "output out: (Int8, Bool) := (14, 3)";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_tuple_access() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.1";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_tuple_access_faulty_type() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.0";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_tuple_access_faulty_len() {
        let spec = "input in: (Int8, Bool)\noutput out: Bool := in.2";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Tuple access larger than tuple was not recognized."
        );
    }

    #[test]
    fn test_input_offset() {
        let spec = "input a: UInt8\n output b: UInt8 := a[3]";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_window_widening() {
        let spec = "input in: Int8\n output out: Int64 {extend @5Hz}:= in[3s, Σ] ? 0";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_window() {
        let spec = "input in: Int8\n output out: Int8 {extend @5Hz} := in[3s, Σ] ? 0";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }

    #[test]
    fn test_window_untimed() {
        let spec = "input in: Int8\n output out: Int16 := in[3s, Σ] ? 5";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible timings were not recognized as such."
        );
    }

    #[test]
    fn test_window_faulty() {
        let spec = "input in: Int8\n output out: Bool {extend @5Hz} := in[3s, Σ] ? true";
        let (_, handler) = setup(spec);
        assert_eq!(
            handler.emitted_errors(),
            1,
            "Incompatible types were not recognized as such."
        );
    }

    #[test]
    fn test_involved() {
        let spec =
            "input velo: Float32\n output avg: Float64 {extend @5Hz} := velo[1h, avg] ? 10000";
        let (_, handler) = setup(spec);
        assert!(
            !handler.contains_error(),
            "There should not be a typing error."
        );
    }
}
