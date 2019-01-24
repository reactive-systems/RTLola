//! Parser for the Lola language.

#![deny(unsafe_code)] // disallow unsafe code by default
#![forbid(unused_must_use)] // disallow discarding errors

extern crate log;

#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate lazy_static;

mod analysis;
pub mod app;
mod ast;
mod ir;
mod lowering;
mod parse;
mod print;
mod reporting;
mod stdlib;
mod ty;

pub trait LolaBackend {
    /// Returns collection of feature flags supported by the `LolaBackend`.
    fn supported_feature_flags() -> Vec<FeatureFlag>;
}

// Replace by more elaborate interface.
pub fn parse(spec_str: &str) -> LolaIR {
    let spec = match crate::parse::parse(&spec_str) {
        Result::Ok(spec) => spec,
        Result::Err(e) => panic!("{}", e),
    };
    let mapper = crate::parse::SourceMapper::new(std::path::PathBuf::new(), spec_str);
    let handler = reporting::Handler::new(mapper);
    let analysis_result = analysis::analyze(&spec, &handler);
    if analysis_result.is_success() {
        lowering::Lowering::new(&spec, &analysis_result).lower()
    } else {
        panic!("Error in analysis.")
    }
}

// Re-export on the root level
pub use crate::ir::*;
pub use crate::ty::{FloatTy, IntTy, UIntTy};

use num::BigRational;
use num::ToPrimitive;
use num::{FromPrimitive, Integer, Signed, Zero};
use std::ops::Mul;

fn convert_to_f64(input: &BigRational) -> f64 {
    // look at https://www.exploringbinary.com/correct-decimal-to-floating-point-using-big-integers/

    if input.is_zero() {
        return 0.0f64;
    }

    if input.is_negative() {
        return -convert_to_f64(&-input);
    }

    // at this point we do have a positive number

    let two = BigRational::from_u64(2).unwrap();
    let upper_bound = BigRational::from_u64(2u64.pow(53)).unwrap();

    let mut exponent = 0; //-k
    let mut scaled_value = input.clone();
    if *input < upper_bound {
        loop {
            let new_value = &scaled_value * &two;
            if new_value < upper_bound {
                scaled_value = new_value;
                exponent += 1;
            } else {
                break;
            }
            if exponent == (80 + 52) {
                // round small values towards 0
                return 0.;
            }
        }
    } else {
        unimplemented!()
    }
    let fraction = scaled_value.fract();
    let remainder = fraction.numer();
    let twice_remainder = remainder.mul(2);
    let denominator = scaled_value.denom();
    let mut quotient = scaled_value.trunc().to_integer();

    // round half to even rounding from IEEE 754
    if twice_remainder > *denominator || (quotient.is_odd() && twice_remainder == *denominator) {
        quotient += 1;
    }

    // we rounded up to 2^53 therefore make this 2^52 and decrement the exponent
    if quotient == upper_bound.to_integer() {
        quotient /= 2;
        exponent -= 1;
    }

    let mut bits = quotient.to_u64().unwrap();
    bits -= 2u64.pow(52);
    let biased_exponent = 1023 + 52 - exponent;
    bits += biased_exponent * 2u64.pow(52); // shift the biased exponent by 52 bits

    f64::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use crate::convert_to_f64;
    use num::{BigRational, Zero};

    #[test]
    fn large_numbers_convert_to_f64() {
        #![allow(clippy::float_cmp)]
        assert_eq!(
            convert_to_f64(&BigRational::from_float(0.1).unwrap()),
            0.1_f64
        )
    }

    #[test]
    #[ignore]
    fn too_large_numbers_convert_to_inf_f64() {
        unimplemented!()
    }

    #[test]
    fn zero_converts_to_f64() {
        #![allow(clippy::float_cmp)]
        assert_eq!(convert_to_f64(&BigRational::zero()), 0.0_f64)
    }
    #[test]
    fn small_positives_converts_to_zero_f64() {
        #![allow(clippy::float_cmp)]
        assert_eq!(
            convert_to_f64(&BigRational::from_float(0.1e-80).unwrap()),
            0.0_f64
        )
    }

    #[test]
    fn negative_numbers_convert_to_f64() {
        #![allow(clippy::float_cmp)]
        assert_eq!(
            convert_to_f64(&BigRational::from_float(-0.1).unwrap()),
            -0.1_f64
        )
    }

}
