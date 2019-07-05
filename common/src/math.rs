use num::integer::gcd as num_gcd;
use num::integer::lcm as num_lcm;
use num::rational::Rational64 as Rational;

pub(crate) fn rational_gcd(a: Rational, b: Rational) -> Rational {
    let numer = num_gcd(*a.numer(), *b.numer());
    let denom = num_lcm(*a.denom(), *b.denom());
    Rational::new(numer, denom)
}

pub(crate) fn rational_lcm(a: Rational, b: Rational) -> Rational {
    let numer = num_lcm(*a.numer(), *b.numer());
    let denom = num_gcd(*a.denom(), *b.denom());
    Rational::new(numer, denom)
}

pub(crate) fn rational_gcd_all(v: &[Rational]) -> Rational {
    assert!(!v.is_empty());
    v.iter().fold(v[0], |a, b| rational_gcd(a, *b))
}

pub(crate) fn rational_lcm_all(v: &[Rational]) -> Rational {
    assert!(!v.is_empty());
    v.iter().fold(v[0], |a, b| rational_lcm(a, *b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::FromPrimitive;

    macro_rules! rat {
        ($i:expr) => {
            Rational::from_i64($i).unwrap()
        };
        ($n:expr, $d:expr) => {
            Rational::new($n, $d)
        };
    }
    #[test]
    fn test_gcd() {
        assert_eq!(rational_gcd(rat!(3), rat!(18)), rat!(3));
        assert_eq!(rational_gcd(rat!(18), rat!(3)), rat!(3));
        assert_eq!(rational_gcd(rat!(1), rat!(25)), rat!(1));
        assert_eq!(rational_gcd(rat!(5), rat!(13)), rat!(1));
        assert_eq!(rational_gcd(rat!(25), rat!(40)), rat!(5));
        assert_eq!(rational_gcd(rat!(7), rat!(7)), rat!(7));
        assert_eq!(rational_gcd(rat!(7), rat!(7)), rat!(7));

        assert_eq!(rational_gcd(rat!(1, 4), rat!(1, 2)), rat!(1, 4));
        assert_eq!(rational_lcm(rat!(1, 4), rat!(1, 2)), rat!(1, 2));
        assert_eq!(rational_gcd(rat!(2, 3), rat!(1, 8)), rat!(1, 24));
        assert_eq!(rational_lcm(rat!(2, 3), rat!(1, 8)), rat!(2));
    }
}
