use num::integer::gcd as num_gcd;
use num::integer::lcm as num_lcm;
use num::rational::Rational64 as Rational;

pub(crate) fn gcd(mut a: u128, mut b: u128) -> u128 {
    // Courtesy of wikipedia.
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

pub(crate) fn lcm(a: u128, b: u128) -> u128 {
    // Courtesy of wikipedia.
    let mul = a * b;
    mul / gcd(a, b)
}

pub(crate) fn gcd_all(v: &[u128]) -> u128 {
    assert!(!v.is_empty());
    v.iter().fold(v[0], |a, b| gcd(a, *b))
}

pub(crate) fn lcm_all(v: &[u128]) -> u128 {
    assert!(!v.is_empty());
    v.iter().fold(v[0], |a, b| lcm(a, *b))
}

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

    #[test]
    fn test_gcd() {
        assert_eq!(super::gcd(3, 18), 3);
        assert_eq!(super::gcd(18, 3), 3);
        assert_eq!(super::gcd(1, 25), 1);
        assert_eq!(super::gcd(5, 13), 1);
        assert_eq!(super::gcd(25, 40), 5);
        assert_eq!(super::gcd(7, 7), 7);
    }
}
