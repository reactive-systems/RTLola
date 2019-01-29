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

mod tests {
    use super::gcd;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(3, 18), 3);
        assert_eq!(gcd(18, 3), 3);
        assert_eq!(gcd(1, 25), 1);
        assert_eq!(gcd(5, 13), 1);
        assert_eq!(gcd(25, 40), 5);
        assert_eq!(gcd(7, 7), 7);
    }
}
