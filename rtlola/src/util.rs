
use std::collections::HashMap;

pub(crate) fn gcd(mut a: u128, mut b: u128) -> u128 {
    // Courtesy of wikipedia.
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

mod tests {
    use crate::util;

    #[test]
    fn test_gcd() {
        assert_eq!(util::gcd(3, 18), 3);
        assert_eq!(util::gcd(18, 3), 3);
        assert_eq!(util::gcd(1, 25), 1);
        assert_eq!(util::gcd(5, 13), 1);
        assert_eq!(util::gcd(25, 40), 5);
        assert_eq!(util::gcd(7, 7), 7);
    }
}