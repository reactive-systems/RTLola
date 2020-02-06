use crate::type_checker::types::{Key, Ty, UnificationError, EvaluationTy, ValueTy, IntTy, PacingTy, ExpressionTy, ActivationCondition, Freq};
use type_checker::{UnifyValue, UnifyKey};

impl type_checker::UnifyKey for Key {
    type Value = Ty;

    fn index(&self) -> u32 {
        self.0
    }

    fn from_index(u: u32) -> Self {
        Self(0)
    }

    fn tag() -> &'static str {
        "Key"
    }
}

impl type_checker::UnifyValue for Ty {
    type Error = UnificationError;

    fn unify_values(lhs: &Self, rhs: &Self) -> Result<Self, Self::Error> {
        Ok(Ty {
            value: <ValueTy as type_checker::UnifyValue>::unify_values(&lhs.value, &rhs.value)?,
            eval: <EvaluationTy as type_checker::UnifyValue>::unify_values(&lhs.eval, &rhs.eval)?,
        })
    }
}

impl type_checker::UnifyValue for ValueTy {
    type Error = UnificationError;

    fn unify_values(lhs: &Self, rhs: &Self) -> Result<Self, Self::Error> {
        use ValueTy::*;
        use crate::type_checker::types::FloatTy;
        use crate::type_checker::types::UIntTy;
        use crate::type_checker::types::IntTy;
        match (lhs, rhs) {
            (Infer, a) | (a, Infer) => Ok(a.clone()),

            (Option(l), Option(r)) => Ok(<ValueTy as UnifyValue>::unify_values(l, r)?),
            (Option(_), _) | (_, Option(_)) => Err(UnificationError::IncompatibleValueTypes(lhs.clone(), rhs.clone())),

            (Tuple(ls), Tuple(rs)) if rs.len() == ls.len() => {
                let mut res = Vec::new();
                for (l, r) in ls.iter().zip(rs.iter()) {
                    let meet = <ValueTy as UnifyValue>::unify_values(l ,r)?;
                    res.push(meet);
                }
                Ok(Tuple(res))
            }
            (Tuple(_), _) | (_, Tuple(_)) => Err(UnificationError::IncompatibleValueTypes(lhs.clone(), rhs.clone())),

            (Numeric, Numeric) => Ok(Numeric),
            (Numeric, Comparable) | (Comparable, Numeric) => Ok(Numeric),
            (Numeric, Equatable) | (Equatable, Numeric) => Ok(Integer), // Floats are not equatable.
            (Numeric, Integer) | (Integer, Numeric) => Ok(Integer),
            (Numeric, Int(i)) | ((Int(i), Numeric)) => Ok(Int(*i)),
            (Numeric, UInt(u)) | (UInt(u), Numeric) => Ok(UInt(*u)),
            (Numeric, Float(f)) | (Float(f), Numeric) => Ok(Float(*f)),

            (Comparable, Comparable) => Ok(Comparable),
            (Comparable, Integer) | (Integer, Comparable) => Ok(Integer),
            (Comparable, Equatable) | (Equatable, Comparable) => Ok(Integer), // Equatable excludes Floats and thus Numeric
            (Comparable, Int(i)) | (Int(i), Comparable) => Ok(Int(*i)),
            (Comparable, UInt(u)) | (UInt(u), Comparable) => Ok(UInt(*u)),
            (Comparable, Float(f)) | (Float(f), Comparable) => Ok(Float(*f)),

            (Equatable, Equatable) => Ok(Equatable),
            (Equatable, Integer) | (Integer, Equatable) => Ok(Integer),
            (Equatable, Int(i)) | (Int(i), Equatable) => Ok(Int(*i)),
            (Equatable, UInt(u)) | (UInt(u), Equatable) => Ok(UInt(*u)),
            (Equatable, Float(f)) | (Float(f), Equatable) => Err(UnificationError::IncompatibleValueTypes(lhs.clone(), rhs.clone())),

            (Integer, Integer) => Ok(Integer),
            (Integer, Float(_)) | (Float(_), Integer) => Err(UnificationError::IncompatibleValueTypes(lhs.clone(), rhs.clone())),
            (Integer, Int(i)) | (Int(i), Integer) => Ok(Int(*i)),
            (Integer, UInt(u)) | (UInt(u), Integer) => Ok(UInt(*u)),

            (Int(i1), Int(i2)) => Ok(Int(IntTy::max(*i1, *i2))),
            (Float(f1), Float(f2)) => Ok(Float(crate::type_checker::types::FloatTy::max(*f1, *f2))),
            (UInt(u1), UInt(u2)) => Ok(UInt(UIntTy::max(*u1, *u2))),

            (Int(_), Float(_)) | (Float(_), Int(_)) | (Int(_), UInt(_)) | (UInt(_), Int(_)) | (Float(_), UInt(_)) | (UInt(_), Float(_)) => Err(UnificationError::IncompatibleValueTypes(lhs.clone(), rhs.clone())),

            (Bool, Bool) => Ok(Bool),
            (String, String) => Ok(String),
            (Bool, _) | (_, Bool) | (String, _) | (_, String) => Err(UnificationError::IncompatibleValueTypes(lhs.clone(), rhs.clone())),
//            _ => unimplemented!(),

        }
    }
}

impl type_checker::UnifyValue for EvaluationTy {
    type Error = UnificationError;

    fn unify_values(lhs: &Self, rhs: &Self) -> Result<Self, Self::Error> {
        Ok(EvaluationTy {
            pacing: <PacingTy as type_checker::UnifyValue>::unify_values(&lhs.pacing, &rhs.pacing)?,
            filter: <ExpressionTy as type_checker::UnifyValue>::unify_values(&lhs.filter, &rhs.filter)?,
            spawn: <ExpressionTy as type_checker::UnifyValue>::unify_values(&lhs.spawn, &rhs.spawn)?,
        })
    }
}

impl type_checker::UnifyValue for PacingTy {
    type Error = UnificationError;

    fn unify_values(lhs: &Self, rhs: &Self) -> Result<Self, Self::Error> {
        use PacingTy::*;
        match (lhs, rhs) {
            (Infer, x) | (x, Infer) => Ok(x.clone()),
            (Event(ac1), Event(ac2)) => Ok(Event(<ActiviationCondition as type_checker::UnifyValue>::unify_values(ac1, ac2)?)),
            (Periodic(p1), Periodic(p2))  => Ok(Periodic(<Freq as type_checker::UnifyValue>::unify_values(p1, p2)?)),
            (Periodic(p), Event(e)) | (Event(e), Periodic(p)) => Err(UnificationError::MixedEventPeriodic(e.clone(), p.clone())),
        }
    }
}

impl type_checker::UnifyValue for ExpressionTy {
    type Error = UnificationError;

    fn unify_values(lhs: &Self, rhs: &Self) -> Result<Self, Self::Error> {
        unimplemented!();

    }
}

impl<Var> type_checker::UnifyValue for ActivationCondition<Var> {
    type Error = UnificationError;

    fn unify_values(lhs: &Self, rhs: &Self) -> Result<Self, Self::Error> {

    }
}

impl type_checker::UnifyValue for Freq {
    type Error = UnificationError;

    fn unify_values(lhs: &Self, rhs: &Self) -> Result<Self, Self::Error> {
        if lhs.is_multiple_of(rhs) || rhs.is_multiple_of(lhs) {
            Ok(p1.conjunction(p2))
        } else {
            UnificationError::IncompatibleFrequencies(p1, p2)
        }
    }
}