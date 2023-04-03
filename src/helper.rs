pub struct ByValue;
pub struct ByReference;

pub trait ReturnKind<'a, T: Sized + 'a> {
    type Type: Sized;
}

impl<'a, T: Sized + 'a> ReturnKind<'a, T> for ByValue {
    type Type = T;
}

impl<'a, T: Sized + 'a> ReturnKind<'a, T> for ByReference {
    type Type = &'a T;
}
