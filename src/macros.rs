#[macro_export]
macro_rules! pyiter {
    ($clsname:ident (module = $module:literal, name = $name:literal) { $itemview:ident: $item:ident }) => {
        #[pyclass(module = $module, name = $name)]
        pub struct $clsname {
            lst: &'static [$item],
            iter: std::slice::Iter<'static, $item>,
        }

        impl $clsname {
            fn new(lst: &[$item]) -> Self {
                let lst2 = unsafe_update_view_lifetime_signature(lst);
                Self {
                    lst: lst2,
                    iter: lst2.iter(),
                }
            }
        }

        #[pymethods]
        impl $clsname {
            fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            fn __next__(&mut self) -> Option<$itemview> {
                if let Some(v) = self.iter.next() {
                    Some($itemview(unsafe_update_view_lifetime_signature(v)))
                } else {
                    None
                }
            }

            fn __len__(&self) -> usize {
                self.lst.len()
            }

            fn __getitem__(&self, i: usize) -> PyResult<$itemview> {
                if i < self.lst.len() {
                    Ok($itemview(unsafe_update_view_lifetime_signature(
                        &self.lst[i],
                    )))
                } else {
                    Err(pyo3::exceptions::PyIndexError::new_err(
                        "index out of range",
                    ))
                }
            }
        }
    };
}

#[macro_export]
macro_rules! pyview {
    ($viewname:ident (module = $module:literal, name = $name:literal, cls = $clsname:ident) {
        $(
            $(c($cel:ident: $cty:ty))?
            $(r($rel:ident: $rty:ty))?
            $(iter($itervec:ident { $iel:ident: $ity:ty }))?
        ),*
    }) => {
        #[pyclass(module = $module, name = $name)]
        pub struct $viewname(&'static $clsname);

        #[pymethods]
        impl $viewname {
            $(
                $(
                    #[getter]
                    fn $cel(&self) -> $cty {
                        self.0.$cel
                    }
                )?

                $(
                    #[getter]
                    fn $rel(&self) -> &$rty {
                        &self.0.$rel
                    }
                )?

                $(
                    fn $itervec(&self) -> $ity {
                        <$ity>::new(&self.0.$iel)
                    }
                )?
            )*
        }
    };
}

#[macro_export]
macro_rules! pywrap {
    ($wrapper:ident (module = $module:literal, name = $name:literal, cls = $clsname:ident) {
        $(
            $(c($cel:ident: $cty:ty))?
            $(r($rel:ident: $rty:ty))?
            $(iter($itervec:ident { $iel:ident: $ity:ty }))?
        ),*
    }) => {
        #[pyclass(module = $module, name = $name)]
        pub struct $wrapname(pub $clsname);

        #[pymethods]
        impl $wrapname {
            $(
                $(
                    #[getter]
                    fn $cel(&self) -> $cty {
                        self.0.$cel
                    }
                )?

                $(
                    #[getter]
                    fn $rel(&self) -> &$rty {
                        &self.0.$rel
                    }
                )?

                $(
                    fn $itervec(&self) -> $ity {
                        <$ity>::new(&self.0.$iel)
                    }
                )?
            )*
        }
    };
}

/// Change signature of a reference from temporary to static. This is unsafe and
/// only be used for temporary views that drop immediately after use.
pub fn unsafe_update_view_lifetime_signature<T: ?Sized>(val: &T) -> &'static T {
    let ptr = val as *const T;
    unsafe { &*ptr }
}
