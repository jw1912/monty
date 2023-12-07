use crate::Vector;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Matrix<const M: usize, const N: usize> {
    inner: [Vector<N>; M],
}

impl<const M: usize, const N: usize> std::ops::Deref for Matrix<M, N> {
    type Target = [Vector<N>; M];
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<const M: usize, const N: usize> std::ops::DerefMut for Matrix<M, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<const M: usize, const N: usize> std::ops::Mul<Vector<N>> for Matrix<M, N> {
    type Output = Vector<M>;
    fn mul(self, rhs: Vector<N>) -> Self::Output {
        Vector::<M>::from_fn(|i| self.inner[i].dot(&rhs))
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    pub const fn from_raw(inner: [Vector<N>; M]) -> Self {
        Self { inner }
    }
}
