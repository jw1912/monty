use crate::activation::Activation;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vector<const N: usize> {
    inner: [f32; N],
}

impl<const N: usize> Default for Vector<N> {
    fn default() -> Self {
        Self::from_raw([0.0; N])
    }
}

impl<const N: usize> std::ops::Index<usize> for Vector<N> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<const N: usize> std::ops::Add<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn add(mut self, rhs: Vector<N>) -> Self::Output {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i += *j;
        }

        self
    }
}

impl<const N: usize> std::ops::Add<f32> for Vector<N> {
    type Output = Vector<N>;
    fn add(mut self, rhs: f32) -> Self::Output {
        for i in self.inner.iter_mut() {
            *i += rhs;
        }

        self
    }
}

impl<const N: usize> std::ops::AddAssign<Vector<N>> for Vector<N> {
    fn add_assign(&mut self, rhs: Vector<N>) {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i += *j;
        }
    }
}

impl<const N: usize> std::ops::Div<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn div(mut self, rhs: Vector<N>) -> Self::Output {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i /= *j;
        }

        self
    }
}

impl<const N: usize> std::ops::Mul<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn mul(mut self, rhs: Vector<N>) -> Self::Output {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i *= *j;
        }

        self
    }
}

impl<const N: usize> std::ops::Mul<Vector<N>> for f32 {
    type Output = Vector<N>;
    fn mul(self, mut rhs: Vector<N>) -> Self::Output {
        for i in rhs.inner.iter_mut() {
            *i *= self;
        }

        rhs
    }
}

impl<const N: usize> std::ops::SubAssign<Vector<N>> for Vector<N> {
    fn sub_assign(&mut self, rhs: Vector<N>) {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i -= *j;
        }
    }
}

impl<const N: usize> Vector<N> {
    pub fn dot(&self, other: &Vector<N>) -> f32 {
        let mut score = 0.0;
        for (&i, &j) in self.inner.iter().zip(other.inner.iter()) {
            score += i * j;
        }

        score
    }

    pub fn out<T: Activation>(&self, other: &Vector<N>) -> f32 {
        let mut score = 0.0;
        for (i, j) in self.inner.iter().zip(other.inner.iter()) {
            score += T::activate(*i) * T::activate(*j);
        }

        score
    }

    pub fn sqrt(mut self) -> Self {
        for i in self.inner.iter_mut() {
            *i = i.sqrt();
        }

        self
    }

    pub const fn from_raw(inner: [f32; N]) -> Self {
        Self { inner }
    }

    pub fn activate<T: Activation>(mut self) -> Self {
        for i in self.inner.iter_mut() {
            *i = T::activate(*i);
        }

        self
    }

    pub fn derivative<T: Activation>(mut self) -> Self {
        for i in self.inner.iter_mut() {
            *i = T::derivative(*i);
        }

        self
    }
}

#[derive(Clone, Copy)]
pub struct Matrix<const M: usize, const N: usize> {
    inner: [Vector<N>; M],
}

impl<const M: usize, const N: usize> std::ops::Mul<Vector<N>> for Matrix<M, N> {
    type Output = Vector<M>;
    fn mul(self, rhs: Vector<N>) -> Self::Output {
        let mut res = Vector::<M>::default();

        for (i, row) in self.inner.iter().enumerate() {
            res.inner[i] = row.dot(&rhs);
        }

        res
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    pub const fn from_raw(inner: [Vector<N>; M]) -> Self {
        Self { inner }
    }
}
