#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use nalgebra::*;
use std::mem::transmute;
use std::ops::*;

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SimdVector4 {
    pub vector: Vector4<f32>,
}

impl From<&Vector4<f32>> for SimdVector4 {
    fn from(vector: &Vector4<f32>) -> Self {
        Self { vector: *vector }
    }
}

impl From<Vector4<f32>> for SimdVector4 {
    fn from(vector: Vector4<f32>) -> Self {
        Self { vector }
    }
}

impl From<SimdVector4> for Vector4<f32> {
    fn from(v: SimdVector4) -> Self {
        v.vector
    }
}

impl From<&SimdVector4> for Vector4<f32> {
    fn from(v: &SimdVector4) -> Self {
        v.vector
    }
}

impl Add for SimdVector4 {
    type Output = SimdVector4;
    fn add(self, rhs: SimdVector4) -> Self::Output {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let vector = x86_add(&self.vector, &rhs.vector);

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let vector = self.vector + rhs.vector;

        SimdVector4 { vector }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_add(a: &Vector4<f32>, b: &Vector4<f32>) -> Vector4<f32> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        let a: __m128 = transmute(*a);
        let b: __m128 = transmute(*b);
        transmute(_mm_add_ps(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn add_test() {
        let mut rng = thread_rng();
        let a = Vector4::<f32>::from_column_slice(&[rng.gen(), rng.gen(), rng.gen(), rng.gen()]);
        let b = Vector4::<f32>::from_column_slice(&[rng.gen(), rng.gen(), rng.gen(), rng.gen()]);

        let result = Vector4::from(a + b);

        let a = SimdVector4::from(a);
        let b = SimdVector4::from(b);

        assert!(result.relative_eq(&(a + b).into(), std::f32::EPSILON, std::f32::EPSILON));
    }
}
