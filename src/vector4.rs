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

impl AddAssign for SimdVector4 {
    fn add_assign(&mut self, rhs: SimdVector4) {
        *self = *self + rhs;
    }
}

impl Sub for SimdVector4 {
    type Output = SimdVector4;
    fn sub(self, rhs: SimdVector4) -> Self::Output {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let vector = x86_sub(&self.vector, &rhs.vector);

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let vector = self.vector - rhs.vector;

        SimdVector4 { vector }
    }
}

impl SubAssign for SimdVector4 {
    fn sub_assign(&mut self, rhs: SimdVector4) {
        *self = *self - rhs;
    }
}

impl Mul for SimdVector4 {
    type Output = SimdVector4;
    fn mul(self, rhs: SimdVector4) -> Self::Output {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let vector = x86_mul(&self.vector, &rhs.vector);

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let vector = self.vector.component_mul(rhs.vector);

        SimdVector4 { vector }
    }
}

impl MulAssign for SimdVector4 {
    fn mul_assign(&mut self, rhs: SimdVector4) {
        *self = *self * rhs;
    }
}

impl Div for SimdVector4 {
    type Output = SimdVector4;
    fn div(self, rhs: SimdVector4) -> Self::Output {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let vector = x86_div(&self.vector, &rhs.vector);

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let vector = self.vector.component_div(rhs.vector);

        SimdVector4 { vector }
    }
}

impl DivAssign for SimdVector4 {
    fn div_assign(&mut self, rhs: SimdVector4) {
        *self = *self / rhs;
    }
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn x86_add(a: &Vector4<f32>, b: &Vector4<f32>) -> Vector4<f32> {
    unsafe { x86_operation(|a, b| _mm_add_ps(a, b), a, b) }
}

fn x86_sub(a: &Vector4<f32>, b: &Vector4<f32>) -> Vector4<f32> {
    unsafe { x86_operation(|a, b| _mm_sub_ps(a, b), a, b) }
}

fn x86_mul(a: &Vector4<f32>, b: &Vector4<f32>) -> Vector4<f32> {
    unsafe { x86_operation(|a, b| _mm_mul_ps(a, b), a, b) }
}

fn x86_div(a: &Vector4<f32>, b: &Vector4<f32>) -> Vector4<f32> {
    unsafe { x86_operation(|a, b| _mm_div_ps(a, b), a, b) }
}

unsafe fn x86_operation<F: Fn(__m128, __m128) -> __m128>(
    f: F,
    a: &Vector4<f32>,
    b: &Vector4<f32>,
) -> Vector4<f32> {
    let a: __m128 = transmute(*a);
    let b: __m128 = transmute(*b);
    transmute(f(a, b))
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
        let mut c = a;
        c += b;

        assert!(result.relative_eq(&(a + b).into(), std::f32::EPSILON, std::f32::EPSILON));
        assert!(result.relative_eq(&c.into(), std::f32::EPSILON, std::f32::EPSILON));
    }

    #[test]
    fn sub_test() {
        let mut rng = thread_rng();
        let a = Vector4::<f32>::from_column_slice(&[rng.gen(), rng.gen(), rng.gen(), rng.gen()]);
        let b = Vector4::<f32>::from_column_slice(&[rng.gen(), rng.gen(), rng.gen(), rng.gen()]);

        let result = Vector4::from(a - b);

        let a = SimdVector4::from(a);
        let b = SimdVector4::from(b);
        let mut c = a;
        c -= b;

        assert!(result.relative_eq(&(a - b).into(), std::f32::EPSILON, std::f32::EPSILON));
        assert!(result.relative_eq(&c.into(), std::f32::EPSILON, std::f32::EPSILON));
    }

    #[test]
    fn mul_test() {
        let mut rng = thread_rng();
        let a = Vector4::<f32>::from_column_slice(&[rng.gen(), rng.gen(), rng.gen(), rng.gen()]);
        let b = Vector4::<f32>::from_column_slice(&[rng.gen(), rng.gen(), rng.gen(), rng.gen()]);

        let result = Vector4::from(a.component_mul(&b));

        let a = SimdVector4::from(a);
        let b = SimdVector4::from(b);
        let mut c = a;
        c *= b;

        assert!(result.relative_eq(&(a * b).into(), std::f32::EPSILON, std::f32::EPSILON));
        assert!(result.relative_eq(&c.into(), std::f32::EPSILON, std::f32::EPSILON));
    }

    #[test]
    fn div_test() {
        let mut rng = thread_rng();
        let a = Vector4::<f32>::from_column_slice(&[rng.gen(), rng.gen(), rng.gen(), rng.gen()]);
        let b = Vector4::<f32>::from_column_slice(&[rng.gen(), rng.gen(), rng.gen(), rng.gen()]);

        let result = Vector4::from(a.component_div(&b));

        let a = SimdVector4::from(a);
        let b = SimdVector4::from(b);
        let mut c = a;
        c /= b;

        assert!(result.relative_eq(&(a / b).into(), std::f32::EPSILON, std::f32::EPSILON));
        assert!(result.relative_eq(&c.into(), std::f32::EPSILON, std::f32::EPSILON));
    }
}
