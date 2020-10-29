#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::*;
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

impl SimdVector4 {
    pub fn dot(&self, rhs: &SimdVector4) -> SimdVector4 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            let mul0 = _mm_mul_ps(transmute(self.vector), transmute(rhs.vector));
            let swp0 = _mm_shuffle_ps(mul0, mul0, internal_mm_shuffle(2, 3, 0, 1));
            let add0 = _mm_add_ps(mul0, swp0);
            let swp1 = _mm_shuffle_ps(add0, add0, internal_mm_shuffle(0, 1, 2, 3));
            transmute(_mm_add_ps(add0, swp1))
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.vector.dot(&rhs.vector)
        }
    }
    pub fn normalize(&self) -> SimdVector4 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            let dot0 = transmute(self.dot(self));
            let isr0 = _mm_rsqrt_ps(dot0);
            transmute(_mm_mul_ps(transmute(*self), isr0))
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.vector.normalize()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::relative_eq;
    use rand::prelude::*;

    #[test]
    fn add_test() {
        let mut rng = thread_rng();
        let a = Vector4::<f32>::from_fn(|_, _| rng.gen());
        let b = Vector4::<f32>::from_fn(|_, _| rng.gen());

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
        let a = Vector4::<f32>::from_fn(|_, _| rng.gen());
        let b = Vector4::<f32>::from_fn(|_, _| rng.gen());

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
        let a = Vector4::<f32>::from_fn(|_, _| rng.gen());
        let b = Vector4::<f32>::from_fn(|_, _| rng.gen());

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
        let a = Vector4::<f32>::from_fn(|_, _| rng.gen());
        let b = Vector4::<f32>::from_fn(|_, _| rng.gen());

        let result = Vector4::from(a.component_div(&b));

        let a = SimdVector4::from(a);
        let b = SimdVector4::from(b);
        let mut c = a;
        c /= b;

        assert!(result.relative_eq(&(a / b).into(), std::f32::EPSILON, std::f32::EPSILON));
        assert!(result.relative_eq(&c.into(), std::f32::EPSILON, std::f32::EPSILON));
    }

    #[test]
    fn dot_test() {
        let mut rng = thread_rng();
        let a = Vector4::<f32>::from_fn(|_, _| rng.gen());
        let b = Vector4::<f32>::from_fn(|_, _| rng.gen());

        let result = a.dot(&b);

        let a = SimdVector4::from(a);
        let b = SimdVector4::from(b);
        relative_eq!(result, a.dot(&b).vector.x);
    }

    #[test]
    fn noramlize_test() {
        let mut rng = thread_rng();
        let a = Vector4::<f32>::from_fn(|_, _| rng.gen());

        let result = a.normalize();

        let a = SimdVector4::from(a);
        result.relative_eq(&a.normalize().into(), std::f32::EPSILON, std::f32::EPSILON);
    }
}
