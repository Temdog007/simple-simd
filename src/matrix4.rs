#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::*;
use nalgebra::*;
use std::mem::transmute;
use std::ops::*;

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SimdMatrix4 {
    pub matrix4: Matrix4<f32>,
}

impl From<&Matrix4<f32>> for SimdMatrix4 {
    fn from(matrix4: &Matrix4<f32>) -> Self {
        Self { matrix4: *matrix4 }
    }
}

impl From<Matrix4<f32>> for SimdMatrix4 {
    fn from(matrix4: Matrix4<f32>) -> Self {
        Self { matrix4 }
    }
}

impl From<SimdMatrix4> for Matrix4<f32> {
    fn from(v: SimdMatrix4) -> Self {
        v.matrix4
    }
}

impl From<&SimdMatrix4> for Matrix4<f32> {
    fn from(v: &SimdMatrix4) -> Self {
        v.matrix4
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn mul_column_row(in1: &[__m128; 4], in2: __m128) -> __m128 {
    let e0 = _mm_shuffle_ps(in2, in2, internal_mm_shuffle(0, 0, 0, 0));
    let e1 = _mm_shuffle_ps(in2, in2, internal_mm_shuffle(1, 1, 1, 1));
    let e2 = _mm_shuffle_ps(in2, in2, internal_mm_shuffle(2, 2, 2, 2));
    let e3 = _mm_shuffle_ps(in2, in2, internal_mm_shuffle(3, 3, 3, 3));

    let m0 = _mm_mul_ps(*in1.get_unchecked(0), e0);
    let m1 = _mm_mul_ps(*in1.get_unchecked(1), e1);
    let m2 = _mm_mul_ps(*in1.get_unchecked(2), e2);
    let m3 = _mm_mul_ps(*in1.get_unchecked(3), e3);

    let a0 = _mm_add_ps(m0, m1);
    let a1 = _mm_add_ps(m2, m3);
    _mm_add_ps(a0, a1)
}

impl Add for SimdMatrix4 {
    type Output = SimdMatrix4;
    fn add(self, rhs: SimdMatrix4) -> SimdMatrix4 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            let in1: [SimdVector4; 4] = transmute(self);
            let in2: [SimdVector4; 4] = transmute(rhs);
            transmute([
                *in1.get_unchecked(0) + *in2.get_unchecked(0),
                *in1.get_unchecked(1) + *in2.get_unchecked(1),
                *in1.get_unchecked(2) + *in2.get_unchecked(2),
                *in1.get_unchecked(3) + *in2.get_unchecked(3),
            ])
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.matrix4 * rhs.matrix4
        }
    }
}

impl Mul for SimdMatrix4 {
    type Output = SimdMatrix4;
    fn mul(self, rhs: SimdMatrix4) -> SimdMatrix4 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            let in1: [__m128; 4] = transmute(self);
            let in2: [__m128; 4] = transmute(rhs);
            transmute([
                mul_column_row(&in1, *in2.get_unchecked(0)),
                mul_column_row(&in1, *in2.get_unchecked(1)),
                mul_column_row(&in1, *in2.get_unchecked(2)),
                mul_column_row(&in1, *in2.get_unchecked(3)),
            ])
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.matrix4 * rhs.matrix4
        }
    }
}

impl SimdMatrix4 {
    pub fn transpose(&self) -> SimdMatrix4 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            let input: [__m128; 4] = transmute(*self);
            let tmp0 = _mm_shuffle_ps(*input.get_unchecked(0), *input.get_unchecked(1), 0x44);
            let tmp2 = _mm_shuffle_ps(*input.get_unchecked(0), *input.get_unchecked(1), 0xEE);
            let tmp1 = _mm_shuffle_ps(*input.get_unchecked(2), *input.get_unchecked(3), 0x44);
            let tmp3 = _mm_shuffle_ps(*input.get_unchecked(2), *input.get_unchecked(3), 0xEE);

            transmute([
                _mm_shuffle_ps(tmp0, tmp1, 0x88),
                _mm_shuffle_ps(tmp0, tmp1, 0xDD),
                _mm_shuffle_ps(tmp2, tmp3, 0x88),
                _mm_shuffle_ps(tmp2, tmp3, 0xDD),
            ])
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.matrix4.transpose()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn add_test(){
        let mut rng = thread_rng();
        let a = Matrix4::<f32>::from_fn(|_, _| rng.gen());
        let b = Matrix4::<f32>::from_fn(|_, _| rng.gen());

        let result = a + b;

        let a = SimdMatrix4::from(a);
        let b = SimdMatrix4::from(b);
        assert!(result.relative_eq(&(a + b).into(), std::f32::EPSILON, std::f32::EPSILON));
    }

    #[test]
    fn mul_test() {
        let mut rng = thread_rng();
        let a = Matrix4::<f32>::from_fn(|_, _| rng.gen());
        let b = Matrix4::<f32>::from_fn(|_, _| rng.gen());

        let result = a * b;

        let a = SimdMatrix4::from(a);
        let b = SimdMatrix4::from(b);
        assert!(result.relative_eq(&(a * b).into(), std::f32::EPSILON, std::f32::EPSILON));
    }

    #[test]
    fn transpose_test() {
        let mut rng = thread_rng();
        let a = Matrix4::<f32>::from_fn(|_, _| rng.gen());

        let result = a.transpose();

        let a = SimdMatrix4::from(a).transpose();
        assert!(result.relative_eq(&a.into(), std::f32::EPSILON, std::f32::EPSILON));
    }
}
