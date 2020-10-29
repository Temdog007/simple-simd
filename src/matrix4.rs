#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::*;
use nalgebra::*;
use std::mem::transmute;

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

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::relative_eq;
    use rand::prelude::*;

    
}
