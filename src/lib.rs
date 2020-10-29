pub mod vector4;
pub use vector4::*;

pub mod matrix4;
pub use matrix4::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const fn internal_mm_shuffle(fp3: i32, fp2: i32, fp1: i32, fp0: i32) -> i32 {
    ((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0)
}