//! # Matrix Determinant and Adjugate Expansion Utilities
//!
//! This module provides functionality for computing polynomial expansions of matrix
//! determinants and adjugates for linear parameter-dependent matrices of the form:
//! M(μ) = A + Bμ where A and B are constant matrices.
//!
//! ### Key Types and Functions
//!
//! `PolynomialCoefficientsFixedLength<T, const LEN: usize>`
//! 
//! A generic container for fixed-length polynomial coefficients. It has methods: `.iter`, `.pow` and `.mul_polynomial`.
//!
//! `DeterminantAndAdjugateExpansions1Parameter<const SIZE: usize, const DEGREE: usize, const DET_LEN: usize, const ADJ_LEN: usize>`
//! 
//! Main struct containing both determinant and adjugate polynomial expansions for a
//! matrix that depends linearly on one parameter (μ).
//!
//! ### Important Constructors:
//!
//! 1. **For 2x2 matrices:**
//!    ```rust
//!    impl DeterminantAndAdjugateExpansions1Parameter<2, 1, 3, 2> {
//!        /// Creates expansions for M(μ) = A + Bμ
//!        /// Returns struct with:
//!        /// - Determinant coefficients [c0, c1, c2] for det(M) = c0 + c1μ + c2μ²
//!        /// - Adjugate coefficients [c0, c1] for adj(M) = c0 + c1μ
//!        fn new_from_matrix(a: &Matrix2<f64>, b: &Matrix2<f64>) -> Self
//!    }
//!    ```
//!
//! 2. **For 3x3 matrices:**
//!    ```rust
//!    impl DeterminantAndAdjugateExpansions1Parameter<3, 1, 4, 3> {
//!        /// Creates expansions for M(μ) = A + Bμ
//!        /// Returns struct with:
//!        /// - Determinant coefficients [c0, c1, c2, c3] for det(M) = c0 + c1μ + c2μ² + c3μ³
//!        /// - Adjugate coefficients [c0, c1, c2] for adj(M) = c0 + c1μ + c2μ²
//!        fn new_from_matrix(a: &Matrix3<f64>, b: &Matrix3<f64>) -> Self
//!    }
//!    ```
//!
//! ### Type Aliases
//! - `DeterminantExpansion1Parameter`: Polynomial coefficients for determinant expansion
//! - `AdjugateExpansion1Parameter`: Polynomial coefficients for adjugate expansion
//!
//! ### Usage Example
//! ```rust
//! // For a 2x2 matrix M(μ) = A + Bμ
//! let a = Matrix2::new(1.0, 2.0, 3.0, 4.0);
//! let b = Matrix2::new(0.5, 0.5, 0.5, 0.5);
//! let expansions = DeterminantAndAdjugateExpansions1Parameter::new_from_matrix(&a, &b);
//!
//! // Access determinant coefficients:
//! let det_coeffs = expansions.determinant.iter();
//! // det(M(μ)) = det_coeffs[0] + det_coeffs[1]*μ + det_coeffs[2]*μ²
//!
//! // Access adjugate coefficients:
//! let adj_coeffs = expansions.adjugate.iter();
//! // adj(M(μ)) = adj_coeffs[0] + adj_coeffs[1]*μ
//! ```
//!
//! ### Note
//! The implementations are specialized for linear parameter dependence (DEGREE=1)
//! and for 2x2 and 3x3 matrices. The length parameters (DET_LEN, ADJ_LEN) must
//! match the expected polynomial lengths for the given matrix size and degree.
//! 
//! 
//! # Inverse Determinant Computation Utilities
//!
//! The `InverseDeterminant3x3` struct provides functionality for computing power series
//! expansions of the inverse determinant (1/det(M(μ))) for 3x3 matrices with linear
//! parameter dependence (M(μ) = A + Bμ).
//!
//! ### Key Methods:
//!
//! 1. **Power Series Coefficients (Vec version):**
//!    ```rust
//!    /// Computes coefficients of 1/det(M(μ)) as a power series up to specified order
//!    /// 
//!    /// Input:
//!    /// - `determinant_expansion`: Polynomial coefficients [c0, c1, c2, c3] where
//!    ///   det(M(μ)) = c0 + c1μ + c2μ² + c3μ³
//!    /// - `maximum_order`: Highest order term to compute
//!    ///
//!    /// Returns:
//!    /// - `Some(PowerSeriesCoefficientsVec)` containing coefficients [a0, a1, ...]
//!    ///   where 1/det(M(μ)) = a0 + a1μ + a2μ² + ...
//!    /// - `None` if matrix is singular (c0 = 0)
//!    pub fn power_series_coefficients_vec(
//!        determinant_expansion: &DeterminantExpansion1Parameter<3,1,4>,
//!        maximum_order: u8
//!    ) -> Option<PowerSeriesCoefficientsVec<f64>>
//!    ```
//!
//! 2. **Fixed-Length Polynomial Coefficients:**
//!    ```rust
//!    /// Computes coefficients of 1/det(M(μ)) as a fixed-length polynomial
//!    ///
//!    /// Input:
//!    /// - `determinant_expansion`: Polynomial coefficients [c0, c1, c2, c3] where
//!    ///   det(M(μ)) = c0 + c1μ + c2μ² + c3μ³
//!    ///
//!    /// Returns:
//!    /// - `Some(PolynomialCoefficientsFixedLength)` with LEN coefficients
//!    /// - `None` if matrix is singular (c0 = 0) or LEN = 0
//!    pub fn polynomial_coefficients_fixed_length<const LEN: usize>(
//!        determinant_expansion: &DeterminantExpansion1Parameter<3,1,4>
//!    ) -> Option<PolynomialCoefficientsFixedLength<f64, LEN>>
//!    ```
//!
//! ### Usage Example:
//! ```rust
//! // Given determinant expansion coefficients for a 3x3 matrix
//! let det_coeffs = PolynomialCoefficientsFixedLength([2.0, 1.0, 0.5, 0.1]);
//!
//! // Compute first 5 terms of inverse determinant series
//! let inv_det_series = InverseDeterminant3x3::power_series_coefficients_vec(
//!     &det_coeffs,
//!     5
//! ).unwrap();
//!
//! // Or get fixed-length polynomial approximation
//! let inv_det_poly = InverseDeterminant3x3::polynomial_coefficients_fixed_length::<4>(
//!     &det_coeffs
//! ).unwrap();
//! ```
//!
//! ### Implementation Notes:
//! - Uses a power series expansion of 1/(c0 + c1μ + c2μ² + c3μ³)
//! - Efficiently computes coefficients using multinomial theorem
//! - Precomputes factorials for better performance with higher orders
//! - Handles singular matrices (c0 = 0) by returning None
//! - Numerically stable through use of `recip()` instead of direct division

use nalgebra::{Matrix3, Matrix2, SMatrix};

// Function to compute the adjugate of a 2x2 matrix
fn adjugate2x2(m: &Matrix2<f64>) -> Matrix2<f64> {
    Matrix2::new(
        m[(1, 1)],  -m[(0, 1)],
       -m[(1, 0)],   m[(0, 0)],
    )
}

#[derive(Clone, Debug)]
pub struct PolynomialCoefficientsFixedLength<T, const LEN: usize> ([T; LEN]);

impl<T, const LEN: usize> PolynomialCoefficientsFixedLength<T, LEN> {
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }
}

type DeterminantExpansion1Parameter<const SIZE: usize, const DEGREE: usize, const LEN: usize>
    = PolynomialCoefficientsFixedLength<f64, LEN>; // LEN = SIZE * DEGREE + 1

type AdjugateExpansion1Parameter<const SIZE: usize, const DEGREE: usize, const LEN: usize>
    = PolynomialCoefficientsFixedLength<SMatrix<f64, SIZE, SIZE>, LEN>; // LEN = (SIZE - 1) * DEGREE + 1

pub struct DeterminantAndAdjugateExpansions1Parameter<const SIZE: usize, const DEGREE: usize, const DET_LEN: usize, const ADJ_LEN: usize> {
    determinant: DeterminantExpansion1Parameter<SIZE, DEGREE, DET_LEN>,
    adjugate: AdjugateExpansion1Parameter<SIZE, DEGREE, ADJ_LEN>,
}

impl DeterminantAndAdjugateExpansions1Parameter<2, 1, 3, 2> {

    /// M(μ) = A + Bμ
    fn new_from_matrix(a: &Matrix2<f64>, b: &Matrix2<f64>) -> Self {

        let c0_adj = adjugate2x2(a);
        let c1_adj = adjugate2x2(b);

        let c0_det: f64 = a.determinant();
        let c1_det: f64 = a.transpose().component_mul(&c1_adj).sum();
        let c2_det: f64 = b.determinant();

        DeterminantAndAdjugateExpansions1Parameter {
            determinant: PolynomialCoefficientsFixedLength([c0_det, c1_det, c2_det]),
            adjugate: PolynomialCoefficientsFixedLength([c0_adj, c1_adj]),
        }
    }
}

impl DeterminantAndAdjugateExpansions1Parameter<3, 1, 4, 3> {

    /// M(μ) = A + Bμ
    fn new_from_matrix(a: &Matrix3<f64>, b: &Matrix3<f64>) -> Self {

        // Common computations
        let i= Matrix3::identity();
        let tr_a: f64 = a.trace();
        let tr_b: f64 = b.trace();

        // Matrix products
        let a_sq: Matrix3<f64> = a * a;
        let b_sq: Matrix3<f64> = b * b;
        let ab: Matrix3<f64> = a * b;

        // Intermediate terms (reused for both det and adj)
        let tr_ab: f64 = ab.trace();
        let term_a: Matrix3<f64> = &a_sq - a * tr_a;
        let term_b: Matrix3<f64> = &b_sq - b * tr_b;
        let tr_term_a: f64 = 0.5 * term_a.trace(); // = (trA_sq - trA**2)/2
        let tr_term_b: f64 = 0.5 * term_b.trace(); // = (trB_sq - trB**2)/2

        // Determinant coefficients
        let c0_det: f64 = a.determinant();
        let c1_det: f64 = (a_sq * b).trace() - tr_term_a * tr_b - tr_a * tr_ab;
        let c2_det: f64 = (b_sq * a).trace() - tr_term_b * tr_a - tr_b * tr_ab;
        let c3_det: f64 = b.determinant();

        // Adjugate coefficients
        let c0_adj: Matrix3<f64> = &term_a - &i * tr_term_a;
        let c1_adj: Matrix3<f64> = &i * (tr_a * tr_b - tr_ab) - (a * tr_b + b * tr_a) + ab + b * a;
        let c2_adj: Matrix3<f64> = &term_b - &i * tr_term_b;

        DeterminantAndAdjugateExpansions1Parameter {
            determinant: PolynomialCoefficientsFixedLength([c0_det, c1_det, c2_det, c3_det]),
            adjugate: PolynomialCoefficientsFixedLength([c0_adj, c1_adj, c2_adj]),
        }
    }
}

pub struct PowerSeriesCoefficientsVec<T> (Vec<T>); // growable length

struct InverseDeterminant3x3;

impl InverseDeterminant3x3 {

    pub fn polynomial_coefficient(
        invdet0: f64,
        h1: f64,
        h2: f64,
        h3: f64,
        order: u8,
        factor_cache: &combinatorics::FactorialCache
    ) -> Option<f64> {
        let terms = combinatorics::expansion_terms(order);
        let mut sum = 0.0f64;
        
        for term in terms {
            let coeff = combinatorics::signed_multinomial_coefficient(term, factor_cache)? as f64;
            let combinatorics::ExpansionTerm(a, b, c) = term; // extract a, b, c from term
            
            sum += coeff * h1.powi(a as i32) * h2.powi(b as i32) * h3.powi(c as i32);
        }
        
        Some(sum * invdet0)
    }

    /// Computes coefficients of power series up to specified maximum order (vec version)
    pub fn power_series_coefficients_vec(
        determinant_expansion: &DeterminantExpansion1Parameter<3,1,4>, // Assuming 3x3 matrix with degree 1
        maximum_order: u8
    ) -> Option<PowerSeriesCoefficientsVec<f64>> {
        if determinant_expansion.0[0] == 0.0 {
            return None; // Matrix is singular
        }
        
        // Use of recip() instead of powi(-1) for better numerical properties
        let invdet0 = determinant_expansion.0[0].recip();
        let h1 = determinant_expansion.0[1] * invdet0;
        let h2 = determinant_expansion.0[2] * invdet0;
        let h3 = determinant_expansion.0[3] * invdet0;
        
        // Precompute factorials up to 3 * maximum_order for efficiency
        let factor_cache = combinatorics::FactorialCache::new(maximum_order as usize);
        
        let mut coefficients: Vec<f64> = Vec::with_capacity((maximum_order + 1) as usize);
        coefficients.push(invdet0); // Order 0 coefficient is always invdet0
        
        for order in 1..=maximum_order {
            coefficients.push(
                Self::polynomial_coefficient(invdet0, h1, h2, h3, order, &factor_cache)?
            );
        }
        
        Some(PowerSeriesCoefficientsVec(coefficients))
    }

    /// Computes polynomial coefficients if the length is known at compile time
    pub fn polynomial_coefficients_fixed_length<const LEN: usize>(
        determinant_expansion: &DeterminantExpansion1Parameter<3,1,4> // Assuming 3x3 matrix with degree 1
    ) -> Option<PolynomialCoefficientsFixedLength<f64, LEN>> {
        if LEN == 0 {
            return None;
        }

        if determinant_expansion.0[0] == 0.0 {
            return None; // Matrix is singular
        }
        
        // Use of recip() instead of powi(-1) for better numerical properties
        let invdet0 = determinant_expansion.0[0].recip();
        if LEN == 1 {
            return Some(PolynomialCoefficientsFixedLength([invdet0; LEN]));
        }

        let h1 = determinant_expansion.0[1] * invdet0;
        let h2 = determinant_expansion.0[2] * invdet0;
        let h3 = determinant_expansion.0[3] * invdet0;

        let maximum_order: usize = LEN - 1;
        
        // Precompute factorials up to 3 * maximum_order for efficiency
        let factor_cache = combinatorics::FactorialCache::new(maximum_order);
        
        let mut coefficients: [f64; LEN] = [0.0; LEN]; // Order 0 coefficient is always invdet0
        coefficients[0] = invdet0;
        
        for order in 1..=maximum_order {
            coefficients[order] = Self::polynomial_coefficient(invdet0, h1, h2, h3, order as u8, &factor_cache)?
        }
        
        Some(PolynomialCoefficientsFixedLength(coefficients))
    }
}

/// Combinatorial calculator for multinomial coefficients
mod combinatorics {
    
    /// Precomputed factorials up to a certain limit
    #[derive(Debug)]
    pub struct FactorialCache {
        values: Vec<u64>,
        max_n: usize,
    }
    
    // Memoization:
    // Factorials are precomputed once and reused
    // Expansion terms are generated lazily
    impl FactorialCache {
        /// Creates a new cache with factorials up to max_n
        pub fn new(max_n: usize) -> Self {
            let mut values = vec![1u64; max_n + 1];
            for i in 1..=max_n {
                values[i] = values[i - 1].wrapping_mul(i as u64);
            }
            Self { values, max_n }
        }
        
        /// Gets factorial(n) from cache
        pub fn get(&self, n: usize) -> Option<u64> {
            if n <= self.max_n {
                Some(self.values[n])
            } else {
                None
            }
        }
    }
    
    /// Represents a term in the expansion (a, b, c exponents)
    #[derive(Debug, Copy, Clone)]
    pub struct ExpansionTerm(pub u8, pub u8, pub u8);
    
    /// Calculates multinomial coefficient with sign based on parity
    pub fn signed_multinomial_coefficient(
        term: ExpansionTerm,
        factorials: &FactorialCache
    ) -> Option<i64> {
        let ExpansionTerm(a, b, c) = term;
        let total: u8 = a + b + c;
        
        let a_fact: u64 = factorials.get(a as usize)?;
        let b_fact: u64 = factorials.get(b as usize)?;
        let c_fact: u64 = factorials.get(c as usize)?;
        let total_fact: u64 = factorials.get(total as usize)?;
        
        let denominator: u64 = a_fact.checked_mul(b_fact)?.checked_mul(c_fact)?;
        let raw_coeff: i64 = total_fact.checked_div(denominator)? as i64;
        
        Some(if total & 1 == 1 { -raw_coeff } else { raw_coeff })
    }
    
    /// Finds all triples (a, b, c) such that a + 2b + 3c = order
    pub fn expansion_terms(order: u8) -> impl Iterator<Item = ExpansionTerm> {
        (0..=order / 3).flat_map(move |c: u8| {
            let remaining_after_c: u8 = order - 3 * c;
            (0..=remaining_after_c / 2).map(move |b: u8| {
                let a: u8 = order - 2 * b - 3 * c;
                ExpansionTerm(a, b, c)
            })
        })
    }
}

impl<const LEN: usize> PolynomialCoefficientsFixedLength<f64, LEN> {

    /// Helper function for polynomial multiplication
    pub fn mul_polynomial(&self, b: &Self) -> Self {
        let mut result = [0.0; LEN];
        
        // Iterate through all possible combinations where i + j < LEN
        for (i, &coeff_i) in self.0.iter().enumerate().take(LEN) {
            let max_j = LEN.saturating_sub(i);
            for (j, &coeff_j) in b.0.iter().enumerate().take(max_j) {
                result[i + j] += coeff_i * coeff_j;
            }
        }
        
        PolynomialCoefficientsFixedLength(result)
    }

    pub fn pow(&self, power: u8) -> Self {
        // Initialize output array with zeros
        let mut result: PolynomialCoefficientsFixedLength<f64, LEN> = PolynomialCoefficientsFixedLength([0.0; LEN]);
        
        // Handle the 0th power case (always 1)
        if power == 0 {
            result.0[0] = 1.0 as f64;
            return result;
        }

        result.0[..LEN].copy_from_slice(&self.0);

        if power == 1 {
            return result;
        }
        
        // Use exponentiation by squaring for better performance
        let mut current_power = 1;
        let mut current_poly = result.clone();

        while current_power * 2 <= power {
            current_poly = current_poly.mul_polynomial(&current_poly);
            current_power *= 2;
        }

        // Multiply by remaining powers if needed
        for _ in current_power..power {
            current_poly = current_poly.mul_polynomial(self);
        }

        current_poly
    }
}

#[cfg(not(test))]
mod tests {

    #[test]
    fn test_polynomial_pow() {
        
        let coeffs: PolynomialCoefficientsFixedLength<f64, 4> = PolynomialCoefficientsFixedLength([1.0, 2.0, 0.0, 0.0]);
        println!("{:?}", coeffs.0);
        
        // Test power of 0 (should return 1)
        let result: PolynomialCoefficientsFixedLength<f64, 4> = coeffs.pow(0);
        println!("{:?}", result.0);
        assert_eq!(result.0, [1.0, 0.0, 0.0, 0.0]);
        
        // Test power of 1 (should return original)
        let result: PolynomialCoefficientsFixedLength<f64, 4> = coeffs.pow(1);
        println!("{:?}", result.0);
        assert_eq!(result.0, [1.0, 2.0, 0.0, 0.0]);

        // Test power of 2 // Test case: (1 + 2x)^2 = 1 + 2x + 4x^2
        let result: PolynomialCoefficientsFixedLength<f64, 4> = coeffs.pow(2);
        println!("{:?}", result.0);
        assert_eq!(result.0, [1.0, 4.0, 4.0, 0.0]);

        // Test power of 3 // Test case: (1 + 2x)^3 = 1 + 6x + 12x^2 + 8x^3
        let result: PolynomialCoefficientsFixedLength<f64, 4> = coeffs.pow(3);
        println!("{:?}", result.0);
        assert_eq!(result.0, [1.0, 6.0, 12.0, 8.0]);

        // Test power of 4 // Test case: (1 + 2x)^4 = 1 + 8x + 24x^2 + 32x^3 |+ 16x^4
        let result: PolynomialCoefficientsFixedLength<f64, 4> = coeffs.pow(4);
        println!("{:?}", result.0);
        assert_eq!(result.0, [1.0, 8.0, 24.0, 32.0]);
    }
}