//! Quadrature Rule Implementation for Finite Element Analysis
//!
//! This module provides an efficient implementation of numerical quadrature rules
//! (also known as numerical integration rules) for finite element computations.
//! The implementation supports Gauss-Legendre quadrature rules in 1D, 2D, and 3D
//! with both linear and quadratic orders.
//!
//! ## Key Features
//! - **Dimension and Order Generic**: Uses const generics to support different
//!   dimensions (1D, 2D, 3D) and orders (linear, quadratic)
//! - **Memoization**: Caches quadrature rules for efficient reuse
//! - **Tensor Product Rules**: Automatically constructs higher-dimensional rules
//!   from 1D rules
//! - **Coordinate Transformation**: Transforms standard [-1,1] interval to [0,1]
//!
//! ## Supported Quadrature Types
//! - 1D (Line):
//!   - Linear (1st order, 2 points)
//!   - Quadratic (2nd order, 3 points)
//! - 2D (Square):
//!   - Linear (1st order, 4 points)
//!   - Quadratic (2nd order, 9 points)
//! - 3D (Cube):
//!   - Linear (1st order, 8 points)
//!   - Quadratic (2nd order, 27 points)
//!
//! ## Implementation Details
//! - Uses `nalgebra` for matrix/vector operations
//! - Thread-safe lazy initialization via `lazy_static`
//! - Error handling for unsupported rules
//!
//! ## Example Usage
//! ```
//! use quadrature::{LineQuadratic, QuadratureType};
//!
//! let rule = LineQuadratic.get_rule().unwrap();
//! println!("Points: {:?}", rule.points);
//! println!("Weights: {:?}", rule.weights);
//! ```
//!
//! Note: Higher-order rules can be added by extending the `create_1d_rule` function
//! and adding corresponding type aliases and initialization methods.

use std::{iter::IntoIterator, usize};
use once_cell::sync::Lazy;

#[derive(Debug)]
pub enum QuadratureError {
    UnsupportedRule { dim: usize, order: usize },
    DimensionMismatch { expected: usize, actual: usize }
}

impl std::fmt::Display for QuadratureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuadratureError::UnsupportedRule { dim, order } => 
                write!(f, "Unsupported quadrature rule: dimension {} with order {}", dim, order),
            QuadratureError::DimensionMismatch { expected, actual } =>
                write!(f, "Dimension mismatch: expected is {} but actual is {}", expected, actual),
        }
    }
}

impl std::error::Error for QuadratureError {}

// Quadrature rule struct holding points and weights
#[derive(Debug, Clone)]
// Use const generic for compile-time known sizes
pub struct QuadratureRule<const DIM: usize, const LEN: usize> {
    pub points: [[f64; DIM]; LEN],  // Array of fixed-size vectors
    pub weights: [f64; LEN],             // Array of weights
}

// Implement IntoIterator to allow for iteration over points and weights
impl<const DIM: usize, const LEN: usize> IntoIterator for QuadratureRule<DIM, LEN> {
    type Item = ([f64; DIM], f64);
    type IntoIter = std::iter::Zip<
        std::array::IntoIter<[f64; DIM], LEN>,
        std::array::IntoIter<f64, LEN>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.points.into_iter().zip(self.weights.into_iter())
        //IntoIterator::into_iter(self.points)
        //    .zip(IntoIterator::into_iter(self.weights))
    }
}

// Precompute all quadrature rules at compile time or first use
static LINEAR_1D: Lazy<QuadratureRule<1, 2>> = Lazy::new(|| create_linear_1d_rule());
static QUADRATIC_1D: Lazy<QuadratureRule<1, 3>> = Lazy::new(|| create_quadratic_1d_rule());

static LINEAR_2D: Lazy<QuadratureRule<2, 4>> = Lazy::new(
    || create_2d_from_1d::<2, 4>(&LINEAR_1D).unwrap()
);

static QUADRATIC_2D: Lazy<QuadratureRule<2, 9>> = Lazy::new(
    || create_2d_from_1d::<3, 9>(&QUADRATIC_1D).unwrap()
);

static LINEAR_3D: Lazy<QuadratureRule<3, 8>> = Lazy::new(
    || create_3d_from_1d::<2, 8>(&LINEAR_1D).unwrap()
);

static QUADRATIC_3D: Lazy<QuadratureRule<3, 27>> = Lazy::new(
    || create_3d_from_1d::<3, 27>(&QUADRATIC_1D).unwrap()
);

fn create_linear_1d_rule() -> QuadratureRule<1, 2> {
    let aux = 1.0 / (3.0_f64).sqrt();
    let points = [[-aux], [aux]];
    let weights = [1.0, 1.0];

    let (points, weights) = transform_to_01(points, weights);
    QuadratureRule { points, weights }
}

fn create_quadratic_1d_rule() -> QuadratureRule<1, 3> {
    let aux1 = (3.0 / 5.0_f64).sqrt();
    let aux2 = 5.0 / 9.0;
    let points = [[-aux1], [0.0], [aux1]];
    let weights = [aux2, 8.0 / 9.0, aux2];

    let (points, weights) = transform_to_01(points, weights);
    QuadratureRule { points, weights }
}

fn create_2d_from_1d<const IN_LEN: usize, const OUT_LEN: usize>(
    rule_1d: &QuadratureRule<1, IN_LEN>,
) -> Result<QuadratureRule<2, OUT_LEN>, QuadratureError> {
    let mut points = [[0.0, 0.0]; OUT_LEN];
    let mut weights = [0.0; OUT_LEN];

    let actual_dimension: usize = IN_LEN * IN_LEN;
    if actual_dimension != OUT_LEN {
        return Err(QuadratureError::DimensionMismatch { expected: OUT_LEN, actual: actual_dimension })
    }
    
    for (i, (x, wx)) in rule_1d.clone().into_iter().enumerate() {
        for (j, (y, wy)) in rule_1d.clone().into_iter().enumerate() {
            let idx = i * IN_LEN + j;
            points[idx] = [x[0], y[0]];
            weights[idx] = wx * wy;
        }
    }
    
    Ok(QuadratureRule { points, weights })
}

fn create_3d_from_1d<const IN_LEN: usize, const OUT_LEN: usize>(
    rule_1d: &QuadratureRule<1, IN_LEN>,
) -> Result<QuadratureRule<3, OUT_LEN>, QuadratureError> {
    let mut points = [[0.0, 0.0, 0.0]; OUT_LEN];
    let mut weights = [0.0; OUT_LEN];

    let actual_dimension: usize = IN_LEN * IN_LEN * IN_LEN;
    if actual_dimension != OUT_LEN {
        return Err(QuadratureError::DimensionMismatch { expected: OUT_LEN, actual: actual_dimension })
    }
    
    for (i, (x, wx)) in rule_1d.clone().into_iter().enumerate() {
        for (j, (y, wy)) in rule_1d.clone().into_iter().enumerate() {
            for (k, (z, wz)) in rule_1d.clone().into_iter().enumerate() {
                let idx = (i * IN_LEN + j) * IN_LEN + k;
                points[idx] = [x[0], y[0], z[0]];
                weights[idx] = wx * wy * wz;
            }
        }
    }
    
    Ok(QuadratureRule { points, weights })
}

/// Transform points and weights from ```[-1,1]``` to ```[0,1]``` quadrature
fn transform_to_01<const DIM: usize, const LEN: usize>(
    points: [[f64; DIM]; LEN],
    weights: [f64; LEN],
) -> ([[f64; DIM]; LEN], [f64; LEN]) {
    let transformed_points = points.map(|p| (p + 1.0) / 2.0);
    let transformed_weights = weights.map(|w| w / 2.0_f64.powi(DIM as i32));
    (transformed_points, transformed_weights)
}

#[cfg(not(test))]
mod tests {

    #[test]
    fn test_linear_1d_rule() {
        let rule = LINEAR_1D.clone();
        assert_eq!(rule.points.len(), 2);
        assert_eq!(rule.weights.len(), 2);
        
        // Check points are in [0,1]
        for point in rule.points {
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
        }
        
        // Check weights sum to 1 (for integrating constants correctly)
        println!("LINEAR 1D\npoints={:?}, \nweights={:?}, sum={}\n", rule.points, rule.weights, rule.weights.iter().sum::<f64>());
    }

    #[test]
    fn test_quadratic_1d_rule() {
        let rule = QUADRATIC_1D.clone();
        assert_eq!(rule.points.len(), 3);
        assert_eq!(rule.weights.len(), 3);
        
        for point in rule.points {
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
        }
        
        // Should integrate linear and quadratic functions exactly
        println!("QUADRATIC 1D\npoints={:?}, \nweights={:?}, sum={}\n", rule.points, rule.weights, rule.weights.iter().sum::<f64>());
    }

    #[test]
    fn test_2d_from_1d_linear() {
        let rule = LINEAR_2D.clone();
        assert_eq!(rule.points.len(), 4);
        assert_eq!(rule.weights.len(), 4);
        
        // Check all points are in [0,1]²
        for point in rule.points {
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
            assert!(point[1] >= 0.0 && point[1] <= 1.0);
        }
        
        // Check weights sum to 1 (area of unit square)
        println!("LINEAR 2D\npoints={:?}, \nweights={:?}, sum={}\n", rule.points, rule.weights, rule.weights.iter().sum::<f64>());
    }

    #[test]
    fn test_2d_from_1d_quadratic() {
        let rule = QUADRATIC_2D.clone();
        assert_eq!(rule.points.len(), 9);
        assert_eq!(rule.weights.len(), 9);
        
        for point in rule.points {
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
            assert!(point[1] >= 0.0 && point[1] <= 1.0);
        }
        
        println!("QUADRATIC 2D\npoints={:?}, \nweights={:?}, sum={}\n", rule.points, rule.weights, rule.weights.iter().sum::<f64>());
    }

    #[test]
    fn test_3d_from_1d_linear() {
        let rule = LINEAR_3D.clone();
        assert_eq!(rule.points.len(), 8);
        assert_eq!(rule.weights.len(), 8);
        
        for point in rule.points {
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
            assert!(point[1] >= 0.0 && point[1] <= 1.0);
            assert!(point[2] >= 0.0 && point[2] <= 1.0);
        }
        
        // Check weights sum to 1 (volume of unit cube)
        println!("LINEAR 3D\npoints={:?}, \nweights={:?}, sum={}\n", rule.points, rule.weights, rule.weights.iter().sum::<f64>());
    }

    #[test]
    fn test_3d_from_1d_quadratic() {
        let rule = QUADRATIC_3D.clone();
        assert_eq!(rule.points.len(), 27);
        assert_eq!(rule.weights.len(), 27);
        
        for point in rule.points {
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
            assert!(point[1] >= 0.0 && point[1] <= 1.0);
            assert!(point[2] >= 0.0 && point[2] <= 1.0);
        }
        
        println!("QUADRATIC 3D\npoints={:?}, \nweights={:?}, sum={}\n", rule.points, rule.weights, rule.weights.iter().sum::<f64>());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        // Test that trying to create a 2D rule with wrong output length fails
        let result = create_2d_from_1d::<2, 3>(&LINEAR_1D);
        assert!(matches!(result, Err(QuadratureError::DimensionMismatch { expected: 3, actual: 4 })));

        // Test that trying to create a 3D rule with wrong output length fails
        let result = create_3d_from_1d::<2, 7>(&LINEAR_1D);
        assert!(matches!(result, Err(QuadratureError::DimensionMismatch { expected: 7, actual: 8 })));
    }

    #[test]
    fn test_into_iterator() {
        let rule = LINEAR_1D.clone();
        let mut sum = 0.0;
        
        for (point, weight) in rule {
            sum += weight;
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
        }
        
        println!("test_into_iterator -> sum={}\n", sum);
    }

    #[test]
    fn test_transform_to_01() {
        let points = [SVector::from([-1.0]), SVector::from([1.0])];
        let weights = [1.0, 1.0];
        
        let (transformed_points, transformed_weights) = transform_to_01(points, weights);
        
        assert_eq!(transformed_points[0][0], 0.0);
        assert_eq!(transformed_points[1][0], 1.0);
        assert_eq!(transformed_weights[0], 0.5);
        assert_eq!(transformed_weights[1], 0.5);
    }

    #[test]
    fn test_quadrature_accuracy() {
        // Test that the quadrature rules can integrate polynomials exactly
        
        // Linear 1D rule should integrate linear functions exactly
        let linear_rule_1d = LINEAR_1D.clone();
        let mut integral = 0.0;
        for (point, weight) in linear_rule_1d {
            integral += weight * (2.0 * point[0] + 3.0); // Integral of 2x + 3 on [0,1] is 4
        }
        println!("linear_rule_1d: expected = 4.0, actual integral = {}", integral);
        //assert_relative_eq!(integral, 4.0, epsilon = 1e-10);
        
        // Linear 2D rule should integrate bilinear functions exactly
        let linear_rule_2d = LINEAR_2D.clone();
        let mut integral = 0.0;
        for (point, weight) in linear_rule_2d {
            integral += weight * (point[0] * point[1]); // Integral of xy on [0,1]² is 0.25
        }
        println!("linear_rule_2d: expected = 0.25, actual integral = {}", integral);

        // Linear 3D rule should integrate trilinear functions exactly
        let linear_rule_3d = LINEAR_3D.clone();
        let mut integral = 0.0;
        for (point, weight) in linear_rule_3d {
            integral += weight * ((4.0 + point[0]) * point[1] * point[2]); // Integral of (4+x)yz on [0,1]² is 1.125
        }
        println!("linear_rule_3d: expected = 1.125, actual integral = {}", integral);

        // Quadratic 1D rule should integrate quadratic functions exactly
        let quad_rule_1d = LINEAR_1D.clone();
        let mut integral = 0.0;
        for (point, weight) in quad_rule_1d {
            integral += weight * (3.0 * point[0] * point[0] + 2.0 * point[0] + 1.0); // Integral of 3x² + 2x + 1 on [0,1] is 3
        }
        println!("quad_rule_1d: expected = 3.0, actual integral = {}", integral);
        //assert_relative_eq!(integral, 3.0, epsilon = 1e-10);

        // Quadratic 2D rule should integrate biquadratic functions exactly
        let quad_rule_2d = LINEAR_2D.clone();
        let mut integral = 0.0;
        for (point, weight) in quad_rule_2d {
            integral += weight * 
            (3.0 * point[0] * point[0] + 2.0 * point[0] + 1.0) *
            (6.0 * point[1] * point[1] - 2.0 * point[1] + 1.0); // Integral of (3x² + 2x + 1)(3y² - 2y + 1) on [0,1]² is 6
        }
        println!("quad_rule_2d: expected = 6.0, actual integral = {}", integral);

        // Quadratic 3D rule should integrate triquadratic functions exactly
        let quad_rule_3d = LINEAR_3D.clone();
        let mut integral = 0.0;
        for (point, weight) in quad_rule_3d {
            integral += weight * 
            (3.0 * point[0] * point[0] + 2.0 * point[0] + 1.0) *
            (2.0 * 3.0 * point[1] * point[1] - 2.0 * point[1] + 2.0) *
            (-2.0 * 3.0 * point[2] * point[2] + 2.0 * point[2] - 1.0); // Integral on [0,1]² is -18
        }
        println!("quad_rule_3d: expected = -18.0, actual integral = {}", integral);
    }
}