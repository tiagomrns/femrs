//! # Jacobian Computation for Finite Element Analysis
//!
//! This module provides functionality for computing Jacobian matrices in finite element analysis.
//! The Jacobian matrix is crucial for mapping between local element coordinates and global coordinates.
//!
//! ### Overview
//! The main function `compute_position_jacobian` calculates the Jacobian matrix given:
//! - Nodal coordinates for the entire system
//! - Node indices for the current element
//! - Derivatives of shape functions
//!
//! Convenience wrappers `compute_position_jacobian_2d` and `compute_position_jacobian_3d` are provided
//! for common 2D and 3D cases respectively.
//!
//! ### Theory
//! The Jacobian matrix J is computed as:
//! J = ∑ (x_i ⊗ ∇N_i)
//! where:
//! - x_i are the nodal coordinates
//! - ∇N_i are the shape function derivatives
//!
//! ### Examples
//! ```
//! use nalgebra::{DMatrix, Matrix3xX};
//! 
//! // 3D example
//! let all_coords = Matrix3xX::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
//! let element_nodes = vec![0, 1, 2];
//! let shape_derivs = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
//! 
//! let jacobian = compute_position_jacobian_3d(&all_coords, &element_nodes, &shape_derivs);
//! ```
//!
//! ### Panics
//! The functions will panic if:
//! - The shape function matrix dimensions don't match the number of element nodes
//! - The shape function column count doesn't match the spatial dimension
//!
//! ### Performance
//! The implementation uses nalgebra's efficient matrix operations and avoids unnecessary allocations.
//! For repeated computations, consider reusing matrix allocations where possible.



/// Computes the Jacobian matrix for finite element analysis.
///
/// # Arguments
/// * `all_nodal_coords` - Matrix containing coordinates of all nodes (2D or 3D)
/// * `element_node_ids` - Indices of nodes belonging to the current element
/// * `jacobian_shape_functions` - Matrix of shape function derivatives
///
/// # Panics
/// Panics if dimensions are incompatible

use ndarray::Array2;

pub fn compute_position_jacobian(
    all_nodal_coords: &Array2<f64>,
    element_node_ids: &[u32],
    jacobian_shape_functions: &Array2<f64>,
) -> Array2<f64> {
    let dim = all_nodal_coords.shape()[0];
    let n_nodes = element_node_ids.len();

    assert_eq!(
        jacobian_shape_functions.shape()[1],
        dim,
        "Shape function columns must match spatial dimension"
    );

    // Build element coordinates matrix by selecting columns from all_nodal_coords
    let mut element_coords = Array2::zeros((dim, n_nodes));
    for (col, &node_id) in element_node_ids.iter().enumerate() {
        let node_col = all_nodal_coords.column(node_id as usize);
        element_coords.column_mut(col).assign(&node_col);
    }

    // Matrix multiplication: element_coords (dim, n_nodes) × jacobian_shape_functions (n_nodes, dim)
    element_coords.dot(jacobian_shape_functions)
}

/// Convenience wrapper for 3D case
pub fn compute_position_jacobian_3d(
    all_nodal_coords: &Array2<f64>,
    element_node_ids: &[u32],
    jacobian_shape_functions: &Array2<f64>,
) -> Array2<f64> {
    assert_eq!(all_nodal_coords.shape()[0], 3, "all_nodal_coords must be 3D");
    compute_position_jacobian(all_nodal_coords, element_node_ids, jacobian_shape_functions)
}

/// Convenience wrapper for 2D case
pub fn compute_position_jacobian_2d(
    all_nodal_coords: &Array2<f64>,
    element_node_ids: &[u32],
    jacobian_shape_functions: &Array2<f64>,
) -> Array2<f64> {
    assert_eq!(all_nodal_coords.shape()[0], 2, "all_nodal_coords must be 2D");
    compute_position_jacobian(all_nodal_coords, element_node_ids, jacobian_shape_functions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_compute_position_jacobian_2d() {
        // Create a simple 2D mesh with 4 nodes
        let all_nodal_coords = array![
            [0.0, 1.0, 1.0, 0.0], // x coordinates
            [0.0, 0.0, 1.0, 1.0], // y coordinates
        ];

        // Select nodes for a quadrilateral element
        let element_node_ids = [0, 1, 2, 3];
        
        // Shape function derivatives for a bilinear element at center (ξ=0, η=0)
        let jacobian_shape_functions = array![
            [-0.25, -0.25],
            [0.25, -0.25],
            [0.25, 0.25],
            [-0.25, 0.25],
        ];

        let result = compute_position_jacobian_2d(
            &all_nodal_coords,
            &element_node_ids,
            &jacobian_shape_functions,
        );

        // Expected Jacobian for a unit square at center
        let expected = array![
            [0.5, 0.0],
            [0.0, 0.5],
        ];

        assert_eq!(result.shape(), expected.shape());
        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert!((result[[i, j]] - expected[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_compute_position_jacobian_3d() {
        // Create a simple 3D mesh with 8 nodes
        let all_nodal_coords = array![
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], // x coordinates
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], // y coordinates
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], // z coordinates
        ];

        // Select nodes for a hexahedral element
        let element_node_ids = [0, 1, 2, 3, 4, 5, 6, 7];
        
        // Shape function derivatives for a trilinear element at center (ξ=0, η=0, ζ=0)
        let jacobian_shape_functions = array![
            [-0.125, -0.125, -0.125],
            [0.125, -0.125, -0.125],
            [0.125, 0.125, -0.125],
            [-0.125, 0.125, -0.125],
            [-0.125, -0.125, 0.125],
            [0.125, -0.125, 0.125],
            [0.125, 0.125, 0.125],
            [-0.125, 0.125, 0.125],
        ];

        let result = compute_position_jacobian_3d(
            &all_nodal_coords,
            &element_node_ids,
            &jacobian_shape_functions,
        );

        // Expected Jacobian for a unit cube at center
        let expected = array![
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ];

        assert_eq!(result.shape(), expected.shape());
        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert!((result[[i, j]] - expected[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_compute_position_jacobian_general() {
        // Test the general function with 2D data
        let all_nodal_coords = array![
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
        ];

        let element_node_ids = [0, 1, 2];
        
        let jacobian_shape_functions = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ];

        let result = compute_position_jacobian(
            &all_nodal_coords,
            &element_node_ids,
            &jacobian_shape_functions,
        );

        // Expected result: element_coords * jacobian_shape_functions
        // element_coords = [[0.0, 1.0, 0.5],
        //                   [0.0, 0.0, 1.0]]
        let expected = array![
            [0.0*1.0 + 1.0*0.0 + 0.5*(-1.0), 0.0*0.0 + 1.0*1.0 + 0.5*(-1.0)],
            [0.0*1.0 + 0.0*0.0 + 1.0*(-1.0), 0.0*0.0 + 0.0*1.0 + 1.0*(-1.0)],
        ];

        assert_eq!(result.shape(), expected.shape());
        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert!((result[[i, j]] - expected[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    #[should_panic(expected = "all_nodal_coords must be 2D")]
    fn test_2d_wrapper_with_3d_data() {
        let all_nodal_coords = array![
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ];

        let element_node_ids = [0, 1, 2, 3];
        let jacobian_shape_functions = array![
            [-0.25, -0.25],
            [0.25, -0.25],
            [0.25, 0.25],
            [-0.25, 0.25],
        ];

        compute_position_jacobian_2d(
            &all_nodal_coords,
            &element_node_ids,
            &jacobian_shape_functions,
        );
    }

    #[test]
    #[should_panic(expected = "Shape function columns must match spatial dimension")]
    fn test_dimension_mismatch() {
        let all_nodal_coords = array![
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ];

        let element_node_ids = [0, 1, 2, 3];
        
        // Wrong shape - 3 columns instead of 2
        let jacobian_shape_functions = array![
            [-0.25, -0.25, 0.0],
            [0.25, -0.25, 0.0],
            [0.25, 0.25, 0.0],
            [-0.25, 0.25, 0.0],
        ];

        compute_position_jacobian(
            &all_nodal_coords,
            &element_node_ids,
            &jacobian_shape_functions,
        );
    }
}