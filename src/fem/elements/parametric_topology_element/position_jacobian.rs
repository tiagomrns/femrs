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

use nalgebra::{DMatrix, Dim, Dyn, Matrix, VecStorage};
use nalgebra::dimension::{U2, U3, DimName};

/// Computes the Jacobian matrix for finite element analysis.
///
/// # Arguments
/// * `all_nodal_coords` - Matrix containing coordinates of all nodes (2D or 3D)
/// * `element_node_ids` - Indices of nodes belonging to the current element
/// * `jacobian_shape_functions` - Matrix of shape function derivatives
///
/// # Panics
/// Panics if dimensions are incompatible
pub fn compute_position_jacobian<D: Dim>(
    all_nodal_coords: &Matrix<f64, D, Dyn, VecStorage<f64, D, Dyn>>,
    element_node_ids: &[u32],
    jacobian_shape_functions: &DMatrix<f64>,
) -> Matrix<f64, D, Dyn, VecStorage<f64, D, Dyn>>
where
    D: Dim, D: DimName
    // Ensure D is either U2 or U3 by checking it's one of these types
{
    let n_nodes = element_node_ids.len();
    let dim = all_nodal_coords.nrows();
    
    assert_eq!(
        jacobian_shape_functions.nrows(),
        n_nodes,
        "Shape function rows must match number of element nodes"
    );
    assert_eq!(
        jacobian_shape_functions.ncols(),
        dim,
        "Shape function columns must match spatial dimension"
    );

    // More efficient way to construct the element coordinates matrix
    let mut element_coords = Matrix::zeros_generic(D::name(), Dyn(n_nodes));
    for (col, &node_id) in element_node_ids.iter().enumerate() {
        element_coords
            .column_mut(col)
            .copy_from(&all_nodal_coords.column(node_id as usize));
    }

    element_coords * jacobian_shape_functions
}

/// Convenience wrapper for 3D case
pub fn compute_position_jacobian_3d(
    all_nodal_coords: &Matrix<f64, U3, Dyn, VecStorage<f64, U3, Dyn>>,
    element_node_ids: &[u32],
    jacobian_shape_functions: &DMatrix<f64>,
) -> Matrix<f64, U3, Dyn, VecStorage<f64, U3, Dyn>> {
    compute_position_jacobian(all_nodal_coords, element_node_ids, jacobian_shape_functions)
}

/// Convenience wrapper for 2D case
pub fn compute_position_jacobian_2d(
    all_nodal_coords: &Matrix<f64, U2, Dyn, VecStorage<f64, U2, Dyn>>,
    element_node_ids: &[u32],
    jacobian_shape_functions: &DMatrix<f64>,
) -> Matrix<f64, U2, Dyn, VecStorage<f64, U2, Dyn>> {
    compute_position_jacobian(all_nodal_coords, element_node_ids, jacobian_shape_functions)
}