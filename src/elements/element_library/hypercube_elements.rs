//! Finite element shape functions for various element types.
//!
//! This module provides implementations of nodal-based shape functions for:
//! - 1D line elements
//! - 2D quadrilateral elements
//! - 3D hexahedral elements (both full Lagrange and serendipity types)
//!
//! # Traits
//! 
//! ### `NodalBasedShapeFunctions`
//! The main trait defining the interface for shape function implementations:
//! ```ignore
//! pub trait NodalBasedShapeFunctions {
//!     type Coordinates;
//!     const DIMENSION: u8;
//!     const NUMBER_OF_NODES: u8;
//!     fn evaluate_shape_functions(coords: &Self::Coordinates) -> Vec<f64>;
//!     fn evaluate_jacobian_of_shape_functions(coords: &Self::Coordinates) -> DMatrix<f64>;
//! }
//! ```
//!
//! # Implementations
//!
//! ## 1D Line Elements
//! - `LineShapeFunctions<ORDER>`: Lagrange shape functions for 1D line elements
//!   - Supported orders: 1 (linear), 2 (quadratic)
//!
//! ## 2D Quadrilateral Elements
//! - `SquareShapeFunctions<ORDER_X, ORDER_Y>`: Tensor product shape functions for quadrilaterals
//!   - Common type aliases:
//!     - `SquareOrder1ShapeFunctions`: Bilinear quadrilateral (4 nodes)
//!     - `SquareOrder2ShapeFunctions`: Biquadratic quadrilateral (9 nodes)
//!
//! ## 3D Hexahedral Elements
//! - `CubeShapeFunctions<ORDER_X, ORDER_Y, ORDER_Z>`: Tensor product shape functions for hexahedrons
//!   - Common type aliases:
//!     - `CubeOrder1ShapeFunctions`: Trilinear hexahedron (8 nodes)
//!     - `CubeOrder2ShapeFunctions`: Triquadratic hexahedron (27 nodes)
//! - `CubeSerendipityShapeFunctions`: 20-node serendipity element (quadratic with no internal nodes)
//!
//! # Examples
//!
//! ```
//! use ndarray::Array2;
//! use shape_functions::{CubeOrder1ShapeFunctions, NodalBasedShapeFunctions};
//!
//! // Evaluate shape functions at a point
//! let coords = [0.5, 0.5, 0.5];
//! let n = CubeOrder1ShapeFunctions::evaluate_shape_functions(&coords);
//! let jac = CubeOrder1ShapeFunctions::evaluate_jacobian_of_shape_functions(&coords);
//! ```

use ndarray::Array2;

// Trait for shape functions
pub trait NodalBasedShapeFunctions {
    type Coordinates;
    const DIMENSION: u8;
    const NUMBER_OF_NODES: u8;
    fn evaluate_shape_functions(coords: &Self::Coordinates) -> Vec<f64>;
    fn evaluate_jacobian_of_shape_functions(coords: &Self::Coordinates) -> Array2<f64>;
    fn node_ids(&self) -> &[u32] {
        &[]
    }
}

// 1D Line elements
struct LineShapeFunctions<const ORDER: u8>;

impl<const ORDER: u8> NodalBasedShapeFunctions for LineShapeFunctions<ORDER> {
    type Coordinates = f64;
    const DIMENSION: u8 = 1;
    const NUMBER_OF_NODES: u8 = ORDER + 1;

    fn evaluate_shape_functions(x: &f64) -> Vec<f64> {
        match ORDER {
            1 => LineShapeFunctions::<1>::evaluate_shape_functions_impl(x),
            2 => LineShapeFunctions::<2>::evaluate_shape_functions_impl(x),
            _ => panic!("Unsupported order for line shape functions"),
        }
    }
    
    fn evaluate_jacobian_of_shape_functions(x: &f64) -> Array2<f64> {
        match ORDER {
            1 => LineShapeFunctions::<1>::evaluate_jacobian_impl(x),
            2 => LineShapeFunctions::<2>::evaluate_jacobian_impl(x),
            _ => panic!("Unsupported order for line shape functions"),
        }
    }
}

impl LineShapeFunctions<1> {
    /*
    x=0      -> N1(x) = 1-x,     N1'(x) = -1
    x=1      -> N2(x) = x,       N2'(x) = 1
    */
    fn evaluate_shape_functions_impl(x: &f64) -> Vec<f64> {
        vec![1.0 - x, *x]
    }
    
    fn evaluate_jacobian_impl(_x: &f64) -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![-1.0, 1.0]).unwrap()
    }
}

impl LineShapeFunctions<2> {
    /*
    x=0      -> N1(x) = (1-2*x)*(1-x),   N1'(x) = 4*x - 3
    x=0.5    -> N2(x) = 4*x*(1-x),       N2'(x) = 4 - 8*x
    x=1      -> N3(x) = -x*(1-2*x),      N3'(x) = 4*x - 1
    */
    fn evaluate_shape_functions_impl(x: &f64) -> Vec<f64> {
        let a1 = 1.0 - x;
        let a2 = 1.0 - 2.0 * x;
        let x00 = a1 * a2;
        let x05 = 4.0 * x * a1;
        let x10 = -x * a2;
        vec![x00, x05, x10]
    }
    
    fn evaluate_jacobian_impl(x: &f64) -> Array2<f64> {
        let aux: f64 = 4.0 * x;
        Array2::from_shape_vec((1, 3), vec![aux - 3.0, 4.0 - 2.0*aux, aux - 1.0]).unwrap()
    }
}

// 2D Square elements - now with separate X and Y orders
struct SquareShapeFunctions<const ORDER_X: u8, const ORDER_Y: u8>;

impl<const ORDER_X: u8, const ORDER_Y: u8> NodalBasedShapeFunctions for SquareShapeFunctions<ORDER_X, ORDER_Y> {
    type Coordinates = [f64; 2];
    const DIMENSION: u8 = 2;
    const NUMBER_OF_NODES: u8 = (ORDER_X+1) * (ORDER_Y+1);
    
    fn evaluate_shape_functions(coords: &[f64; 2]) -> Vec<f64> {
        let x: f64 = coords[0];
        let y: f64 = coords[1];
        
        let line_functions_x = LineShapeFunctions::<ORDER_X>::evaluate_shape_functions(&x);
        let line_functions_y = LineShapeFunctions::<ORDER_Y>::evaluate_shape_functions(&y);
        
        // Outer product and flatten
        let mut result = vec![0.0; line_functions_x.len() * line_functions_y.len()];
        for i in 0..line_functions_y.len() {
            for j in 0..line_functions_x.len() {
                result[i * line_functions_x.len() + j] = line_functions_y[i] * line_functions_x[j];
            }
        }
        result
    }
    
    fn evaluate_jacobian_of_shape_functions(coords: &[f64; 2]) -> Array2<f64> {
        let x: f64 = coords[0];
        let y: f64 = coords[1];
        
        let line_functions_x = LineShapeFunctions::<ORDER_X>::evaluate_shape_functions(&x);
        let line_jacobian_x = LineShapeFunctions::<ORDER_X>::evaluate_jacobian_of_shape_functions(&x);
        
        let line_functions_y = LineShapeFunctions::<ORDER_Y>::evaluate_shape_functions(&y);
        let line_jacobian_y = LineShapeFunctions::<ORDER_Y>::evaluate_jacobian_of_shape_functions(&y);
        
        let n_nodes = line_functions_x.len() * line_functions_y.len();
        let mut jacobian = Array2::zeros((n_nodes, 2));
        
        // dx component (df/dx)
        for i in 0..line_functions_y.len() {
            for j in 0..line_functions_x.len() {
                let idx = i * line_functions_x.len() + j;
                jacobian[[idx, 0]] = line_functions_y[i] * line_jacobian_x[(0, j)];
            }
        }
        
        // dy component (df/dy)
        for i in 0..line_functions_y.len() {
            for j in 0..line_functions_x.len() {
                let idx = i * line_functions_x.len() + j;
                jacobian[[idx, 1]] = line_jacobian_y[(0, i)] * line_functions_x[j];
            }
        }
        
        jacobian
    }
}

// Type aliases for common cases
type SquareOrder1ShapeFunctions = SquareShapeFunctions<1, 1>;
/*
    Number of nodes of a linear square element (4)

    x   y
    1   0
    0   0
    1   1
    0   1
*/

type SquareOrder2ShapeFunctions = SquareShapeFunctions<2, 2>;
/*
    Number of nodes of a quadratic Lagrange square element (9)

    x   y 
    0   0 
    0.5 0
    1   0

    0   0.5
    0.5 0.5
    1   0.5
        
    0   1
    0.5 1
    1   1
*/

// 3D Cube elements - now with separate X, Y, and Z orders
struct CubeShapeFunctions<const ORDER_X: u8, const ORDER_Y: u8, const ORDER_Z: u8>;

impl<const ORDER_X: u8, const ORDER_Y: u8, const ORDER_Z: u8> NodalBasedShapeFunctions 
for CubeShapeFunctions<ORDER_X, ORDER_Y, ORDER_Z> {
    type Coordinates = [f64; 3];
    const DIMENSION: u8 = 3;
    const NUMBER_OF_NODES: u8 = (ORDER_X+1) * (ORDER_Y+1) * (ORDER_Z+1);
    
    fn evaluate_shape_functions(coords: &[f64; 3]) -> Vec<f64> {
        let x = coords[0];
        let y = coords[1];
        let z = coords[2];
        
        let square_functions = SquareShapeFunctions::<ORDER_X, ORDER_Y>::evaluate_shape_functions(&[x, y]);
        let line_functions_z = LineShapeFunctions::<ORDER_Z>::evaluate_shape_functions(&z);
        
        // Outer product and flatten
        let mut result = vec![0.0; square_functions.len() * line_functions_z.len()];
        for i in 0..line_functions_z.len() {
            for j in 0..square_functions.len() {
                result[i * square_functions.len() + j] = line_functions_z[i] * square_functions[j];
            }
        }
        result
    }
    
    fn evaluate_jacobian_of_shape_functions(coords: &[f64; 3]) -> Array2<f64> {
        let x = coords[0];
        let y = coords[1];
        let z = coords[2];
        
        let square_functions = SquareShapeFunctions::<ORDER_X, ORDER_Y>::evaluate_shape_functions(&[x, y]);
        let square_jacobian = SquareShapeFunctions::<ORDER_X, ORDER_Y>::evaluate_jacobian_of_shape_functions(&[x, y]);
        
        let line_functions_z = LineShapeFunctions::<ORDER_Z>::evaluate_shape_functions(&z);
        let line_jacobian_z = LineShapeFunctions::<ORDER_Z>::evaluate_jacobian_of_shape_functions(&z);
        
        let n_nodes = square_functions.len() * line_functions_z.len();
        let mut jacobian = Array2::zeros((n_nodes, 3));
        
        // dx component (df/dx)
        for i in 0..line_functions_z.len() {
            for j in 0..square_functions.len() {
                let idx = i * square_functions.len() + j;
                jacobian[[idx, 0]] = line_functions_z[i] * square_jacobian[(j, 0)];
            }
        }
        
        // dy component (df/dy)
        for i in 0..line_functions_z.len() {
            for j in 0..square_functions.len() {
                let idx = i * square_functions.len() + j;
                jacobian[[idx, 1]] = line_functions_z[i] * square_jacobian[(j, 1)];
            }
        }
        
        // dz component (df/dz)
        for i in 0..line_functions_z.len() {
            for j in 0..square_functions.len() {
                let idx = i * square_functions.len() + j;
                jacobian[[idx, 2]] = line_jacobian_z[(0, i)] * square_functions[j];
            }
        }
        
        jacobian
    }
}

// Type aliases for common cases
type CubeOrder1ShapeFunctions = CubeShapeFunctions<1, 1, 1>;
/*
    Number of nodes of a linear hexahedral element (8)

    x   y   z
    0   0   0
    1   0   0
    0   1   0
    1   1   0
    0   0   1
    1   0   1
    0   1   1
    1   1   1
*/

type CubeOrder2ShapeFunctions = CubeShapeFunctions<2, 2, 2>;
/*
    Number of nodes of a quadratic Lagrange hexahedral element (27)
    
    x   y   z
    0   0   0
    0.5 0   0
    1   0   0
    0   0.5 0
    0.5 0.5 0
    1   0.5 0
    0   1   0
    0.5 1   0
    1   1   0
    
    0   0   0.5
    0.5 0   0.5
    1   0   0.5
    0   0.5 0.5
    0.5 0.5 0.5
    1   0.5 0.5
    0   1   0.5
    0.5 1   0.5
    1   1   0.5
    
    0   0   1
    0.5 0   1
    1   0   1
    0   0.5 1
    0.5 0.5 1
    1   0.5 1
    0   1   1
    0.5 1   1
    1   1   1
*/

struct CubeSerendipityShapeFunctions;

impl CubeSerendipityShapeFunctions {
    // Computes the shape functions for a quadratic line segment
    // Returns tuple of (N0, N05, N1) corresponding to nodes at 0, 0.5, and 1
    fn line_shape_functions(t: f64) -> (f64, f64, f64) {
        let a1: f64 = 1.0-t;      // t=0 -> 1 ,  t=0.5 -> 0.5 ,  t=1 -> 0
        let a2: f64 = 1.0-2.0*t;  // t=0 -> 1 ,  t=0.5 -> 0 ,    t=1 -> -1
        let n00: f64 = a1*a2;     // t=0 -> 1 ,  t=0.5 -> 0 ,    t=1 -> 0
        let n05: f64 = 4.0*t*a1;  // t=0 -> 0 ,  t=0.5 -> 1 ,    t=1 -> 0
        let n10: f64 = -t*a2;     // t=0 -> 0 ,  t=0.5 -> 0 ,    t=1 -> -1

        (n00, n05, n10)
    }

    // Computes derivatives of the shape functions for a quadratic line segment
    // Returns tuple of (dN0/dt, dN05/dt, dN1/dt)
    fn jacobian_of_line_shape_functions(t: f64) -> (f64, f64, f64) {
        let aux: f64 = 4.0 * t;
        
        (
            aux - 3.0, 
            4.0 - 2.0 * aux, 
            aux - 1.0
        )
    }     
}

impl NodalBasedShapeFunctions for CubeSerendipityShapeFunctions {
    type Coordinates = [f64; 3];
    const DIMENSION: u8 = 3;
    const NUMBER_OF_NODES: u8 = 20;

    fn evaluate_shape_functions(coords: &[f64; 3]) -> Vec<f64> {
        let x: f64 = coords[0];
        let y: f64 = coords[1];
        let z: f64 = coords[2];

        let mut result = vec![0.0; 20];
        
        let (x00, x05, x10) = Self::line_shape_functions(x);
        let (y00, y05, y10) = Self::line_shape_functions(y);
        let (z00, z05, z10) = Self::line_shape_functions(z);

        let y00z00: f64 = y00 * z00;
        let y05z00: f64 = y05 * z00;
        let y10z00: f64 = y10 * z00;

        // Bottom face (z=0)
        result[0] = x00 * y00z00;  // Node 0: (0, 0, 0)
        result[1] = x05 * y00z00;  // Node 1: (0.5, 0, 0)
        result[2] = x10 * y00z00;  // Node 2: (1, 0, 0)
        result[3] = x00 * y05z00;  // Node 3: (0, 0.5, 0)
        result[4] = x10 * y05z00;  // Node 4: (1, 0.5, 0)
        result[5] = x00 * y10z00;  // Node 5: (0, 1, 0)
        result[6] = x05 * y10z00;  // Node 6: (0.5, 1, 0)
        result[7] = x10 * y10z00;  // Node 7: (1, 1, 0)
        
        let y00z05: f64 = y00 * z05;
        let y10z05: f64 = y10 * z05;

        // Mid-edge nodes (z=0.5)
        result[8] = x00 * y00z05;   // Node 8: (0, 0, 0.5)
        result[9] = x10 * y00z05;   // Node 9: (1, 0, 0.5)
        result[10] = x00 * y10z05;  // Node 10: (0, 1, 0.5)
        result[11] = x10 * y10z05;  // Node 11: (1, 1, 0.5)
        
        let y00z10: f64 = y00 * z10;
        let y05z10: f64 = y05 * z10;
        let y10z10: f64 = y10 * z10;

        // Top face (z=1)
        result[12] = x00 * y00z10;  // Node 12: (0, 0, 1)
        result[13] = x05 * y00z10;  // Node 13: (0.5, 0, 1)
        result[14] = x10 * y00z10;  // Node 14: (1, 0, 1)
        result[15] = x00 * y05z10;  // Node 15: (0, 0.5, 1)
        result[16] = x10 * y05z10;  // Node 16: (1, 0.5, 1)
        result[17] = x00 * y10z10;  // Node 17: (0, 1, 1)
        result[18] = x05 * y10z10;  // Node 18: (0.5, 1, 1)
        result[19] = x10 * y10z10;  // Node 19: (1, 1, 1)
        
        result
    }

    fn evaluate_jacobian_of_shape_functions(coords: &[f64; 3]) -> Array2<f64> {
        let x: f64 = coords[0];
        let y: f64 = coords[1];
        let z: f64 = coords[2];

        let mut jacobian = Array2::zeros((20, 3));
    
        let (x00, x05, x10) = Self::line_shape_functions(x);
        let (y00, y05, y10) = Self::line_shape_functions(y);
        let (z00, z05, z10) = Self::line_shape_functions(z);

        let (dx00, dx05, dx10) = Self::jacobian_of_line_shape_functions(x);
        let (dy00, dy05, dy10) = Self::jacobian_of_line_shape_functions(y);
        let (dz00, dz05, dz10) = Self::jacobian_of_line_shape_functions(z);

        let y00z00: f64 = y00 * z00;
        let y05z00: f64 = y05 * z00;
        let y10z00: f64 = y10 * z00;

        let dy00z00: f64 = dy00 * z00;
        let dy05z00: f64 = dy05 * z00;
        let dy10z00: f64 = dy10 * z00;

        let y00dz00: f64 = y00 * dz00;
        let y05dz00: f64 = y05 * dz00;
        let y10dz00: f64 = y10 * dz00;

        // Node 0: (0, 0, 0)
        jacobian[(0, 0)] = dx00 * y00z00;
        jacobian[(0, 1)] = x00 * dy00z00;
        jacobian[(0, 2)] = x00 * y00dz00;

        // Node 1: (0.5, 0, 0)
        jacobian[(1, 0)] = dx05 * y00z00;
        jacobian[(1, 1)] = x05 * dy00z00;
        jacobian[(1, 2)] = x05 * y00dz00;

        // Node 2: (1, 0, 0)
        jacobian[(2, 0)] = dx10 * y00z00;
        jacobian[(2, 1)] = x10 * dy00z00;
        jacobian[(2, 2)] = x10 * y00dz00;

        // Node 3: (0, 0.5, 0)
        jacobian[(3, 0)] = dx00 * y05z00;
        jacobian[(3, 1)] = x00 * dy05z00;
        jacobian[(3, 2)] = x00 * y05dz00;

        // Node 4: (1, 0.5, 0)
        jacobian[(4, 0)] = dx10 * y05z00;
        jacobian[(4, 1)] = x10 * dy05z00;
        jacobian[(4, 2)] = x10 * y05dz00;

        // Node 5: (0, 1, 0)
        jacobian[(5, 0)] = dx00 * y10z00;
        jacobian[(5, 1)] = x00 * dy10z00;
        jacobian[(5, 2)] = x00 * y10dz00;

        // Node 6: (0.5, 1, 0)
        jacobian[(6, 0)] = dx05 * y10z00;
        jacobian[(6, 1)] = x05 * dy10z00;
        jacobian[(6, 2)] = x05 * y10dz00;

        // Node 7: (1, 1, 0)
        jacobian[(7, 0)] = dx10 * y10z00;
        jacobian[(7, 1)] = x10 * dy10z00;
        jacobian[(7, 2)] = x10 * y10dz00;

        let y00z05: f64 = y00 * z05;
        let y10z05: f64 = y10 * z05;

        let dy00z05: f64 = dy00 * z05;
        let dy10z05: f64 = dy10 * z05;

        let y00dz05: f64 = y00 * dz05;
        let y10dz05: f64 = y10 * dz05;

        // Node 8: (0, 0, 0.5)
        jacobian[(8, 0)] = dx00 * y00z05;
        jacobian[(8, 1)] = x00 * dy00z05;
        jacobian[(8, 2)] = x00 * y00dz05;

        // Node 9: (1, 0, 0.5)
        jacobian[(9, 0)] = dx10 * y00z05;
        jacobian[(9, 1)] = x10 * dy00z05;
        jacobian[(9, 2)] = x10 * y00dz05;

        // Node 10: (0, 1, 0.5)
        jacobian[(10, 0)] = dx00 * y10z05;
        jacobian[(10, 1)] = x00 * dy10z05;
        jacobian[(10, 2)] = x00 * y10dz05;

        // Node 11: (1, 1, 0.5)
        jacobian[(11, 0)] = dx10 * y10z05;
        jacobian[(11, 1)] = x10 * dy10z05;
        jacobian[(11, 2)] = x10 * y10dz05;

        let y00z10: f64 = y00 * z10;
        let y05z10: f64 = y05 * z10;
        let y10z10: f64 = y10 * z10;

        let dy00z10: f64 = dy00 * z10;
        let dy05z10: f64 = dy05 * z10;
        let dy10z10: f64 = dy10 * z10;

        let y00dz10: f64 = y00 * dz10;
        let y05dz10: f64 = y05 * dz10;
        let y10dz10: f64 = y10 * dz10;

        // Node 12: (0, 0, 1)
        jacobian[(12, 0)] = dx00 * y00z10;
        jacobian[(12, 1)] = x00 * dy00z10;
        jacobian[(12, 2)] = x00 * y00dz10;

        // Node 13: (0.5, 0, 1)
        jacobian[(13, 0)] = dx05 * y00z10;
        jacobian[(13, 1)] = x05 * dy00z10;
        jacobian[(13, 2)] = x05 * y00dz10;

        // Node 14: (1, 0, 1)
        jacobian[(14, 0)] = dx10 * y00z10;
        jacobian[(14, 1)] = x10 * dy00z10;
        jacobian[(14, 2)] = x10 * y00dz10;

        // Node 15: (0, 0.5, 1)
        jacobian[(15, 0)] = dx00 * y05z10;
        jacobian[(15, 1)] = x00 * dy05z10;
        jacobian[(15, 2)] = x00 * y05dz10;

        // Node 16: (1, 0.5, 1)
        jacobian[(16, 0)] = dx10 * y05z10;
        jacobian[(16, 1)] = x10 * dy05z10;
        jacobian[(16, 2)] = x10 * y05dz10;

        // Node 17: (0, 1, 1)
        jacobian[(17, 0)] = dx00 * y10z10;
        jacobian[(17, 1)] = x00 * dy10z10;
        jacobian[(17, 2)] = x00 * y10dz10;

        // Node 18: (0.5, 1, 1)
        jacobian[(18, 0)] = dx05 * y10z10;
        jacobian[(18, 1)] = x05 * dy10z10;
        jacobian[(18, 2)] = x05 * y10dz10;

        // Node 19: (1, 1, 1)
        jacobian[(19, 0)] = dx10 * y10z10;
        jacobian[(19, 1)] = x10 * dy10z10;
        jacobian[(19, 2)] = x10 * y10dz10;

        jacobian
    }

}