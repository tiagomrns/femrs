//! Module for reading and representing 2D and 3D nodes from input data.
//!
//! This module provides structures and functions for parsing node coordinates
//! from text input and representing them as validated 2D or 3D points.

use std::io::BufRead;
use std::str::FromStr;
use ndarray::{Array2, Array1};

/// Error types for node parsing and validation.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeError {
    /// Empty input encountered
    EmptyInput,
    /// Invalid coordinate format
    InvalidCoordinate { position: usize, value: String },
    /// Incorrect number of coordinates
    WrongCoordinateCount { expected: usize, found: usize },
    /// Extra coordinates found beyond expected count
    ExtraCoordinates { expected: usize },
    /// Invalid array dimension for node type
    InvalidDimension { expected: usize, found: usize },
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeError::EmptyInput => write!(f, "Empty input encountered"),
            NodeError::InvalidCoordinate { position, value } => {
                write!(f, "Invalid coordinate at position {}: '{}'", position, value)
            }
            NodeError::WrongCoordinateCount { expected, found } => {
                write!(f, "Expected {} coordinates, found {}", expected, found)
            }
            NodeError::ExtraCoordinates { expected } => {
                write!(f, "Found more than {} coordinates", expected)
            }
            NodeError::InvalidDimension { expected, found } => {
                write!(f, "Expected {} elements, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for NodeError {}

/// Represents a 3D node with exactly three coordinates (x, y, z).
///
/// # Examples
/// ```
/// use node_reader::Node3;
///
/// let node = Node3::new(1.0, 2.0, 3.0);
/// assert_eq!(node.x(), 1.0);
/// assert_eq!(node.y(), 2.0);
/// assert_eq!(node.z(), 3.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Node3(Array1<f64>);

/// Represents a 2D node with exactly two coordinates (x, y).
///
/// # Examples
/// ```
/// use node_reader::Node2;
///
/// let node = Node2::new(1.0, 2.0);
/// assert_eq!(node.x(), 1.0);
/// assert_eq!(node.y(), 2.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Node2(Array1<f64>);

impl Node3 {
    /// Creates a new 3D node with the given coordinates.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate
    /// * `y` - The y-coordinate
    /// * `z` - The z-coordinate
    ///
    /// # Returns
    /// A new `Node3` instance.
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Node3(Array1::from_vec(vec![x, y, z]))
    }
    
    /// Creates a 3D node from an existing array, validating the dimension.
    ///
    /// # Arguments
    /// * `arr` - An array containing exactly 3 elements
    ///
    /// # Returns
    /// * `Ok(Node3)` if the array has exactly 3 elements
    /// * `Err(NodeError)` if the array has incorrect dimensions
    #[inline]
    pub fn from_array(arr: Array1<f64>) -> Result<Self, NodeError> {
        if arr.len() != 3 {
            return Err(NodeError::InvalidDimension {
                expected: 3,
                found: arr.len(),
            });
        }
        Ok(Node3(arr))
    }
    
    /// Returns the x-coordinate of the node.
    #[inline]
    pub fn x(&self) -> f64 {
        self.0[0]
    }
    
    /// Returns the y-coordinate of the node.
    #[inline]
    pub fn y(&self) -> f64 {
        self.0[1]
    }
    
    /// Returns the z-coordinate of the node.
    #[inline]
    pub fn z(&self) -> f64 {
        self.0[2]
    }
    
    /// Returns the node coordinates as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.0.as_slice().unwrap()
    }
    
    /// Returns a reference to the underlying array.
    #[inline]
    pub fn as_array(&self) -> &Array1<f64> {
        &self.0
    }
    
    /// Returns the coordinates as a tuple (x, y, z).
    #[inline]
    pub fn as_tuple(&self) -> (f64, f64, f64) {
        (self.0[0], self.0[1], self.0[2])
    }
}

impl Node2 {
    /// Creates a new 2D node with the given coordinates.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate
    /// * `y` - The y-coordinate
    ///
    /// # Returns
    /// A new `Node2` instance.
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Node2(Array1::from_vec(vec![x, y]))
    }
    
    /// Creates a 2D node from an existing array, validating the dimension.
    ///
    /// # Arguments
    /// * `arr` - An array containing exactly 2 elements
    ///
    /// # Returns
    /// * `Ok(Node2)` if the array has exactly 2 elements
    /// * `Err(NodeError)` if the array has incorrect dimensions
    #[inline]
    pub fn from_array(arr: Array1<f64>) -> Result<Self, NodeError> {
        if arr.len() != 2 {
            return Err(NodeError::InvalidDimension {
                expected: 2,
                found: arr.len(),
            });
        }
        Ok(Node2(arr))
    }
    
    /// Returns the x-coordinate of the node.
    #[inline]
    pub fn x(&self) -> f64 {
        self.0[0]
    }
    
    /// Returns the y-coordinate of the node.
    #[inline]
    pub fn y(&self) -> f64 {
        self.0[1]
    }
    
    /// Returns the node coordinates as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.0.as_slice().unwrap()
    }
    
    /// Returns a reference to the underlying array.
    #[inline]
    pub fn as_array(&self) -> &Array1<f64> {
        &self.0
    }
    
    /// Returns the coordinates as a tuple (x, y).
    #[inline]
    pub fn as_tuple(&self) -> (f64, f64) {
        (self.0[0], self.0[1])
    }
}

impl TryFrom<Array1<f64>> for Node3 {
    type Error = NodeError;
    
    /// Attempts to convert an `Array1<f64>` into a `Node3`.
    ///
    /// # Arguments
    /// * `arr` - Array containing node coordinates
    ///
    /// # Returns
    /// * `Ok(Node3)` if the array has exactly 3 elements
    /// * `Err(NodeError)` if the array has incorrect dimensions
    fn try_from(arr: Array1<f64>) -> Result<Self, Self::Error> {
        Node3::from_array(arr)
    }
}

impl TryFrom<Array1<f64>> for Node2 {
    type Error = NodeError;
    
    /// Attempts to convert an `Array1<f64>` into a `Node2`.
    ///
    /// # Arguments
    /// * `arr` - Array containing node coordinates
    ///
    /// # Returns
    /// * `Ok(Node2)` if the array has exactly 2 elements
    /// * `Err(NodeError)` if the array has incorrect dimensions
    fn try_from(arr: Array1<f64>) -> Result<Self, Self::Error> {
        Node2::from_array(arr)
    }
}

impl TryFrom<Vec<f64>> for Node3 {
    type Error = NodeError;
    
    /// Attempts to convert a `Vec<f64>` into a `Node3`.
    ///
    /// # Arguments
    /// * `vec` - Vector containing node coordinates
    ///
    /// # Returns
    /// * `Ok(Node3)` if the vector has exactly 3 elements
    /// * `Err(NodeError)` if the vector has incorrect dimensions
    fn try_from(vec: Vec<f64>) -> Result<Self, Self::Error> {
        if vec.len() != 3 {
            return Err(NodeError::InvalidDimension {
                expected: 3,
                found: vec.len(),
            });
        }
        Ok(Node3(Array1::from_vec(vec)))
    }
}

impl TryFrom<Vec<f64>> for Node2 {
    type Error = NodeError;
    
    /// Attempts to convert a `Vec<f64>` into a `Node2`.
    ///
    /// # Arguments
    /// * `vec` - Vector containing node coordinates
    ///
    /// # Returns
    /// * `Ok(Node2)` if the vector has exactly 2 elements
    /// * `Err(NodeError)` if the vector has incorrect dimensions
    fn try_from(vec: Vec<f64>) -> Result<Self, Self::Error> {
        if vec.len() != 2 {
            return Err(NodeError::InvalidDimension {
                expected: 2,
                found: vec.len(),
            });
        }
        Ok(Node2(Array1::from_vec(vec)))
    }
}

/// Parses a line of text into an array of exactly N floating-point coordinates.
///
/// # Arguments
/// * `line` - A line of text containing coordinates separated by whitespace or commas
///
/// # Returns
/// * `Ok([f64; N])` - Array of parsed coordinates
/// * `Err(NodeError)` - If parsing fails for any reason
fn parse_line<const N: usize>(line: &str) -> Result<[f64; N], NodeError> {
    let line = line.trim();
    if line.is_empty() {
        return Err(NodeError::EmptyInput);
    }

    let mut coords = [0.0; N];
    let mut count = 0;
    let mut parts = line.split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty());

    // Parse exactly N coordinates
    for (i, coord) in coords.iter_mut().enumerate() {
        let part = match parts.next() {
            Some(p) => p,
            None => break,
        };
        
        *coord = f64::from_str(part).map_err(|_| NodeError::InvalidCoordinate {
            position: i,
            value: part.to_string(),
        })?;
        count += 1;
    }

    // Check for correct number of coordinates
    if count != N {
        return Err(NodeError::WrongCoordinateCount {
            expected: N,
            found: count,
        });
    }

    // Check for extra coordinates
    if parts.next().is_some() {
        return Err(NodeError::ExtraCoordinates { expected: N });
    }

    Ok(coords)
}

/// Reads 2D or 3D nodes from a reader and returns them as an array of shape (DIM, n_nodes).
///
/// Every single line of the input should contain exactly DIM coordinates separated by whitespace or commas.
/// For 2D nodes, use DIM=2 with coordinates (x, y).
/// For 3D nodes, use DIM=3 with coordinates (x, y, z).
///
/// The resulting array has shape (DIM, n_nodes), where each column represents a single node.
///
/// # Arguments
/// * `reader` - An input reader implementing `std::io::Read`
///
/// # Returns
/// * `Ok(Array2<f64>)` - 2D array with shape (DIM, n_nodes) containing node coordinates
/// * `Err(NodeError)` - If reading or parsing fails
///
/// # Examples
/// ```
/// use std::io::Cursor;
/// use node_reader::read_nodes;
///
/// // 3D example
/// let data_3d = "1.0 2.0 3.0\n4.0 5.0 6.0\n".as_bytes();
/// let nodes = read_nodes::<3, _>(data_3d).unwrap();
/// assert_eq!(nodes.shape(), [3, 2]);
///
/// // 2D example
/// let data_2d = "1.0 2.0\n3.0 4.0\n".as_bytes();
/// let nodes = read_nodes::<2, _>(data_2d).unwrap();
/// assert_eq!(nodes.shape(), [2, 2]);
/// ```
pub fn read_nodes<const DIM: usize, R: std::io::Read>(reader: R) -> Result<Array2<f64>, NodeError> {
    let reader = std::io::BufReader::new(reader);
    let mut nodes: Vec<[f64; DIM]> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| NodeError::InvalidCoordinate {
            position: 0,
            value: e.to_string(),
        })?;
        
        let coords: [f64; DIM] = parse_line(&line)?;
        nodes.push(coords);
    }

    if nodes.is_empty() {
        return Ok(Array2::zeros((DIM, 0)));
    }

    // Convert Vec<[f64; DIM]> to array with shape (DIM, n_nodes)
    // Each column represents a node
    let n_nodes = nodes.len();
    let mut array = Array2::zeros((DIM, n_nodes));
    
    for (row_idx, node_coords) in nodes.iter().enumerate() {
        for (col_idx, &coord) in node_coords.iter().enumerate() {
            array[[row_idx, col_idx]] = coord;
        }
    }
    
    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_read_nodes() {
        let data = "\
            0.0234 3.45 7.546\n\
            2.4534, 564.44, 6.453\n\
            5.34 7.883 10.44".as_bytes();
        
        let nodes = read_nodes::<3, _>(data).unwrap();
        
        assert_eq!(nodes.shape(), [3,3]);
        //println!("nodes = {:?}", nodes);
        assert_eq!(nodes[[0,0]], 0.0234);
        assert_eq!(nodes[[0,1]], 3.45);
        assert_eq!(nodes[[0,2]], 7.546);
        
        assert_eq!(nodes[[1,0]], 2.4534);
        assert_eq!(nodes[[1,1]], 564.44);
        assert_eq!(nodes[[1,2]], 6.453);
        
        assert_eq!(nodes[[2,0]], 5.34);
        assert_eq!(nodes[[2,1]], 7.883);
        assert_eq!(nodes[[2,2]], 10.44);
    }
    
    #[test]
    fn test_empty_input() {
        let data = "".as_bytes();
        let nodes = read_nodes::<2, _>(data).unwrap();
        assert!(nodes.is_empty());
    }
    
    #[test]
    fn test_invalid_input() {
        let data = "1.0 2.0".as_bytes();
        let result = read_nodes::<3, _>(data);
        assert!(result.is_err());
    }
}