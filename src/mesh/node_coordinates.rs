use std::io::{self, BufRead};
use std::str::FromStr;
use nalgebra::{Vector3, Vector2, Matrix3xX, Matrix2xX, U3, U2};
use nalgebra::dimension::{Dynamic};

#[derive(Debug)]
pub struct Node3(Vector3<f64>);

#[derive(Debug)]
pub struct Node2(Vector2<f64>);

impl Node3 {
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Node3(Vector3::new(x, y, z))
    }
}

impl Node2 {
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Node2(Vector2::new(x, y))
    }
}

/// Common parsing function for both 2D and 3D cases
fn parse_line<const N: usize>(line: &str) -> io::Result<[f64; N]> {
    let line = line.trim();
    if line.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Empty line encountered",
        ));
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
        
        *coord = f64::from_str(part).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid coordinate at position {}: {}", i, e),
            )
        })?;
        count += 1;
    }

    // Check for correct number of coordinates
    if count != N {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Expected {} coordinates, found {}", N, count),
        ));
    }

    // Check for extra coordinates
    if parts.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Found more than {} coordinates", N),
        ));
    }

    Ok(coords)
}

pub fn read_nodes_3D<R: io::Read>(reader: R) -> io::Result<Matrix3xX<f64>> {
    let reader = io::BufReader::new(reader);
    let mut nodes = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let coords: [f64; 3] = parse_line(&line)?;
        nodes.push(Vector3::new(coords[0], coords[1], coords[2]));
    }

    if nodes.is_empty() {
        return Ok(Matrix3xX::zeros(0));
    }

    Ok(Matrix3xX::from_columns(&nodes))
}

pub fn read_nodes_2D<R: io::Read>(reader: R) -> io::Result<Matrix2xX<f64>> {
    let reader = io::BufReader::new(reader);
    let mut nodes = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let coords: [f64; 2] = parse_line(&line)?;
        nodes.push(Vector2::new(coords[0], coords[1]));
    }

    if nodes.is_empty() {
        return Ok(Matrix2xX::zeros(0));
    }

    Ok(Matrix2xX::from_columns(&nodes))
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
        
        let nodes = read_nodes_3D(data).unwrap();
        
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].x, 0.0234);
        assert_eq!(nodes[0].y, 3.45);
        assert_eq!(nodes[0].z, 7.546);
        
        assert_eq!(nodes[1].x, 2.4534);
        assert_eq!(nodes[1].y, 564.44);
        assert_eq!(nodes[1].z, 6.453);
        
        assert_eq!(nodes[2].x, 5.34);
        assert_eq!(nodes[2].y, 7.883);
        assert_eq!(nodes[2].z, 10.44);
    }
    
    #[test]
    fn test_empty_input() {
        let data = "".as_bytes();
        let nodes = read_nodes_3D(data).unwrap();
        assert!(nodes.is_empty());
    }
    
    #[test]
    fn test_invalid_input() {
        let data = "1.0 2.0".as_bytes();
        let result = read_nodes_3D(data);
        assert!(result.is_err());
    }
}