use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct MeshNodeConverter {
    // element_to_nodes[dense_index] -> Vec of global node numbers
    element_to_nodes: Vec<Vec<u32>>,
    // node_to_elements[global_node_id] -> Vec of (element_id, local_node_num)
    node_to_elements: Vec<Vec<(u32, u8)>>,
    // index_to_element_id[dense_index] -> element_id
    index_to_element_id: Vec<u32>,
    // Total number of nodes (for sizing the node_to_elements Vec)
    max_node_id: u32,
    // Total number of elements (for bounds checking)
    num_elements: usize,
}

#[derive(Debug)]
pub enum MeshError {
    IoError(std::io::Error),
    ParseError(String),
    ElementNotFound(u32),
    InvalidLocalNode(u8),
    NodeOutOfRange(u8),
}

impl From<std::io::Error> for MeshError {
    fn from(err: std::io::Error) -> Self {
        MeshError::IoError(err)
    }
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::IoError(e) => write!(f, "IO error: {}", e),
            MeshError::ParseError(s) => write!(f, "Parse error: {}", s),
            MeshError::ElementNotFound(id) => write!(f, "Element {} not found", id),
            MeshError::InvalidLocalNode(num) => write!(f, "Invalid local node number {}", num),
            MeshError::NodeOutOfRange(num) => write!(f, "Local node number {} out of range", num),
        }
    }
}

impl std::error::Error for MeshError {}

impl MeshNodeConverter {
    pub fn new<P: AsRef<Path>>(connectivity_file: P) -> Result<Self, MeshError> {
        let file = File::open(&connectivity_file)?;
        let reader = BufReader::new(file);

        // First pass: count elements and find max node ID
        let mut max_node_id = 0;
        let mut element_count = 0;

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            element_count += 1;
            
            for s in &parts[1..] {
                let node_id: u32 = s.parse().map_err(|_| {
                    MeshError::ParseError(format!("Node ID {} is invalid as u32", s))
                })?;
                if node_id > max_node_id {
                    max_node_id = node_id;
                }
            }
        }

        // Reset file reader for second pass
        let file = File::open(&connectivity_file)?;
        let reader = BufReader::new(file);

        // Pre-allocate with known capacities
        let mut element_to_nodes = Vec::with_capacity(element_count);
        let mut index_to_element_id = Vec::with_capacity(element_count);
        let mut node_to_elements = vec![Vec::new(); max_node_id as usize + 1];

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let element_id: u32 = parts[0].parse().map_err(|_| {
                MeshError::ParseError(format!("Node ID {} is invalid as u32", parts[0]))
            })?;
            
            let mut node_ids = Vec::with_capacity(parts.len() - 1);
            for s in &parts[1..] {
                let node_id = s.parse().map_err(|_| {
                    MeshError::ParseError(format!("Invalid node ID: {}", s))
                })?;
                node_ids.push(node_id);
            }

            let dense_idx = element_to_nodes.len();
            index_to_element_id.push(element_id);
            element_to_nodes.push(node_ids);

            // Update node_to_elements mapping
            for (local_idx, &node_id) in element_to_nodes[dense_idx].iter().enumerate() {
                node_to_elements[node_id as usize].push((element_id, local_idx as u8));
            }
        }

        Ok(Self {
            element_to_nodes,
            node_to_elements,
            index_to_element_id,
            max_node_id,
            num_elements: element_count,
        })
    }

    /// Convert local node number to global node number (strictly 0-indexed)
    pub fn local_to_global(&self, element_id: u32, local_node_num: u8) -> Result<u32, MeshError> {
        let dense_idx = self.find_element_index(element_id)?;
        let nodes = &self.element_to_nodes[dense_idx];
        
        if (local_node_num as usize) < nodes.len() {
            Ok(nodes[local_node_num as usize])
        } else {
            Err(MeshError::NodeOutOfRange(local_node_num))
        }
    }

    /// Get all global node numbers for the given element IDs
    pub fn get_global_nodes_for_elements(&self, element_ids: &[u32]) -> Result<Vec<(u32, Vec<u32>)>, MeshError> {
        let mut result = Vec::with_capacity(element_ids.len());
        
        for &elem_id in element_ids {
            let dense_idx = self.find_element_index(elem_id)?;
            result.push((elem_id, self.element_to_nodes[dense_idx].clone()));
        }
        
        Ok(result)
    }

    /// Convert a pair of local node numbers to global node numbers (strictly 0-indexed)
    pub fn local_pair_to_global(
        &self,
        element_id: u32,
        local_node1: u8,
        local_node2: u8,
    ) -> Result<(u32, u32), MeshError> {
        let dense_idx = self.find_element_index(element_id)?;
        let nodes = &self.element_to_nodes[dense_idx];
        
        if (local_node1 as usize) >= nodes.len() || (local_node2 as usize) >= nodes.len() {
            return Err(MeshError::NodeOutOfRange(local_node1.max(local_node2)));
        }
        
        Ok((nodes[local_node1 as usize], nodes[local_node2 as usize]))
    }

    /// Get all global node pairs for the given element IDs
    pub fn get_global_pairs_for_elements(&self, element_ids: &[u32]) -> Result<Vec<(u32, Vec<(u32, u32)>)>, MeshError> {
        let mut result = Vec::with_capacity(element_ids.len());
        
        for &elem_id in element_ids {
            let dense_idx = self.find_element_index(elem_id)?;
            let nodes = &self.element_to_nodes[dense_idx];
            let num_nodes = nodes.len() as usize;
            let mut pairs = Vec::with_capacity(num_nodes * (num_nodes - 1) / 2);
            
            for i in 0..num_nodes {
                for j in i..num_nodes {
                    pairs.push((nodes[i], nodes[j]));
                }
            }
            
            result.push((elem_id, pairs));
        }
        
        Ok(result)
    }

    /// Get all possible pairs of local node numbers for an element
    pub fn get_all_local_pairs(&self, element_id: u32) -> Result<Vec<(u8, u8)>, MeshError> {
        let dense_idx = self.find_element_index(element_id)?;
        let num_nodes = self.element_to_nodes[dense_idx].len() as u8;
        let mut pairs = Vec::with_capacity((num_nodes as usize) * (num_nodes as usize - 1) / 2);
        
        for i in 0..num_nodes  {
            for j in i..num_nodes {
                pairs.push((i, j));
            }
        }
        
        Ok(pairs)
    }

    /// Get all possible pairs of global node numbers for an element
    pub fn get_all_global_pairs(&self, element_id: u32) -> Result<Vec<(u32, u32)>, MeshError> {
        let dense_idx = self.find_element_index(element_id)?;
        let nodes = &self.element_to_nodes[dense_idx];
        let num_nodes = nodes.len() as usize;
        let mut pairs = Vec::with_capacity((num_nodes as usize) * (num_nodes as usize - 1) / 2);
        
        for i in 0..num_nodes {
            for j in i..num_nodes {
                pairs.push((nodes[i], nodes[j]));
            }
        }
        
        Ok(pairs)
    }

    /// Helper function to find element index
    fn find_element_index(&self, element_id: u32) -> Result<usize, MeshError> {
        self.index_to_element_id.iter()
            .position(|&id| id == element_id)
            .ok_or(MeshError::ElementNotFound(element_id))
    }

    /// Get the maximum node ID in the mesh
    pub fn max_node_id(&self) -> u32 {
        self.max_node_id
    }

    /// Get the number of elements in the mesh
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use rand::{rng, Rng, seq::SliceRandom};

    // Helper function to generate test files
    fn generate_test_file(
        filename: &str,
        num_elements: u32,
        max_nodes_per_element: usize,
        num_unique_nodes: u32,
    ) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        let mut rng = rng();

        for elem_id in 0..num_elements {
            let num_nodes = rng.random_range(3..=max_nodes_per_element);
            
            // Create a vector of node IDs (0..num_unique_nodes)
            let mut all_nodes: Vec<u32> = (0..num_unique_nodes).collect();
            
            // Shuffle and take the first num_nodes
            all_nodes.shuffle(&mut rng);
            let nodes = &all_nodes[..num_nodes];
            
            write!(file, "{}", elem_id)?;
            for node in nodes {
                write!(file, " {}", node)?;
            }
            writeln!(file)?;
        }

        Ok(())
    }

    // Helper function to create a simple deterministic test file
    fn create_simple_test_file(filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        writeln!(file, "0 10 11 12")?;  // Element 0 with nodes 10, 11, 12
        writeln!(file, "1 11 12 13")?;  // Element 1 with nodes 11, 12, 13
        writeln!(file, "2 12 13 14")?;  // Element 2 with nodes 12, 13, 14
        Ok(())
    }

    //#[test]
    //#[ignore = "not yet implemented"]
    fn test_new_with_valid_file() {
        let filename = "input/mesh/test_mesh.txt";
        create_simple_test_file(filename).unwrap();
        
        let converter = MeshNodeConverter::new(filename).unwrap();
        
        assert_eq!(converter.num_elements(), 3);
        assert_eq!(converter.max_node_id(), 14);
        
        std::fs::remove_file(filename).unwrap();
    }

    //#[test]
    fn test_new_with_nonexistent_file() {
        let result = MeshNodeConverter::new("nonexistent_file.txt");
        assert!(matches!(result, Err(MeshError::IoError(_))));
    }

    //#[test]
    fn test_local_to_global() {
        let filename = "input/mesh/test_local_global.txt";
        create_simple_test_file(filename).unwrap();
        
        let converter = MeshNodeConverter::new(filename).unwrap();
        
        assert_eq!(converter.local_to_global(0, 0).unwrap(), 10);
        assert_eq!(converter.local_to_global(0, 1).unwrap(), 11);
        assert_eq!(converter.local_to_global(0, 2).unwrap(), 12);
        assert_eq!(converter.local_to_global(1, 0).unwrap(), 11);
        
        // Test invalid cases
        assert!(matches!(converter.local_to_global(99, 0), Err(MeshError::ElementNotFound(99))));
        assert!(matches!(converter.local_to_global(0, 3), Err(MeshError::NodeOutOfRange(3))));
        
        std::fs::remove_file(filename).unwrap();
    }

    //#[test]
    fn test_local_pair_to_global() {
        let filename = "input/mesh/test_local_pair.txt";
        create_simple_test_file(filename).unwrap();
        
        let converter = MeshNodeConverter::new(filename).unwrap();
        
        assert_eq!(converter.local_pair_to_global(0, 0, 1).unwrap(), (10, 11));
        assert_eq!(converter.local_pair_to_global(0, 1, 2).unwrap(), (11, 12));
        assert_eq!(converter.local_pair_to_global(1, 0, 2).unwrap(), (11, 13));
        
        // Test invalid cases
        assert!(matches!(
            converter.local_pair_to_global(99, 0, 1),
            Err(MeshError::ElementNotFound(99))
        ));
        assert!(matches!(
            converter.local_pair_to_global(0, 3, 0),
            Err(MeshError::NodeOutOfRange(3))
        ));
        
        std::fs::remove_file(filename).unwrap();
    }

    //#[test]
    fn test_get_global_nodes_for_elements() {
        let filename = "input/mesh/test_global_nodes.txt";
        create_simple_test_file(filename).unwrap();
        
        let converter = MeshNodeConverter::new(filename).unwrap();
        
        let result = converter.get_global_nodes_for_elements(&[0, 1]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0, vec![10, 11, 12]));
        assert_eq!(result[1], (1, vec![11, 12, 13]));
        
        // Test with non-existent element
        assert!(matches!(
            converter.get_global_nodes_for_elements(&[0, 99]),
            Err(MeshError::ElementNotFound(99))
        ));
        
        std::fs::remove_file(filename).unwrap();
    }

    //#[test]
    fn test_get_all_local_pairs() {
        let filename = "input/mesh/test_local_pairs.txt";
        create_simple_test_file(filename).unwrap();
        
        let converter = MeshNodeConverter::new(filename).unwrap();
        
        let pairs = converter.get_all_local_pairs(0).unwrap();
        println!("Pairs {:?}", pairs);
        assert_eq!(pairs, vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]);
        
        // Test with non-existent element
        assert!(matches!(
            converter.get_all_local_pairs(99),
            Err(MeshError::ElementNotFound(99))
        ));
        
        std::fs::remove_file(filename).unwrap();
    }

    //#[test]
    fn test_get_all_global_pairs() {
        let filename = "input/mesh/test_global_pairs.txt";
        create_simple_test_file(filename).unwrap();
        
        let converter = MeshNodeConverter::new(filename).unwrap();
        
        let pairs = converter.get_all_global_pairs(0).unwrap();
        println!("Global Pairs {:?}", pairs);
        assert_eq!(pairs, vec![(10, 10), (10, 11), (10, 12), (11, 11), (11, 12), (12, 12)]);
        
        std::fs::remove_file(filename).unwrap();
    }

    //#[test]
    fn test_get_global_pairs_for_elements() {
        let filename = "input/mesh/test_global_pairs_multi.txt";
        create_simple_test_file(filename).unwrap();
        
        let converter = MeshNodeConverter::new(filename).unwrap();
        
        let result = converter.get_global_pairs_for_elements(&[0, 1]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0);
        assert_eq!(result[0].1, vec![(10, 10), (10, 11), (10, 12), (11, 11), (11, 12), (12, 12)]);
        assert_eq!(result[1].0, 1);
        assert_eq!(result[1].1, vec![(11, 11), (11, 12), (11, 13), (12, 12), (12, 13), (13, 13)]);
        
        std::fs::remove_file(filename).unwrap();
    }

    //#[test]
    fn test_large_mesh() {
        let filename = "input/mesh/large_mesh_test.txt";
        let num_elements = 1000;
        let max_nodes_per_element = 27;
        let num_unique_nodes = 5000;
        
        generate_test_file(filename, num_elements, max_nodes_per_element, num_unique_nodes).unwrap();
        
        let converter = MeshNodeConverter::new(filename).unwrap();
        
        assert_eq!(converter.num_elements(), num_elements as usize);
        assert!(converter.max_node_id() < num_unique_nodes);
        
        // Test random elements
        for _ in 0..10 {
            let elem_id = rand::rng().random_range(0..num_elements);
            let nodes = converter.get_global_nodes_for_elements(&[elem_id]).unwrap();
            assert_eq!(nodes.len(), 1);
            assert!(nodes[0].1.len() >= 3 && nodes[0].1.len() <= max_nodes_per_element);
        }
        
        std::fs::remove_file(filename).unwrap();
    }

    //#[test]
    fn test_node_to_elements_mapping() {
        let filename = "input/mesh/test_node_mapping.txt";
        create_simple_test_file(filename).unwrap();
        
        let converter = MeshNodeConverter::new(filename).unwrap();
        
        // Node 11 is in elements 0 and 1
        let elements = &converter.node_to_elements[11];
        assert_eq!(elements.len(), 2);
        assert!(elements.contains(&(0, 1)));  // Node 11 is local node 1 in element 0
        assert!(elements.contains(&(1, 0)));  // Node 11 is local node 0 in element 1
        
        // Node 14 is only in element 2
        let elements = &converter.node_to_elements[14];
        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0], (2, 2));
        
        std::fs::remove_file(filename).unwrap();
    }
}