use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct MeshNodeConverter {
    element_to_nodes: Vec<Vec<u32>>,
    index_to_element_id: Vec<u32>, // Sorted array for binary search
    max_node_id: u32,
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
        // First pass: count elements and find max node ID
        let (max_node_id, element_count) = Self::first_pass(&connectivity_file)?;

        // Second pass: build data structures
        let (element_to_nodes, index_to_element_id) = 
            Self::second_pass(&connectivity_file, element_count)?;

        // Create a vector of indices and sort them based on element IDs
        let mut indices: Vec<usize> = (0..index_to_element_id.len()).collect();
        indices.sort_by_key(|&i| index_to_element_id[i]);

        // Reorder both arrays using the sorted indices
        let index_to_element_id = indices.iter().map(|&i| index_to_element_id[i]).collect();
        let element_to_nodes = indices.iter().map(|&i| element_to_nodes[i].clone()).collect();

        Ok(Self {
            element_to_nodes,
            index_to_element_id,
            max_node_id,
            num_elements: element_count,
        })
    }

    fn first_pass<P: AsRef<Path>>(path: P) -> Result<(u32, usize), MeshError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut max_node_id = 0;
        let mut element_count = 0;

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.trim().split_whitespace();
            
            if parts.next().is_none() {
                continue;
            }

            element_count += 1;
            
            for s in parts {
                let node_id: u32 = s.parse().map_err(|_| {
                    MeshError::ParseError(format!("Node ID {} is invalid as u32", s))
                })?;
                max_node_id = max_node_id.max(node_id);
            }
        }

        Ok((max_node_id, element_count))
    }

    fn second_pass<P: AsRef<Path>>(
        path: P,
        element_count: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<u32>), MeshError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut element_to_nodes = Vec::with_capacity(element_count);
        let mut index_to_element_id = Vec::with_capacity(element_count);

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.trim().split_whitespace();
            
            let Some(element_str) = parts.next() else { continue };
            let element_id: u32 = element_str.parse().map_err(|_| {
                MeshError::ParseError(format!("Element ID {} is invalid as u32", element_str))
            })?;
            
            let node_ids: Result<Vec<u32>, MeshError> = parts
                .map(|s| s.parse().map_err(|_| MeshError::ParseError(format!("Invalid node ID: {}", s))))
                .collect();
            let node_ids = node_ids?;

            index_to_element_id.push(element_id);
            element_to_nodes.push(node_ids);
        }

        Ok((element_to_nodes, index_to_element_id))
    }

    pub fn local_to_global(&self, element_id: u32, local_node_num: u8) -> Result<u32, MeshError> {
        let nodes = self.get_element_nodes(element_id)?;
        
        nodes.get(local_node_num as usize)
            .copied()
            .ok_or(MeshError::NodeOutOfRange(local_node_num))
    }

    pub fn get_global_nodes_for_elements(&self, element_ids: &[u32]) -> Result<Vec<(u32, Vec<u32>)>, MeshError> {
        element_ids.iter()
            .map(|&elem_id| {
                let nodes = self.get_element_nodes(elem_id)?;
                Ok((elem_id, nodes.to_vec()))
            })
            .collect()
    }

    pub fn local_pair_to_global(
        &self,
        element_id: u32,
        local_node1: u8,
        local_node2: u8,
    ) -> Result<(u32, u32), MeshError> {
        let nodes = self.get_element_nodes(element_id)?;
        
        if (local_node1 as usize) >= nodes.len() || (local_node2 as usize) >= nodes.len() {
            return Err(MeshError::NodeOutOfRange(local_node1.max(local_node2)));
        }
        
        Ok((nodes[local_node1 as usize], nodes[local_node2 as usize]))
    }

    pub fn get_global_pairs_for_elements(&self, element_ids: &[u32]) -> Result<Vec<(u32, Vec<(u32, u32)>)>, MeshError> {
        element_ids.iter()
            .map(|&elem_id| {
                let nodes = self.get_element_nodes(elem_id)?;
                let pairs = nodes.iter()
                    .enumerate()
                    .flat_map(|(i, &x)| nodes[i..].iter().map(move |&y| (x, y)))
                    .collect();
                Ok((elem_id, pairs))
            })
            .collect()
    }

    pub fn get_all_local_pairs(&self, element_id: u32) -> Result<Vec<(u8, u8)>, MeshError> {
        let num_nodes = self.get_element_nodes(element_id)?.len() as u8;
        let nodes: Vec<u8> = (0..num_nodes).collect();  // Convert range to Vec<u8>
        Ok(Self::generate_all_pairs(&nodes))  // Pass a reference to the Vec
    }

    pub fn get_all_global_pairs(&self, element_id: u32) -> Result<Vec<(u32, u32)>, MeshError> {
        let nodes = self.get_element_nodes(element_id)?;
        Ok(nodes.iter()
            .enumerate()
            .flat_map(|(i, &x)| nodes[i..].iter().map(move |&y| (x, y)))
            .collect())
    }

    fn generate_all_pairs<T: Copy>(items: &[T]) -> Vec<(T, T)> {
        let mut pairs = Vec::with_capacity(items.len().pow(2) / 2);
        
        for (i, &item1) in items.iter().enumerate() {
            for &item2 in &items[i..] {
                pairs.push((item1, item2));
            }
        }
        
        pairs
    }

    fn get_element_nodes(&self, element_id: u32) -> Result<&Vec<u32>, MeshError> {
        let dense_idx = self.find_element_index(element_id)?;
        Ok(&self.element_to_nodes[dense_idx])
    }

    fn find_element_index(&self, element_id: u32) -> Result<usize, MeshError> {
        self.index_to_element_id
            .binary_search(&element_id)
            .map_err(|_| MeshError::ElementNotFound(element_id))
    }

    pub fn max_node_id(&self) -> u32 {
        self.max_node_id
    }

    pub fn num_elements(&self) -> usize {
        self.num_elements
    }
}

#[cfg(not(test))]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use rand::{rng, Rng, seq::SliceRandom};

    // Helper function to generate test files
    fn generate_test_file(
        filename: &str,
        num_elements: u32,
        max_nodes_per_element: usize,
        num_unique_nodes: u32,
    ) -> std::io::Result<()> {
        let mut file = File::create(filename)?;

        for elem_id in 0..num_elements {
            let num_nodes = rng().random_range(3..=max_nodes_per_element);
            
            // Create a vector of node IDs (0..num_unique_nodes)
            let mut all_nodes: Vec<u32> = (0..num_unique_nodes).collect();
            
            // Shuffle and take the first num_nodes
            all_nodes.shuffle(&mut rng());
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
    fn create_simple_test_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "0 10 11 12").unwrap();  // Element 0 with nodes 10, 11, 12
        writeln!(file, "1 11 12 13").unwrap();  // Element 1 with nodes 11, 12, 13
        writeln!(file, "2 12 13 14").unwrap();  // Element 2 with nodes 12, 13, 14
        file
    }

    //#[test]
    fn test_new_with_simple_file() {
        let file = create_simple_test_file();
        let converter = MeshNodeConverter::new(file.path()).unwrap();
        
        assert_eq!(converter.max_node_id(), 14);
        assert_eq!(converter.num_elements(), 3);
    }

    //#[test]
    fn test_local_to_global() {
        let file = create_simple_test_file();
        let converter = MeshNodeConverter::new(file.path()).unwrap();
        
        assert_eq!(converter.local_to_global(0, 0).unwrap(), 10);
        assert_eq!(converter.local_to_global(0, 1).unwrap(), 11);
        assert_eq!(converter.local_to_global(0, 2).unwrap(), 12);
        assert_eq!(converter.local_to_global(1, 0).unwrap(), 11);
        assert_eq!(converter.local_to_global(2, 2).unwrap(), 14);
        
        // Test error cases
        assert!(matches!(converter.local_to_global(0, 3), Err(MeshError::NodeOutOfRange(3))));
        assert!(matches!(converter.local_to_global(99, 0), Err(MeshError::ElementNotFound(99))));
    }

    //#[test]
    fn test_get_global_nodes_for_elements() {
        let file = create_simple_test_file();
        let converter = MeshNodeConverter::new(file.path()).unwrap();
        
        let result = converter.get_global_nodes_for_elements(&[0, 1]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0, vec![10, 11, 12]));
        assert_eq!(result[1], (1, vec![11, 12, 13]));
    }

    //#[test]
    fn test_local_pair_to_global() {
        let file = create_simple_test_file();
        let converter = MeshNodeConverter::new(file.path()).unwrap();
        
        assert_eq!(converter.local_pair_to_global(0, 0, 1).unwrap(), (10, 11));
        assert_eq!(converter.local_pair_to_global(1, 1, 2).unwrap(), (12, 13));
        assert_eq!(converter.local_pair_to_global(2, 0, 2).unwrap(), (12, 14));
        
        // Test error cases
        assert!(matches!(
            converter.local_pair_to_global(0, 0, 3),
            Err(MeshError::NodeOutOfRange(3))
        ));
    }

    //#[test]
    fn test_get_all_local_pairs() {
        let file = create_simple_test_file();
        let converter = MeshNodeConverter::new(file.path()).unwrap();
        
        let pairs = converter.get_all_local_pairs(0).unwrap();
        assert_eq!(pairs.len(), 6); // 3 nodes = 3*2/2 + 3 = 6 pairs (including self-pairs)
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(0, 2)));
        assert!(pairs.contains(&(1, 2)));
        assert!(pairs.contains(&(0, 0)));
        assert!(pairs.contains(&(1, 1)));
        assert!(pairs.contains(&(2, 2)));
    }

    //#[test]
    fn test_get_all_global_pairs() {
        let file = create_simple_test_file();
        let converter = MeshNodeConverter::new(file.path()).unwrap();
        
        let pairs = converter.get_all_global_pairs(0).unwrap();
        assert_eq!(pairs.len(), 6);
        assert!(pairs.contains(&(10, 11)));
        assert!(pairs.contains(&(10, 12)));
        assert!(pairs.contains(&(11, 12)));
        assert!(pairs.contains(&(10, 10)));
        assert!(pairs.contains(&(11, 11)));
        assert!(pairs.contains(&(12, 12)));
    }

    //#[test]
    fn test_large_file_performance() {
        let filename = "input/mesh/large_mesh_test.txt";
        let num_elements = 1_000;
        let max_nodes_per_element = 27;
        let num_unique_nodes = 5_000;
        generate_test_file(filename, num_elements, max_nodes_per_element, num_unique_nodes).unwrap();
        
        // Time the construction
        let start = std::time::Instant::now();
        let converter = MeshNodeConverter::new(filename).unwrap();
        let duration = start.elapsed();
        
        println!("Construction time for large file: {:?}", duration);
        assert!(duration < std::time::Duration::from_secs(1), "Construction took too long");
        
        // Test lookup performance
        let start = std::time::Instant::now();
        let number_of_queries = 10_000;
        for _ in 0..number_of_queries {
            let _ = converter.local_to_global(5000, 5);
        }
        let duration = start.elapsed();
        println!("Lookup time for {:?} queries: {:?}", number_of_queries, duration);
        assert!(duration < std::time::Duration::from_millis(100), "Lookups took too long");
    }

    //#[test]
    fn test_element_sorting() {
        // Create a file with non-sequential element IDs
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "100 1 2 3").unwrap();
        writeln!(file, "50 4 5 6").unwrap();
        writeln!(file, "200 7 8 9").unwrap();
        
        let converter = MeshNodeConverter::new(file.path()).unwrap();
        
        // Verify elements are stored in sorted order
        assert_eq!(converter.index_to_element_id, vec![50, 100, 200]);
        
        // Verify the element_to_nodes mapping is consistent with the sorted order
        assert_eq!(converter.element_to_nodes[0], vec![4, 5, 6]); // Element 50
        assert_eq!(converter.element_to_nodes[1], vec![1, 2, 3]); // Element 100
        assert_eq!(converter.element_to_nodes[2], vec![7, 8, 9]); // Element 200
    }

    //#[test]
    fn test_invalid_file_format() {
        // File with non-numeric data
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "abc def ghi").unwrap();
        writeln!(file, "1 2 3").unwrap();
        
        let result = MeshNodeConverter::new(file.path());
        assert!(matches!(result, Err(MeshError::ParseError(_))));
        
        // File with missing element ID
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "1 2 3").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "2 4 5").unwrap();
        
        let converter = MeshNodeConverter::new(file.path()).unwrap();
        assert_eq!(converter.num_elements(), 2);
    }

    //#[test]
    fn test_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let converter = MeshNodeConverter::new(file.path()).unwrap();
        
        assert_eq!(converter.num_elements(), 0);
        assert_eq!(converter.max_node_id(), 0);
    }
}