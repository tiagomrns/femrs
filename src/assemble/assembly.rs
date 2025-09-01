use scirs2_sparse::bsr::BsrMatrix;
use scirs2_sparse::SparseResult;
use itertools::Itertools;
use num_integer::binomial;

/// Initialize a stiffness matrix with proper block structure
///
/// # Arguments
/// * `num_node` - Number of nodes in the mesh
/// * `elements` - List of element connectivity (each element is a list of node indices)
/// * `dimension` - Block size (e.g., 2 for 2D problems, 3 for 3D)
///
/// # Returns
/// BSR matrix with the specified block structure
pub fn initialize_stiffness_matrix(
    num_node: usize,
    elements: &[Vec<usize>],
    dimension: usize,
) -> SparseResult<BsrMatrix<f64>> {
    // Track only the block positions (indices)
    let mut rows_of_blocks: Vec<Vec<usize>> = vec![Vec::new(); num_node];

    // Process each element
    for nodes in elements {
        for &i in nodes {
            let row = &mut rows_of_blocks[i];
            for &j in nodes {
                // Insert block if not already present
                if row.binary_search(&j).is_err() {
                    let pos = row.partition_point(|&x| x < j);
                    row.insert(pos, j);
                }
            }
        }
    }

    // Calculate total blocks and allocate data
    let total_blocks: usize = rows_of_blocks.iter().map(Vec::len).sum();

    // Convert to BSR format components
    let mut indices: Vec<Vec<usize>> = Vec::with_capacity(total_blocks);
    let mut indptr: Vec<usize> = Vec::with_capacity(num_node + 1);
    indptr.push(0);

    for row in rows_of_blocks {
        indptr.push(indptr.last().unwrap() + row.len());
        indices.extend(row.into_iter().map(|col| vec![col])); // Push the entire Vec<usize> for this row
    }

    // Create data array filled with ones matrices
    let block_row: Vec<f64> = vec![1.0; dimension];
    let block_values: Vec<Vec<f64>> = vec![block_row; dimension];
    let data: Vec<Vec<Vec<f64>>> = vec![block_values; total_blocks];

    let block_size: (usize, usize) = (dimension, dimension);
    let shape: (usize, usize) = (num_node * dimension, num_node * dimension);

    BsrMatrix::from_blocks(data, indices, indptr, shape, block_size)
}

/// Initialize a nonlinear stiffness matrix with proper block structure
///
/// # Arguments
/// * `num_node` - Number of nodes in the mesh
/// * `elements` - List of element connectivity (each element is a list of node indices)
/// * `dimension` - Block rows (e.g., 2 for 2D problems, 3 for 3D)
/// * `order` - Order of the nonlinear force (e.g. 1 for linear, 2 for quadratic)
///
/// # Returns
/// BSR matrix with the specified block structure
pub fn initialize_nonlinear_stiffness_matrix(
    num_node: usize,
    elements: &[Vec<usize>],
    dimension: usize,
    order: usize,
) -> SparseResult<BsrMatrix<f64>> {
    // Track only the block positions (indices)
    let mut rows_of_blocks: Vec<Vec<usize>> = vec![Vec::new(); num_node];

    // Process each element
    for nodes in elements {
        for &node_row in nodes {
            let row = &mut rows_of_blocks[node_row];
            for node_cols in nodes.iter().combinations_with_replacement(order) {
                // Locate the column number for this combination of column nodes
                let column_block: usize = locate_block_column(num_node, &node_cols);
                // Insert block if not already present
                let pos = row.partition_point(|&x| x < column_block);
                if pos >= row.len() || row[pos] != column_block {
                    row.insert(pos, column_block);
                }
            }
        }
    }

    // Calculate total blocks and allocate data
    let total_blocks: usize = rows_of_blocks.iter().map(Vec::len).sum();

    // Convert to BSR format components
    let mut indices: Vec<Vec<usize>> = Vec::with_capacity(total_blocks);
    let mut indptr: Vec<usize> = Vec::with_capacity(num_node + 1);
    indptr.push(0);

    for row in rows_of_blocks {
        indptr.push(indptr.last().unwrap() + row.len());
        indices.extend(row.into_iter().map(|col| vec![col])); // Push the entire Vec<usize> for this row
    }

    // Create data array filled with ones matrices
    let block_col_size: usize = dimension.pow(order as u32);
    let block_row: Vec<f64> = vec![1.0; block_col_size];
    let block_values: Vec<Vec<f64>> = vec![block_row; dimension];
    let data: Vec<Vec<Vec<f64>>> = vec![block_values; total_blocks];

    // Calculate block shape and matrix shape
    let total_column_blocks: usize = number_of_columns(num_node, order);
    let block_size: (usize, usize) = (dimension, block_col_size);
    let shape: (usize, usize) = (num_node * dimension, total_column_blocks * block_col_size);
    
    BsrMatrix::from_blocks(data, indices, indptr, shape, block_size)
}

/// Calculates the number of columns in the block matrix
fn number_of_columns(n_nodes: usize, order: usize) -> usize {
    match order {
        1 => n_nodes,
        2 => n_nodes * (n_nodes + 1) / 2,
        3 => n_nodes * (n_nodes + 1) * (n_nodes + 2) / 6,
        _ => binomial(n_nodes + order - 1, order),
    }
}

/// Calculates the block size for an element
fn element_block_size(n_nodes: usize, order: usize) -> usize {
    n_nodes * number_of_columns(n_nodes, order)
}

/// Locates the block column index for given nodes
fn locate_block_column(num_nodes: usize, nodes: &[&usize]) -> usize {
    let order = nodes.len();
    
    match order {
        1 => *nodes[0],
        2 => (2 * num_nodes - nodes[0] + 1) * nodes[0] / 2 + (nodes[1] - nodes[0]),
        3 => {
            let before_i: usize = tetrahedral(num_nodes) - tetrahedral(num_nodes - nodes[0]);
            let before_j: usize = (nodes[1] - nodes[0]) * (2 * num_nodes - nodes[0] - nodes[1] + 1) / 2;
            let position_ijk: usize = nodes[2] - nodes[1];
            before_i + before_j + position_ijk
        },
        _ => {
            let first: usize = *nodes[0];
            if *nodes[0] == num_nodes - 1 {
                binomial(num_nodes + order - 1, order) - 1
            } else {
                let sum: usize = (0..*nodes[0])
                    .map(|i| binomial(num_nodes - i + order - 2, order - 1)).sum();
                let remaining_nodes: Vec<usize> = nodes[1..].iter().map(|&&x| x - first).collect();
                sum + locate_block_column(num_nodes - first, &remaining_nodes.iter().collect::<Vec<_>>())
            }
        }
    }
}

/// Calculates the tetrahedral number
fn tetrahedral(m: usize) -> usize {
    m * (m + 1) * (m + 2) / 6
}

/// Given a BSR matrix and a list of block positions (i,j), return the corresponding data indices
///
/// # Arguments
/// * `indptr` - indptr of the stiffness matrix in BSR format
/// * `indices` - indices of the stiffness matrix in BSR format
/// * `list_of_indices` - List of block positions (i,j) to find in the data array
///
/// # Returns
/// Vector of indices in the data array corresponding to the blocks (i,j).
/// Returns None for blocks not found in the sparse structure.
pub fn get_data_indices_from_block_positions_binary_search(
    indptr: &Vec<usize>,
    indices: &Vec<Vec<usize>>,
    list_of_indices: &[(usize, usize)],
) -> Vec<Option<usize>> {
    
    list_of_indices
        .iter()
        .map(|&(i, j)| {
            // Check if row i exists in the matrix
            if i >= indptr.len() - 1 {
                return None;
            }
            
            let row_start: usize = indptr[i];
            let row_end: usize = indptr[i + 1];
            
            // Binary search within this row's columns
            let search_result: Result<usize, usize> = indices[row_start..row_end].binary_search(&vec![j]);
            
            match search_result {
                Ok(pos) => Some(row_start + pos),
                Err(_) => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_stiffness_matrix() {
        let num_node: usize = 3;
        let elements: Vec<Vec<usize>> = vec![vec![0, 1], vec![1, 2]]; // list of nodes of each element
        let dimension: usize = 2;
        let order: usize = 2;

        let matrix: BsrMatrix<f64> = initialize_nonlinear_stiffness_matrix(num_node, &elements, dimension, order).unwrap();

        //println!("to_dense ->\n{:?}", matrix.to_dense());

        assert_eq!(matrix.shape(), (num_node * dimension, binomial(num_node + order - 1, order) * dimension.pow(order as u32)));
        assert_eq!(matrix.block_size(), (dimension, dimension.pow(order as u32)));

        //println!("indptr = {:?}", matrix.indptr());
        //println!("indices = {:?}", matrix.indices());
        //println!("data = {:?}", matrix.data_mut());
    }

    /*
    #[test]
    fn test_get_data_indices() {
        let num_node = 3;
        let elements = vec![vec![0, 1], vec![1, 2]];
        let dimension = 2;

        let matrix = initialize_stiffness_matrix(num_node, &elements, dimension);

        let queries = vec![(0, 0), (0, 1), (1, 1), (2, 2)];
        let indices = get_data_indices_from_block_positions_binary_search(&matrix, &queries);

        // The exact indices depend on the ordering, but we can check some properties
        assert_eq!(indices.len(), queries.len());
        assert!(indices[0].is_some()); // (0,0) should exist
        assert!(indices[1].is_some()); // (0,1) should exist
        assert!(indices[2].is_some()); // (1,1) should exist
        assert!(indices[3].is_none()); // (2,2) shouldn't exist in this case
    }*/
}