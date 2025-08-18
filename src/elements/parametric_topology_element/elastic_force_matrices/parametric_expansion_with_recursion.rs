#[derive(Debug, Clone)]
pub struct RecursionConfig {
    //max_degree: u8,
    max_force_order: u8,
    //dimension: u8,
    //morphing_degree: u8,
    max_degree_per_force_order: Vec<u8>,
}

impl RecursionConfig {
    pub fn new(
        max_degree: u8,
        max_force_order: u8,
        dimension: u8,
        morphing_degree: u8,
    ) -> Self {
        let max_degree_per_force_order =
            calculate_max_degrees_for_all_force_orders(max_degree, max_force_order, dimension, morphing_degree);
        
        Self {
            //max_degree,
            max_force_order,
            //dimension,
            //morphing_degree,
            max_degree_per_force_order,
        }
    }
}

/// Calculates the maximum degrees for all force orders based on given parameters
pub fn calculate_max_degrees_for_all_force_orders(
    max_degree: u8,
    max_force_order: u8,
    dimension: u8,
    morphing_degree: u8,
) -> Vec<u8> {
    let base_degree = (dimension - 1) * morphing_degree;
    (0..=max_force_order)
        .map(|n| std::cmp::min(base_degree * (n + 1), max_degree))
        .collect()
}

/// Performs recursive construction of force orders
pub fn recursive_construction(
    a_prev: Vec<f64>,
    a0: Vec<Vec<f64>>,
    config: &RecursionConfig,
    force_order: u8,
) {
    /*
    Args:
        a_prev: array of sum{(dNi1dx @ adj(J)|o1) @ ... @ (dNindx @ adj(J)|on)}
            - Fixed sequence of nodes i1,...,in
            - n = force_order-1
            - the array is indexed by the degree o w.r.t. to the parameter
        a0: array of dNidx @ adj(J)|o
            - first index is the node i of the shapefunction
            - second index is the degree o w.r.t. to the parameter
        config: Configuration of the recursion tree
        force_order: Force order of current node w.r.t the displacement
     */

    if force_order >= config.max_force_order {
        _process_leaf_force_order(&a_prev, &a0, force_order, &config.max_degree_per_force_order);
        return;
    }

    for (node_count, a0_node) in a0.iter().enumerate() {

        let a_current = 
        _compute_current_force_order(&a_prev, a0_node, force_order, &config.max_degree_per_force_order);
        
        let remaining_a0 = a0[node_count..].to_vec();

        recursive_construction(
            a_current.clone(),
            remaining_a0,
            config,
            force_order + 1,
        );
        
        _postprocess_force_order(&a_current, force_order, config.max_degree_per_force_order[force_order as usize]);
    }
}

fn _compute_current_force_order(
    a_prev: &[f64],
    a0_node: &[f64],  // Now takes a slice of f64
    force_order: u8,
    max_degree_per_force_order: &Vec<u8>,
) -> Vec<f64> {
    let degrees = 0..=max_degree_per_force_order[force_order as usize];
    degrees
        .map(|degree| {
            _compute_current_force_order_and_degree(a_prev, a0_node, degree, force_order, max_degree_per_force_order)
        })
        .collect()
}

// Branch Node

fn _postprocess_force_order(a_current: &[f64], force_order: u8, max_degree: u8) {
    for degree in 0..=max_degree {
        postprocess(a_current[degree as usize], force_order, degree);
    }
}

fn postprocess(a_current: f64, force_order: u8, degree: u8) {
    println!(
        "Branch-result for force_order={} and degree={}: \t Final result: A_current={:?} \t ",
        force_order, degree, a_current
    );
}

// Leaf Node

fn _process_leaf_force_order(
    a_prev: &[f64],
    a0: &[Vec<f64>],  // Changed to slice of Vec<f64>
    force_order: u8,
    max_degree_per_force_order: &Vec<u8>,
) {
    for a0_node in a0.iter() {
        for degree in 0..=max_degree_per_force_order[force_order as usize] {
            _compute_and_postprocess_leaf(a_prev, &a0_node, force_order, degree, max_degree_per_force_order);
        }
    }
}

fn _compute_and_postprocess_leaf(
    a_prev: &[f64], 
    a0_node: &[f64], 
    force_order: u8, 
    degree: u8, 
    max_degree_per_force_order: &Vec<u8>
) {

    let a_current = 
    _compute_current_force_order_and_degree(a_prev, a0_node, degree, force_order, max_degree_per_force_order);

    println!(
        "Leaf-result for force_order={} and degree={}: \t Final result: A_current={:?} \t ",
        force_order, degree, a_current
    );
    // Placeholder for actual computation and postprocessing
    // In real code you'd do something like:
    // let result = compute(a_prev, a0_element, degree);
    // postprocess(&result, force_order, degree);
}

// Compute Tensors

fn _compute_current_force_order_and_degree(
    a_prev: &[f64],
    a0_node: &[f64],
    total_degree: u8,
    force_order: u8,
    max_degree_per_force_order: &Vec<u8>,
) -> f64 {
    let start = (0.max(total_degree as i8 - max_degree_per_force_order[0] as i8)) as u8;
    let end = max_degree_per_force_order[force_order as usize - 1].min(total_degree);
    
    (start..=end)
        .map(|degree| {
            // Simplified tensor product simulation - in real code you'd use actual tensor operations
            a_prev[degree as usize] * a0_node[(total_degree - degree) as usize]
        })
        .sum()
}


#[cfg(test)]
mod tests {
    use super::*;

    //#[test]
    fn test_config_creation() {
        let config = RecursionConfig::new(
            3, 
            4, 
            2, 
            1
        );

        println!("{:?}", config);
    }

    //#[test]
    fn test_calculate_max_degrees() {
        // Test case from before
        assert_eq!(
            calculate_max_degrees_for_all_force_orders(10, 3, 3, 1),
            vec![2, 4, 6, 8]
        );

        // Different parameters
        assert_eq!(
            calculate_max_degrees_for_all_force_orders(8, 2, 2, 2),
            vec![2, 4, 6]
        );

        // Edge case where max_degree limits the degrees
        assert_eq!(
            calculate_max_degrees_for_all_force_orders(5, 4, 3, 1),
            vec![2, 4, 5, 5, 5]
        );
    }

    //#[test]
    fn test_recursive_construction_basic() {

        let config = RecursionConfig::new(
            3, 
            4, 
            2, 
            1
        );

        
        let a0 = vec![
            vec![0.5, 1.5, 2.5],  // Node 0, degrees 0, 1, 2
            vec![1.0, 2.0, 3.0],  // Node 1, degrees 0, 1, 2
            vec![1.5, 2.5, 3.5],  // Node 2, degrees 0, 1, 2
        ];

        let a_prev = a0[0].clone(); //vec![1.0, 2.0, 3.0, 4.0, 5.0]; // degrees 0, 1, 2, 3, 4

        // Just verify it runs without panicking
        recursive_construction(a_prev, a0, &config, 1);
    }

    //#[test]
    fn test_compute_current_force_order() {
        let config = RecursionConfig::new(
            4, 
            2, 
            3, 
            1
        );

        let a_prev = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // degrees 0, 1, 2, 3, 4
        let a0_node = vec![0.5, 1.5, 2.5]; // degree 0, 1, 2

        let result = _compute_current_force_order(&a_prev, &a0_node, 2, &config.max_degree_per_force_order);
        
        // Expected calculations:
        // degree 0: a_prev[0] * a0[0] = 1.0 * 0.5 = 0.5
        // degree 1: a_prev[0] * a0[1] + a_prev[1] * a0[0] = 1.0*1.5 + 2.0*0.5 = 1.5 + 1.0 = 2.5
        // degree 2: a_prev[0] * a0[2] + a_prev[1] * a0[1] + a_prev[2] * a0[0] = 1.0*2.5 + 2.0*1.5 + 3.0*0.5 = 2.5 + 3.0 + 1.5 = 7.0
        // degree 3: a_prev[1] * a0[2] + a_prev[2] * a0[1] + a_prev[3] * a0[0] = 2.0*2.5 + 3.0*1.5 + 4.0*0.5 = 5.0 + 4.5 + 2.0 = 11.5
        // degree 4: a_prev[2] * a0[2] + a_prev[3] * a0[1] + a_prev[4] * a0[0] = 3.0*2.5 + 4.0*1.5 + 5.0*0.5 = 7.5 + 6.0 + 2.5 = 16.0
        print!("Result: {:?}", result);
        assert_eq!(result.len(), 5); // degrees 0..=2 (max_degree_per_force_order[0] = 2)
        assert_eq!(result[0], 0.5);
        assert_eq!(result[1], 2.5);
        assert_eq!(result[2], 7.0);
    }

    //#[test]
    fn test_process_leaf_force_order() {
        let config = RecursionConfig::new(2, 1, 3, 1);
        let a_prev = vec![1.0, 2.0];
        let a0 = vec![
            vec![0.5, 1.5],  // Node 0 (won't be processed)
            vec![1.0, 2.0],  // Node 1
            vec![1.5, 2.5],  // Node 2
        ];

        // This should process nodes 1 and 2 (node_start=1, node_end=2)
        _process_leaf_force_order(&a_prev, &a0, 1, &config.max_degree_per_force_order);
        
        // The function just prints output, so we're just testing it doesn't panic
    }

    //#[test]
    fn test_compute_and_postprocess() {

        let config = RecursionConfig::new(
            4, 
            4, 
            3, 
            1
        );

        let a_prev = vec![1.0, 2.0];
        let a0_element = vec![0.5, 1.5];
        
        // Just test it runs without panicking
        _compute_and_postprocess_leaf(&a_prev, &a0_element, 1, 1, &config.max_degree_per_force_order);
    }
}