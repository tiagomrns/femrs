use ndarray::Array2;

use crate::elements::quadrature::quadrature_rules::QuadratureRule;
use crate::elements::parametric_topology_element::position_jacobian::compute_position_jacobian;

trait Element {
    type Coordinates;
    const DIMENSION: u8;
    const NUMBER_OF_NODES: u8;
    fn evaluate_shape_functions(coords: &Self::Coordinates) -> [f64];
    fn evaluate_jacobian_of_shape_functions(coords: &Self::Coordinates) -> Array2<f64>;
}

fn integrate_elements<const DIM: usize, const LEN: usize, E: Element>(
    elements: Vec<&E>,
    quadrature_rule: &QuadratureRule<DIM, LEN>,
    all_nodal_coords: &[f64], // Assuming nodal coordinates are stored as a slice of f64
) {
    for element in elements.iter() {
        // Iterate over points and weights in the quadrature rule
        for (point, weight) in quadrature_rule.points.iter().zip(quadrature_rule.weights.iter()) {
            let jacobian_shape_functions: Array2<f64> = element.evaluate_jacobian_of_shape_functions(*point);

            let position_jacobian = compute_position_jacobian(
                all_nodal_coords,
                &element.node_ids,
                &jacobian_shape_functions,
            );

            let (determinant, adjugate) = determinant_and_adjugate_expansions(&position_jacobian);
            
            increment_mass_matrix(
                element.evaluate_shape_functions(*point),
                &element.node_ids,
                determinant * weight,
            );

            increment_stiffness_matrices_all_force_orders(
                &jacobian_shape_functions,
                &element.node_ids,
                &adjugate,
                determinant,
                *weight,
            );
        }
    }
}