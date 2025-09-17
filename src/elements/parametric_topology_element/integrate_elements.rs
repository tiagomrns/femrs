use ndarray::Array2;

use crate::elements::quadrature::quadrature_rules::QuadratureRule;
use crate::elements::parametric_topology_element::position_jacobian::compute_position_jacobian;
use crate::elements::element_library::hypercube_elements::NodalBasedShapeFunctions;
use crate::elements::parametric_topology_element::determinant_and_adjugate::determinant_and_adjugate_expansions;

fn integrate_elements<const DIM: usize, const LEN: usize, Element>(
    elements: Vec<&Element>,
    quadrature_rule: &QuadratureRule<DIM, LEN>,
    all_nodal_coords: &Array2<f64>,
) where
    Element: NodalBasedShapeFunctions<Coordinates = [f64; DIM]>,
{
    for element in elements {
        // Iterate over points and weights in the quadrature rule 
        for (point, weight) in quadrature_rule.iter() {
            let jacobian_shape_functions: Array2<f64> = Element::evaluate_jacobian_of_shape_functions(point);

            let position_jacobian = compute_position_jacobian(
                all_nodal_coords,
                element.node_ids(),
                &jacobian_shape_functions,
            );

            let (determinant, adjugate) = determinant_and_adjugate_expansions(&position_jacobian);
            
            increment_mass_matrix(
                Element::evaluate_shape_functions(point),
                element.node_ids(),
                determinant * weight,
            );

            increment_stiffness_matrices_all_force_orders(
                &jacobian_shape_functions,
                element.node_ids(),
                &adjugate,
                determinant,
                *weight,
            );
        }
    }
}