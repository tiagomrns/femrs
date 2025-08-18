use nalgebra::{DVector, DMatrix, Vector2, Vector3};

use crate::fem::elements::quadrature::quadrature_rules::QuadratureRule;

struct Element;

fn integrate_elements(elements: Vec<&Element>, quadrature_rule: &QuadratureRule) {
    for element in elements.iter() {


        for point, weight in quadrature_rule {
            jacobian_shape_functions: DMatrix<f64> = element.evaluate_jacobian_of_shape_functions(point)

            position_jacobian = compute_position_jacobian(all_nodal_coords, element.node_ids, jacobian_shape_functions)

            let (determinant, adjugate)  = determinant_and_adjugate_expansions(position_jacobian);
            
            incremement_mass_matrix(
                element.evaluate_shape_functions(point),
                element.node_ids,
                &determinant * weight,
            )

            incremement_stiffness_matrices_all_force_orders(
                jacobian_shape_functions,
                element.node_ids,
                adjugate,
                determinant,
                weight,
            )

        }


    }
}