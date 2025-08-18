use nalgebra::{DVector, DMatrix, Vector2, Vector3};

fn increment_mass_matrix(mass_matrix_data: &DVector<f64>, shape_functions: DVector<f64>, node_ids: &[u8], weight_times_determinant: &f64) {
    // Implementation for incrementing the mass matrix
    let data = shape_functions * shape_functions * weight_times_determinant;


}