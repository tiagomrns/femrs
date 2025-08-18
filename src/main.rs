#![allow(dead_code)]

pub mod assembly;

pub mod elements {
    pub mod parametric_topology_element {
        pub mod elastic_force_matrices {
            pub mod parametric_expansion_with_recursion;
        }
        pub mod determinant_and_adjugate;
        pub mod position_jacobian;
        //pub mod integrate_elements;
        //pub mod mass_matrices;
    }
    pub mod quadrature {
        pub mod quadrature_rules;
    }
    pub mod element_library {
        pub mod hypercube_elements;
    }
}

pub mod read_mesh {
    pub mod locate_nodes_o_log_n;
}


fn main() { 
    println!("Hello, world!");
}