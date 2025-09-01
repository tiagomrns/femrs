#![allow(dead_code)]

pub mod assemble {
    pub mod assembly;
    pub mod write_data;
}

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

pub mod mesh {
    pub mod locate_nodes_o_log_n;
    pub mod node_coordinates_ndarray;
    //pub mod hypernode;
}


fn main() { 
    println!("Hello, world!");
}