struct Element<ShapeFunctions: NodalBasedShapeFunctions> {
    node_ids: Vec<u32>,
    shape_functions: ShapeFunctions
}

impl<ShapeFunctions: NodalBasedShapeFunctions> Element<ShapeFunctions> {
    fn new(node_ids: Vec<u32>, shape_functions: ShapeFunctions) -> Self {
        Self { node_ids, shape_functions }
    }
}