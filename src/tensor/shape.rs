#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Box<[usize]>,
}

impl Shape {
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
        Self {
            dims: dims.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims().len()
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_metadata() {
        let shape = Shape::new([2, 3, 4]);

        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.numel(), 24);
    }
}
