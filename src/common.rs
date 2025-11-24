//! Common types and helper functions for DE variants.
use rand::seq::SliceRandom;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DonorStrategy {
    Rand,
    Best,
    RandToBest,
    CurrentToBest,
    CurrentToRand,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryHandling {
    Identity,
    Wrap,
    Clamp,
}

#[derive(Clone, Debug)]
pub struct Individual<C> {
    pub pos: Vec<f32>,
    pub cost: Option<C>,
    pub cr: f32,
    pub f: f32,
}

impl<C> Individual<C> {
    pub fn new(dim: usize) -> Self {
        Self {
            pos: vec![0.0; dim],
            cost: None,
            cr: 0.5,
            f: 0.5,
        }
    }
}

#[inline(always)]
pub fn clamp_x(x: &mut f32, bounds: &(f32, f32)) {
    *x = x.max(bounds.0).min(bounds.1)
}

#[inline(always)]
pub fn wrap_x(x: &mut f32, bounds: &(f32, f32)) {
    *x = ((*x - bounds.0).rem_euclid(bounds.1 - bounds.0)) + bounds.0
}

pub fn get_random_indices<R>(pop_size: usize, num_diffs: usize, rng: &mut R) -> Vec<Vec<usize>> 
where
    R: rand::Rng
{
    (0..pop_size).into_iter().map(
        |i| { 
            let mut v = (0..pop_size-1).collect::<Vec<usize>>();
            v.shuffle(rng);
            v[..(num_diffs * 2 + 1)].into_iter().map(
                |&idx| if idx < i { idx } else { idx + 1 }
            ).collect()
        }
    ).collect::<Vec<Vec<usize>>>()
}
