// Copyright 2016 Martin Ankerl.
// Licensed under the Apache License, Version 2.0 or the MIT license.

//! Differential Evolution optimizer for rust.

pub mod common;
pub mod traits;
pub mod sde;
pub mod shade;

pub use common::{BoundaryHandling, DonorStrategy};
pub use traits::DifferentialEvolution;
pub use sde::{Sde, SdeSettings};
pub use shade::{Shade, ShadeSettings};

/// Convenience function to create a fully configured self adaptive
/// differential evolution population.
pub fn self_adaptive_de<F, C>(min_max_pos: Vec<(f32, f32)>,
                              cost_function: F)
                              -> Sde<F, rand::rngs::SmallRng, C>
    where F: Fn(&[f32]) -> C + Sync + Send,
          C: PartialOrd + Clone + Send + Sync
{
    Sde::new(SdeSettings::default(min_max_pos, cost_function))
}

/// Convenience function to create a fully configured SHADE population.
pub fn shade<F, C>(min_max_pos: Vec<(f32, f32)>,
                   cost_function: F)
                   -> Shade<F, rand::rngs::SmallRng, C>
    where F: Fn(&[f32]) -> C + Sync + Send,
          C: PartialOrd + Clone + Send + Sync + Copy + num_traits::Float
{
    Shade::new(ShadeSettings::default(min_max_pos, cost_function))
}
