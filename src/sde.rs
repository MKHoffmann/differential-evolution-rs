use crate::common::{clamp_x, wrap_x, BoundaryHandling, DonorStrategy, Individual};
use crate::traits::DifferentialEvolution;
use rand::distr::{Distribution, Uniform};
use rand::SeedableRng;
use rayon::prelude::*;

pub struct SdeSettings<F, R> {
    pub min_max_pos: Vec<(f32, f32)>,
    pub donor_strategy: DonorStrategy,
    pub boundary_handling: BoundaryHandling,
    pub num_diffs: usize,
    pub cr_min_max: (f32, f32),
    pub cr_change_probability: f32,
    pub f_min_max: (f32, f32),
    pub f_change_probability: f32,
    pub pop_size: usize,
    pub rng: R,
    pub cost_function: F,
}

impl<F> SdeSettings<F, rand::rngs::SmallRng> {
    pub fn default(min_max_pos: Vec<(f32, f32)>, cost_function: F) -> Self {
        Self {
            min_max_pos,
            donor_strategy: DonorStrategy::Rand,
            boundary_handling: BoundaryHandling::Clamp,
            num_diffs: 1,
            cr_min_max: (0.0, 1.0),
            cr_change_probability: 0.1,
            f_min_max: (0.1, 1.0),
            f_change_probability: 0.1,
            pop_size: 100,
            rng: rand::rngs::SmallRng::from_rng(&mut rand::rng()),
            cost_function,
        }
    }
}

pub struct Sde<F, R, C> {
    pop: Vec<Individual<C>>,
    trial: Vec<Individual<C>>,
    settings: SdeSettings<F, R>,
    best_idx: Option<usize>,
    best_cost: Option<C>,
    
    // Helpers
    dim: usize,
    between_popsize: Uniform<usize>,
    between_dim: Uniform<usize>,
    between_cr: Uniform<f32>,
    between_f: Uniform<f32>,
}

impl<F, R, C> Sde<F, R, C>
where
    F: Fn(&[f32]) -> C + Sync + Send,
    R: rand::Rng,
    C: PartialOrd + Clone + Send + Sync,
{
    pub fn new(s: SdeSettings<F, R>) -> Self {
        let dim = s.min_max_pos.len();
        let dummy = Individual::new(dim);
        
        let mut sde = Self {
            pop: vec![dummy.clone(); s.pop_size],
            trial: vec![dummy; s.pop_size],
            best_idx: None,
            best_cost: None,
            dim,
            between_popsize: Uniform::new(0, s.pop_size).unwrap(),
            between_dim: Uniform::new(0, dim).unwrap(),
            between_cr: Uniform::new(s.cr_min_max.0, s.cr_min_max.1).unwrap(),
            between_f: Uniform::new(s.f_min_max.0, s.f_min_max.1).unwrap(),
            settings: s,
        };

        // Initialize population
        for ind in &mut sde.pop {
            ind.cr = sde.between_cr.sample(&mut sde.settings.rng);
            ind.f = sde.between_f.sample(&mut sde.settings.rng);
            for d in 0..dim {
                let range = Uniform::new(sde.settings.min_max_pos[d].0, sde.settings.min_max_pos[d].1).unwrap();
                ind.pos[d] = range.sample(&mut sde.settings.rng);
            }
        }
        
        // Initial evaluation
        let cost_function = &sde.settings.cost_function;
        sde.pop.par_iter_mut().for_each(|ind| {
            ind.cost = Some((cost_function)(&ind.pos));
        });
        
        // Find initial best
        for (i, ind) in sde.pop.iter().enumerate() {
            let cost = ind.cost.as_ref().unwrap();
            if sde.best_cost.is_none() || cost < sde.best_cost.as_ref().unwrap() {
                sde.best_cost = Some(cost.clone());
                sde.best_idx = Some(i);
            }
        }

        sde
    }
}

impl<F, R, C> DifferentialEvolution<C> for Sde<F, R, C>
where
    F: Fn(&[f32]) -> C + Sync + Send,
    R: rand::Rng,
    C: PartialOrd + Clone + Send + Sync,
{
    fn pop(&self) -> &[Individual<C>] { &self.pop }
    fn pop_mut(&mut self) -> &mut Vec<Individual<C>> { &mut self.pop }
    fn trial(&self) -> &[Individual<C>] { &self.trial }
    fn trial_mut(&mut self) -> &mut Vec<Individual<C>> { &mut self.trial }
    fn best_idx(&self) -> Option<usize> { self.best_idx }
    fn set_best_idx(&mut self, idx: usize) { self.best_idx = Some(idx); }
    fn best_cost(&self) -> Option<&C> { self.best_cost.as_ref() }
    fn set_best_cost(&mut self, cost: C) { self.best_cost = Some(cost); }

    fn generate_trial_vectors(&mut self) {
        let rng = &mut self.settings.rng;
        let dim = self.dim;
        let num_diffs = self.settings.num_diffs;

        for i in 0..self.pop.len() {
            // 1. Update Parameters (SDE specific)
            // We update parameters on the *trial* vector based on the *parent* vector
            // But SDE usually updates the parent's parameters in place or creates new ones for the trial.
            // Here we copy parent to trial first, then mutate trial params.
            self.trial[i].cr = self.pop[i].cr;
            self.trial[i].f = self.pop[i].f;

            if rng.random::<f32>() < self.settings.cr_change_probability {
                self.trial[i].cr = self.between_cr.sample(rng);
            }
            if rng.random::<f32>() < self.settings.f_change_probability {
                self.trial[i].f = self.between_f.sample(rng);
            }

            let f = self.trial[i].f;
            let cr = self.trial[i].cr;

            // 2. Select Indices
            let mut indices = Vec::with_capacity(1 + 2 * num_diffs);
            while indices.len() < 1 + 2 * num_diffs {
                let idx = self.between_popsize.sample(rng);
                if idx != i && !indices.contains(&idx) {
                    indices.push(idx);
                }
            }

            let forced_mutation_dim = self.between_dim.sample(rng);
            let best_global_pos = &self.pop[self.best_idx.unwrap_or(indices[0])].pos;
            let mut mutant = vec![0.0; dim];

            // 3. Mutation
            match self.settings.donor_strategy {
                DonorStrategy::Rand => {
                    let r1 = indices[0];
                    for d in 0..dim {
                        let mut val = self.pop[r1].pos[d];
                        for k in 0..num_diffs {
                            val += f * (self.pop[indices[1+2*k]].pos[d] - self.pop[indices[2+2*k]].pos[d]);
                        }
                        mutant[d] = val;
                    }
                },
                DonorStrategy::Best => {
                    for d in 0..dim {
                        let mut val = best_global_pos[d];
                        for k in 0..num_diffs {
                            val += f * (self.pop[indices[0+2*k]].pos[d] - self.pop[indices[1+2*k]].pos[d]);
                        }
                        mutant[d] = val;
                    }
                },
                DonorStrategy::CurrentToBest => {
                    for d in 0..dim {
                        let mut val = self.pop[i].pos[d] + f * (best_global_pos[d] - self.pop[i].pos[d]);
                        for k in 0..num_diffs {
                            val += f * (self.pop[indices[0+2*k]].pos[d] - self.pop[indices[1+2*k]].pos[d]);
                        }
                        mutant[d] = val;
                    }
                },
                DonorStrategy::RandToBest => {
                    let r1 = indices[0];
                    for d in 0..dim {
                        let mut val = self.pop[r1].pos[d] + f * (best_global_pos[d] - self.pop[r1].pos[d]);
                        for k in 0..num_diffs {
                            val += f * (self.pop[indices[1+2*k]].pos[d] - self.pop[indices[2+2*k]].pos[d]);
                        }
                        mutant[d] = val;
                    }
                },
                DonorStrategy::CurrentToRand => {
                    let r1 = indices[0];
                    for d in 0..dim {
                        let mut val = self.pop[i].pos[d] + f * (self.pop[r1].pos[d] - self.pop[i].pos[d]);
                        for k in 0..num_diffs {
                            val += f * (self.pop[indices[1+2*k]].pos[d] - self.pop[indices[2+2*k]].pos[d]);
                        }
                        mutant[d] = val;
                    }
                },
            }

            // 4. Crossover & Boundary
            for d in 0..dim {
                if d == forced_mutation_dim || rng.random::<f32>() < cr {
                    self.trial[i].pos[d] = mutant[d];
                } else {
                    self.trial[i].pos[d] = self.pop[i].pos[d];
                }
                match self.settings.boundary_handling {
                    BoundaryHandling::Clamp => clamp_x(&mut self.trial[i].pos[d], &self.settings.min_max_pos[d]),
                    BoundaryHandling::Wrap => wrap_x(&mut self.trial[i].pos[d], &self.settings.min_max_pos[d]),
                    BoundaryHandling::Identity => {},
                }
            }
            self.trial[i].cost = None;
        }
    }

    fn evaluate_trial_vectors(&mut self) {
        let cost_function = &self.settings.cost_function;
        self.trial.par_iter_mut().for_each(|ind| {
            let cost = (cost_function)(&ind.pos);
            ind.cost = Some(cost);
        });
    }
}
