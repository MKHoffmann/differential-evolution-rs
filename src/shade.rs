use crate::common::{clamp_x, wrap_x, BoundaryHandling, DonorStrategy, Individual};
use crate::traits::DifferentialEvolution;
use rand_distr::{Distribution, Uniform, Normal, Cauchy};
use rand::SeedableRng;
use rayon::prelude::*;
use ringbuffer::{AllocRingBuffer, RingBuffer};

pub struct ShadeSettings<F, R> {
    pub min_max_pos: Vec<(f32, f32)>,
    pub donor_strategy: DonorStrategy,
    pub boundary_handling: BoundaryHandling,
    pub num_diffs: usize,
    pub memory_size: usize,
    pub cr_min_max: (f32, f32),
    pub cr_init: f32,
    pub f_min_max: (f32, f32),
    pub f_init: f32,
    pub pop_size: usize,
    pub rng: R,
    pub cost_function: F,
}

impl<F> ShadeSettings<F, rand::rngs::SmallRng> {
    pub fn default(min_max_pos: Vec<(f32, f32)>, cost_function: F) -> Self {
        Self {
            min_max_pos,
            num_diffs: 1,
            boundary_handling: BoundaryHandling::Clamp,
            donor_strategy: DonorStrategy::Rand,
            memory_size: 20,
            cr_min_max: (0.0, 1.0),
            cr_init: 0.5,
            f_min_max: (-10.0, 10.0),
            f_init: 0.5,
            pop_size: 100,
            rng: rand::rngs::SmallRng::from_os_rng(),
            cost_function,
        }
    }
}

fn sample_cr<R>(cr_mem: &AllocRingBuffer<f32>, rng: &mut R) -> f32 
where 
    R: rand::Rng
{
    let idx = rng.random_range(0..cr_mem.len());
    Normal::new(cr_mem[idx], 0.1).unwrap().sample(rng).max(0.0).min(1.0)            
}

fn sample_f<R>(f_mem: &AllocRingBuffer<f32>, rng:&mut R, f_min_max: &(f32, f32)) -> f32  
where 
    R: rand::Rng
{
    let idx = rng.random_range(0..f_mem.len());
    let dist = Cauchy::new(f_mem[idx], 0.1).unwrap();
    dist.sample(rng).min(f_min_max.1).max(f_min_max.0)
}

pub struct Shade<F, R, C> {
    // Mapped: trial -> curr (lib.rs)
    trial: Vec<Individual<C>>,
    // Mapped: pop -> best (lib.rs)
    pop: Vec<Individual<C>>,
    
    settings: ShadeSettings<F, R>,
    
    best_idx: Option<usize>,
    best_cost: Option<C>,
    
    archive: AllocRingBuffer<Individual<C>>,
    between_dim: Uniform<usize>,
    
    // Memory
    success: Vec<bool>,
    f_mem: AllocRingBuffer<f32>,
    cr_mem: AllocRingBuffer<f32>,
    
    num_cost_evaluations: usize,
}

impl<F, R, C> Shade<F, R, C>
where
    F: Fn(&[f32]) -> C + Sync + Send,
    R: rand::Rng,
    C: PartialOrd + Clone + Send + Sync + Copy + num_traits::Float,
{
    pub fn new(s: ShadeSettings<F, R>) -> Self {
        let dim = s.min_max_pos.len();
        let dummy = Individual::new(dim);
        
        let mut shade = Self {
            trial: vec![dummy.clone(); s.pop_size],
            pop: vec![dummy; s.pop_size],
            best_idx: None,
            best_cost: None,
            num_cost_evaluations: 0,
            archive: AllocRingBuffer::new(s.pop_size),
            between_dim: Uniform::try_from(0..dim).unwrap(),
            success: vec![false; s.pop_size],
            cr_mem: AllocRingBuffer::from(vec![s.cr_init; s.memory_size]),
            f_mem: AllocRingBuffer::from(vec![s.f_init; s.memory_size]),
            settings: s,
        };

        // Initialize trial (curr in lib.rs) with random positions
        for d in 0..dim {
            let between_min_max = Uniform::try_from(shade.settings.min_max_pos[d].0..=shade.settings.min_max_pos[d].1).unwrap();
            for ind in &mut shade.trial {
                ind.cr = sample_cr(&shade.cr_mem, &mut shade.settings.rng);
                ind.f = sample_f(&shade.f_mem, &mut shade.settings.rng, &shade.settings.f_min_max);
                ind.pos[d] = between_min_max.sample(&mut shade.settings.rng);
            }
        }
        
        shade.pop = shade.trial.clone();
        
        shade
    }
    
    fn attach_new_cr(&mut self) {
        if self.success.iter().fold(0, |acc, s| if *s { acc + 1 } else { acc }) == 0 {
            self.cr_mem.enqueue(self.cr_mem.back().unwrap() * 0.9);
            return;
        }
        let (s_cr_scaled_sum, w_sum) = self
            .success
            .iter()
            .zip(&self.trial) // curr
            .zip(&self.pop)   // best
            .fold((0.0f32, 0.0f32), |(mut sum_cr, mut sum_w), ((s, curr), best)| {
                if *s {
                    let diff = (curr.cost.unwrap() - best.cost.unwrap()).to_f32().unwrap().abs();
                    sum_w    += diff;
                    sum_cr   += curr.cr * diff;
                }
                (sum_cr, sum_w)
            });
        if w_sum == 0.0 {
            return;
        }
        let new_val = s_cr_scaled_sum / w_sum;
        self.cr_mem.enqueue(new_val);
    }

    fn attach_new_f(&mut self) {
        if self.success.iter().fold(0, |acc, s| if *s { acc + 1 } else { acc }) == 0 {
            self.f_mem.enqueue(self.f_mem.back().unwrap() * 0.9);
            return;
        }
        
        let (s_f_scaled_sum, s_f_scaled_sum_sq) = self
            .success
            .iter()
            .zip(&self.trial) // curr
            .zip(&self.pop)   // best
            .fold((0.0f32, 0.0f32), |(mut sum, mut sum_sq), ((s, curr), best)| {
                if *s {
                    let diff = (curr.cost.unwrap() - best.cost.unwrap()).to_f32().unwrap().abs();
                    let f = curr.f;
                    sum    += f * diff;
                    sum_sq += f * f * diff;
                }
                (sum, sum_sq)
            });

        if s_f_scaled_sum == 0.0 {
            return;
        }
        let new_val = s_f_scaled_sum_sq / (s_f_scaled_sum);
        self.f_mem.enqueue(new_val);
    }
    
    pub fn num_cost_evaluations(&self) -> usize {
        self.num_cost_evaluations
    }
}

impl<F, R, C> DifferentialEvolution<C> for Shade<F, R, C>
where
    F: Fn(&[f32]) -> C + Sync + Send,
    R: rand::Rng,
    C: PartialOrd + Clone + Send + Sync + Copy + num_traits::Float,
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
        // Maps to update_positions in lib.rs
        let rng = &mut self.settings.rng;
        let indices_population = crate::common::get_random_indices(self.settings.pop_size, self.settings.num_diffs, rng);
        
        let curr_read = self.trial.clone(); // Snapshot for reading
        let pop_read = self.pop.clone(); // Snapshot of best/pop
        
        for (i, indices) in indices_population.iter().enumerate() {
             self.trial[i].f = sample_f(&self.f_mem, rng, &self.settings.f_min_max);
             self.trial[i].cr = sample_cr(&self.cr_mem, rng);
             
             let member = &curr_read[i];
             let forced_mutation_dim = self.between_dim.sample(rng);
             
             let mut donor = match self.settings.donor_strategy {
                DonorStrategy::Rand => pop_read[indices[0]].pos.clone(),
                DonorStrategy::Best => pop_read[indices[0]].pos.clone(),
                DonorStrategy::CurrentToBest => {
                    let factor = rng.random::<f32>();
                    let best_member = pop_read.last().unwrap();
                    member.pos.iter().zip(&best_member.pos).map(|(m, b)| m + factor * (b - m)).collect()
                },
                DonorStrategy::CurrentToRand => {
                    let factor = rng.random::<f32>();
                    member.pos.iter().zip(&curr_read[indices[0]].pos).map(|(m, r)| m + factor * (r - m)).collect()
                },
                DonorStrategy::RandToBest => {
                    let factor = rng.random::<f32>();
                    let best_member = pop_read.last().unwrap();
                    curr_read[indices[0]].pos.iter().zip(&best_member.pos).map(|(r, b)| r + factor * (b - r)).collect()
                }
             };
             
             let mut take_from_archive = vec![false; self.settings.num_diffs];
             if self.archive.is_full() {
                 for j in 0..self.settings.num_diffs {
                     if rng.random::<f32>() < 0.5 {
                         take_from_archive[j] = true;
                     }
                 }
             }
             
             let random_values: Vec<f32> = (0..donor.len()).map(|_| rng.random::<f32>()).collect();
             
             donor.iter_mut().enumerate().zip(random_values.iter()).for_each(|((i_value, v), &rand_val)| {
                 if i_value == forced_mutation_dim || rand_val < self.trial[i].cr {
                     for j_diff in 0..self.settings.num_diffs {
                         let xr1 = &curr_read[indices[1 + 2*j_diff]];
                         let xr2 = if take_from_archive[j_diff] {
                             &self.archive[indices[2 + 2*j_diff]]
                         } else {
                             &curr_read[indices[2 + 2*j_diff]]
                         };
                         *v = *v + self.trial[i].f * (xr1.pos[i_value] - xr2.pos[i_value]);
                     }
                     match self.settings.boundary_handling {
                         BoundaryHandling::Clamp => clamp_x(v, &self.settings.min_max_pos[i_value]),
                         BoundaryHandling::Wrap => wrap_x(v, &self.settings.min_max_pos[i_value]),
                         BoundaryHandling::Identity => {}
                     }
                 }
             });
             
             self.trial[i].pos = donor;
             self.trial[i].cost = None;
        }
    }

    fn evaluate_trial_vectors(&mut self) {
        let cost_function = &self.settings.cost_function;
        self.trial.par_iter_mut().for_each(|ind| {
            let cost = (cost_function)(&ind.pos);
            ind.cost = Some(cost);
        });
        self.num_cost_evaluations += self.trial.len();
    }

    fn selection(&mut self) {
        // Maps to update_best in lib.rs
        for i in 0..self.trial.len() {
            let curr = &mut self.trial[i];
            let best = &mut self.pop[i];

            if best.cost.is_none() {
                std::mem::swap(curr, best);
            } else if curr.cost.unwrap() <= best.cost.unwrap() {
                self.archive.enqueue(best.clone());
                std::mem::swap(curr, best);
                self.success[i] = true;
            }
            
            // Update global best cache
            let cost = best.cost.unwrap();
            if self.best_cost.is_none() || cost < self.best_cost.unwrap() {
                self.best_cost = Some(cost);
                self.best_idx = Some(i);
            }
        }
    }

    fn on_generation_end(&mut self) {
        // Maps to update_memory in lib.rs
        self.attach_new_cr();
        self.attach_new_f();
        self.success.iter_mut().for_each(|s| *s = false);
    }
}