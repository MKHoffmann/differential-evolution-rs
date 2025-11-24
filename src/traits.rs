use crate::common::Individual;
use ringbuffer::{AllocRingBuffer, RingBuffer};

pub trait DifferentialEvolution<C>
where
    C: PartialOrd + Clone + Send + Sync,
{
    // --- Abstract Methods (Must be implemented) ---

    /// Access to the current population
    fn pop(&self) -> &[Individual<C>];
    fn pop_mut(&mut self) -> &mut Vec<Individual<C>>;

    /// Access to the trial vectors (candidates for the next generation)
    fn trial(&self) -> &[Individual<C>];
    fn trial_mut(&mut self) -> &mut Vec<Individual<C>>;

    /// Access to global best
    fn best_idx(&self) -> Option<usize>;
    fn set_best_idx(&mut self, idx: usize);
    fn best_cost(&self) -> Option<&C>;
    fn set_best_cost(&mut self, cost: C);

    /// Perform mutation and crossover to fill the trial vectors.
    fn generate_trial_vectors(&mut self);

    /// Evaluate the cost of all trial vectors.
    fn evaluate_trial_vectors(&mut self);

    // --- Hooks (Optional) ---
    fn on_generation_start(&mut self) {}
    fn on_generation_end(&mut self) {}
    fn on_success(&mut self, _idx: usize, _old_ind: &Individual<C>, _new_ind: &Individual<C>) {}

    // --- Default Implementations ---

    /// Selection step: Compares trial vectors against the current population.
    /// Updates the population with better candidates and updates the global best.
    fn selection(&mut self) {
        let pop_len = self.pop().len();
        let mut best_updated = false;
        let mut new_best_idx = self.best_idx();
        let mut new_best_cost = self.best_cost().cloned();

        for i in 0..pop_len {
            let trial_cost = self.trial()[i].cost.as_ref().expect("Trial vector must be evaluated").clone();
            let pop_cost = self.pop()[i].cost.clone();

            let is_better = match pop_cost.as_ref() {
                Some(pc) => trial_cost <= *pc,
                None => true,
            };

            if is_better {
                // Hook for SHADE (archive update, memory update)
                let old_ind = self.pop()[i].clone();
                let new_ind = self.trial()[i].clone();
                self.on_success(i, &old_ind, &new_ind);

                // Replace current with trial
                self.pop_mut()[i] = new_ind;

                // Check global best
                if new_best_cost.is_none() || trial_cost < *new_best_cost.as_ref().unwrap() {
                    new_best_cost = Some(trial_cost);
                    new_best_idx = Some(i);
                    best_updated = true;
                }
            }
        }

        if best_updated {
            if let Some(cost) = new_best_cost {
                self.set_best_cost(cost);
            }
            if let Some(idx) = new_best_idx {
                self.set_best_idx(idx);
            }
        }
    }

    /// Performs one full generation step.
    /// Returns the best cost found so far.
    fn step(&mut self) -> Option<C> {
        self.on_generation_start();
        self.generate_trial_vectors();
        self.evaluate_trial_vectors();
        self.selection();
        self.on_generation_end();
        self.best_cost().cloned()
    }

    /// Main solver loop.
    fn solve(&mut self, max_generations: usize, max_stall_generations: usize) -> usize {
        let mut cost_hist = AllocRingBuffer::<C>::new(max_stall_generations);

        // Ensure initial population is evaluated
        // We assume the constructor or caller has evaluated the initial population.
        // If not, we could add a check here, but `evaluate_trial_vectors` works on trial, not pop.
        // Implementations should ensure `pop` has costs before `solve` is called or `step` is run.

        for i in 0..max_generations {
            let cost = self.step().unwrap();
            cost_hist.enqueue(cost.clone());

            if cost_hist.is_full() {
                let min_val = cost_hist
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                
                if min_val == cost_hist.peek().unwrap() {
                    return i + 1;
                }
            }
        }
        max_generations
    }
    
    fn best(&self) -> Option<(&C, &[f32])> {
        if let Some(idx) = self.best_idx() {
            if let Some(cost) = self.best_cost() {
                return Some((cost, &self.pop()[idx].pos));
            }
        }
        None
    }
}
