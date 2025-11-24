// Copyright 2016 Martin Ankerl. 
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate differential_evolution;

use differential_evolution::{shade, DifferentialEvolution};
use std::f32::consts::PI;
use std::env;

// The Rastrigin function is a non-convex function used as a
// performance test problem for optimization algorithms.
// see https://en.wikipedia.org/wiki/Rastrigin_function 
fn rastrigin(pos: &[f32]) -> f32 {
    pos.iter().fold(0.0, |sum, x| 
        sum + x * x - 10.0 * (2.0 * PI * x).cos() + 10.0)
}

fn main() {
    // command line args: dimension, number of evaluations
    let args: Vec<String> = env::args().collect();
    let dim = if args.len() > 1 { args[1].parse::<usize>().unwrap() } else { 10 };

    // initial search space for each dimension
    let initial_min_max = vec![(-5.12, 5.12); dim];

    // initialize differential evolution
    let mut de = shade(initial_min_max, rastrigin);

    // perform optimization for a maximum of 100000 cost evaluations,
    // or until best cost is below 0.1.
    // Pop size is 100. So 1000 generations.
    
    let epochs = de.solve(1000, 100);

    // see what we've found
    println!("{} epochs done", epochs);
    
    let (cost, pos) = de.best().unwrap();
    println!("{} best cost", cost);
    println!("{:?} best position", pos);
}