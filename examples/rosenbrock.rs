extern crate differential_evolution;

use differential_evolution::{self_adaptive_de, shade, DifferentialEvolution};

fn rosenbrock(pos: &[f32]) -> f32 {
    let mut result = 0.0;
    for i in 0..pos.len() - 1 {
        let x = pos[i];
        let y = pos[i + 1];
        let a = 1.0 - x;
        let b = y - x * x;
        result += a * a + 100.0 * b * b;
    }
    result
}

fn main() {
    let dim = 6;
    let min_max = vec![(-5.0, 5.0); dim];
    
    println!("Optimizing Rosenbrock function (dim={})", dim);
    println!("Global minimum is 0 at (1, 1, ..., 1)");
    println!("------------------------------------------------");

    // SDE
    println!("Running SDE...");
    let mut sde = self_adaptive_de(min_max.clone(), rosenbrock);
    let epochs = sde.solve(10000, 100);
    let (cost, pos) = sde.best().unwrap();
    println!("SDE Result ({} epochs):", epochs);
    println!("  Cost: {}", cost);
    println!("  Pos:  {:?}", pos);
    println!();

    // SHADE
    println!("Running SHADE...");
    let mut shade = shade(min_max.clone(), rosenbrock);
    let epochs = shade.solve(10000, 100);
    let (cost, pos) = shade.best().unwrap();
    println!("SHADE Result ({} epochs):", epochs);
    println!("  Cost: {}", cost);
    println!("  Pos:  {:?}", pos);
}
