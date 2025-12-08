
pub mod monte_carlo;
use monte_carlo::{MonteCarlo};

fn main() {
    let monte_carlo_simulation = MonteCarlo::new()
        .set_num_of_iter(200)
        .set_num_of_process(15)
        .run_simulation();

    if let Ok(sim_result) = monte_carlo_simulation{
        println!("Round Robin Result");
        println!("{:?}", sim_result.round_robin_stats);
        println!("");

        println!("SJF Result");
        println!("{:?}", sim_result.sjf_stats);
        println!("");

        println!("Priority Result");
        println!("{:?}", sim_result.priority_stats);
        
    }

}