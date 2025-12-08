
pub mod monte_carlo;
use monte_carlo::{SchedulingAlgorithm, MonteCarlo};

fn main() {
    let round_robin_scheduling = MonteCarlo::new()
        .set_num_of_iter(200)
        .set_num_of_process(10)
        .set_scheduling_algorithm(SchedulingAlgorithm::RoundRobin)
        .run_simulation();

    if let Ok(sim_result) = round_robin_scheduling{
        println!("Average waiting time: {}", sim_result.avg_waiting_time);
        println!("Waiting Time Standdard Deviation: {}", sim_result.waiting_time_std);
        println!("");
        println!("Average Turn Around Time: {}", sim_result.avg_turn_around);
        println!("Turn Around Time Standard Deviation: {}", sim_result.turn_around_std);
    }

}