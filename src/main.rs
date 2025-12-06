pub mod models;
pub mod algorithm;
use models::Process;
use algorithm::{generate_poisson_sample, generate_exp_sample};

fn main() {
    let number_of_process:usize = 8;

    let arrival_time = generate_poisson_sample(5.0, number_of_process);
    let burst_time = generate_exp_sample(5.0, number_of_process);

    let mut processes_list: Vec<Process> = Vec::with_capacity(number_of_process);

    for i in 0..number_of_process{
        processes_list.push(Process::new(i, arrival_time[i], burst_time[i]));
    }

    //algorithm::sjf_scheduling(&mut processes_list);
    algorithm::round_robin_scheduling(&mut processes_list);
    algorithm::print_results(&processes_list);
}