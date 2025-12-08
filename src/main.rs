pub mod monte_carlo;
use monte_carlo::{MonteCarlo};
use csv::{WriterBuilder};

fn main(){

    let monte_carlo_simulation = MonteCarlo::new()
        .set_num_of_iter(1000)
        .set_num_of_process(50)
        .run_simulation();

    let mut wtr = WriterBuilder::new()
        .from_path("result.csv").unwrap();

    if let Ok(sim_result) = monte_carlo_simulation{
        let rr_result = sim_result.round_robin_stats;
        let sjf_result = sim_result.sjf_stats;
        let priority_result = sim_result.priority_stats;

        println!("Banyak Iterasi : {} ", sim_result.number_of_iteration);
        println!("Banyak Proses per Iterasi: {}", sim_result.number_of_process);

        println!("Round Robin Result");
        println!("{:?}", rr_result);
        println!("");

        println!("SJF Result");
        println!("{:?}", sjf_result);
        println!("");

        println!("Priority Result");
        println!("{:?}", priority_result);

        wtr.write_record(&["algorithm","avg_turn_around", "avg_waiting_time", "turn_around_std", "waiting_time_std"]).unwrap();
        wtr.write_record(
            &["round_robin",
                rr_result.avg_turn_around.to_string().as_str(),
                rr_result.avg_waiting_time.to_string().as_str(),
                rr_result.turn_around_std.to_string().as_str(),
                rr_result.waiting_time_std.to_string().as_str()
            ]).unwrap();

        wtr.write_record(
            &["sjf",
                sjf_result.avg_turn_around.to_string().as_str(),
                sjf_result.avg_waiting_time.to_string().as_str(),
                sjf_result.turn_around_std.to_string().as_str(),
                sjf_result.waiting_time_std.to_string().as_str()
            ]).unwrap();
        
        wtr.write_record(
            &["priority",
                priority_result.avg_turn_around.to_string().as_str(),
                priority_result.avg_waiting_time.to_string().as_str(),
                priority_result.turn_around_std.to_string().as_str(),
                priority_result.waiting_time_std.to_string().as_str()
            ]).unwrap();
    }
}