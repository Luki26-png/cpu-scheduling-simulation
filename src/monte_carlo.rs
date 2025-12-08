pub mod algorithm;
pub mod models;

use algorithm::{
    generate_exp_sample,
    generate_poisson_sample,
    sjf_scheduling,
    priority_scheduling,
    round_robin_scheduling
};

use models::Process;

#[derive(Default, Debug)]
pub struct MonteCarloStats{
    pub avg_turn_around: f64,
    pub avg_waiting_time: f64,
    pub turn_around_std: f64,
    pub waiting_time_std: f64      
}

impl MonteCarloStats{
    pub fn calculate_stats(&mut self, monte_carlo_result: &Vec<(f64, f64)>){
        let result_length = monte_carlo_result.len();
        //calculate the Overall average of turnaround time and waiting time
        for (turnaround, waiting) in monte_carlo_result{
            self.avg_turn_around += turnaround;
            self.avg_waiting_time += waiting;
        }

        self.avg_turn_around /= result_length as f64;
        self.avg_waiting_time /= result_length as f64;

        //calculate the standard deviation
        //calculate each different to the mean
        for (turnaround, waiting) in monte_carlo_result{
            self.turn_around_std = (self.turn_around_std + self.avg_turn_around - turnaround).abs();
            self.waiting_time_std = (self.waiting_time_std + self.avg_waiting_time - waiting).abs();
        }

        //divide with total element to get the variant
        self.turn_around_std /= result_length as f64;
        self.waiting_time_std /= result_length as f64;

        //get standard deviation from the variant by taking its square root
        self.turn_around_std = self.turn_around_std.sqrt();
        self.waiting_time_std = self.waiting_time_std.sqrt();

        //round each stat so that it only show 3 digit after decimal
        self.avg_turn_around = self.round_to_decimals(self.avg_turn_around, 3);
        self.avg_waiting_time = self.round_to_decimals(self.avg_waiting_time, 3);
        self.turn_around_std = self.round_to_decimals(self.turn_around_std, 3);
        self.waiting_time_std = self.round_to_decimals(self.waiting_time_std, 3);
    }

    //Round to N decimals
    pub fn round_to_decimals(&self, x: f64, decimals: u32) -> f64 {
        let factor = 10_f64.powi(decimals as i32);
        (x * factor).round() / factor
    }
}

pub struct MonteCarlo{
    pub number_of_process: usize,
    pub number_of_iteration: usize,

    //vector that will store statistic for each iteration in a tuple: (turn around, waiting time), 
    each_iter_result_round_robin: Vec<(f64,f64)>,
    each_iter_result_sjf: Vec<(f64,f64)>,
    each_iter_result_priority: Vec<(f64,f64)>,

    //variable to store the overall stats
    pub round_robin_stats: MonteCarloStats,
    pub priority_stats: MonteCarloStats,
    pub sjf_stats: MonteCarloStats
}

impl MonteCarlo{
    pub fn new()->Self{
        Self {
            number_of_process: 0, 
            number_of_iteration: 0,
            each_iter_result_round_robin: Vec::new(),
            each_iter_result_priority: Vec::new(),
            each_iter_result_sjf: Vec::new(),
            round_robin_stats: MonteCarloStats::default(),
            priority_stats: MonteCarloStats::default(),
            sjf_stats: MonteCarloStats::default()
        }
    }

    pub fn set_num_of_process(mut self, num_process: usize)->Self{
        self.number_of_process = num_process;
        self
    }

    pub fn set_num_of_iter(mut self, num_iter: usize)->Self{
        self.number_of_iteration = num_iter;
        self.each_iter_result_round_robin = Vec::with_capacity(num_iter);
        self.each_iter_result_priority = Vec::with_capacity(num_iter);
        self.each_iter_result_sjf = Vec::with_capacity(num_iter);
        self
    }

    pub fn run_simulation(mut self)->Result<Self, String>{
        //guard clause
        if self.number_of_process < 2{
            return Err(String::from(
                "Please set the number of process in each iteration"
            ));
        }

        if self.number_of_iteration < 1{
            return Err(String::from(
                "Please set the number of iteration the Monte Carlo should run"
            ));
        }

        //start the monte carlo simulation
        for _ in 0..self.number_of_iteration{
            //create random arrival time
            let arrival_time = generate_poisson_sample(5.0, self.number_of_process);

            //create random burst time
            let burst_time = generate_exp_sample(5.0, self.number_of_process);

            //process list for round robin
            let mut processes_list_rr: Vec<Process> = Vec::with_capacity(self.number_of_process);

            //process lsit for priority
            let mut processes_list_priority: Vec<Process> = Vec::with_capacity(self.number_of_process);
            //process list for sjf
            let mut processes_list_sjf: Vec<Process> = Vec::with_capacity(self.number_of_process);

            //put randomized arrival time and burst time, into vecotr
            for i in 0..self.number_of_process{
                processes_list_rr.push(Process::new(i, arrival_time[i], burst_time[i]));
                processes_list_priority.push(Process::new(i, arrival_time[i], burst_time[i]));
                processes_list_sjf.push(Process::new(i, arrival_time[i], burst_time[i]));
            }

            //put them insto scheduling algorithm
            round_robin_scheduling(&mut processes_list_rr);
            priority_scheduling(&mut processes_list_priority);
            sjf_scheduling(&mut processes_list_sjf);

            //for the current iteration cycle result,
            //calculate the average turn around and waiting time
            let mut curr_rr_result = (0.0, 0.0);//(turn around, waiting)
            let mut curr_priority_result = (0.0, 0.0);
            let mut curr_sjf_result = (0.0, 0.0);

            //calculate for the round robin
            for process in processes_list_rr{
                curr_rr_result.0 += process.turnaround_time;
                curr_rr_result.1 += process.waiting_time;
            }
            curr_rr_result.0 /= self.number_of_process as f64;
            curr_rr_result.1 /= self.number_of_process as f64;

            //calculate for the priority
            for process in processes_list_priority{
                curr_priority_result.0 += process.turnaround_time;
                curr_priority_result.1 += process.waiting_time;
            }
            curr_priority_result.0 /= self.number_of_process as f64;
            curr_priority_result.1 /= self.number_of_process as f64;

            //calculate for the shorted job first
            for process in processes_list_sjf{
                curr_sjf_result.0 += process.turnaround_time;
                curr_sjf_result.1 += process.waiting_time;
            }
            curr_sjf_result.0 /= self.number_of_process as f64;
            curr_sjf_result.1 /= self.number_of_process as f64;

            //push the result into corresponding iteration result container
            self.each_iter_result_round_robin.push(curr_rr_result);
            self.each_iter_result_priority.push(curr_priority_result);
            self.each_iter_result_sjf.push(curr_sjf_result);
        }

        self.round_robin_stats.calculate_stats(&self.each_iter_result_round_robin);
        self.priority_stats.calculate_stats(&self.each_iter_result_priority);
        self.sjf_stats.calculate_stats(&self.each_iter_result_sjf);

        Ok(self)
    }
}