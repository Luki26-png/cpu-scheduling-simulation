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
pub enum SchedulingAlgorithm {
    ShortestJobFirst,
    Priority,
    RoundRobin,
    None
}

pub struct MonteCarlo{
    number_of_process: usize,
    number_of_iteration: usize,
    algorithm: SchedulingAlgorithm,

    //vector that will store statistic for each iteration in a tuple: (turn around, waiting time), 
    each_iter_result: Vec<(f64,f64)>,
    //variable to store the overall stats
    pub avg_turn_around: f64,
    pub avg_waiting_time: f64,
    pub turn_around_std: f64,
    pub waiting_time_std: f64
}

impl MonteCarlo{
    pub fn new()->Self{
        Self {
            number_of_process: 0, 
            number_of_iteration: 0,
            algorithm: SchedulingAlgorithm::None,
            each_iter_result: Vec::new(),
            avg_turn_around: 0.0,
            avg_waiting_time: 0.0,
            turn_around_std: 0.0,
            waiting_time_std: 0.0 
        }
    }

    pub fn set_num_of_process(mut self, num_process: usize)->Self{
        self.number_of_process = num_process;
        self
    }

    pub fn set_num_of_iter(mut self, num_iter: usize)->Self{
        self.number_of_iteration = num_iter;
        self.each_iter_result = Vec::with_capacity(num_iter);
        self
    }

    pub fn set_scheduling_algorithm(mut self, algorithm: SchedulingAlgorithm)->Self{
        self.algorithm = algorithm;
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

          //turn them into a vector
          let mut processes_list: Vec<Process> = Vec::with_capacity(self.number_of_process);
          for i in 0..self.number_of_process{
            processes_list.push(Process::new(i, arrival_time[i], burst_time[i]));
          }  

          //put then insto scheduling algorithm
          match self.algorithm{
            SchedulingAlgorithm::Priority=> priority_scheduling(&mut processes_list),
            SchedulingAlgorithm::ShortestJobFirst=>sjf_scheduling(&mut processes_list),
            SchedulingAlgorithm::RoundRobin=>round_robin_scheduling(&mut processes_list),
            _ => return Err(String::from("Please set the scheduling algorithm that will be used"))
          };

          //for the current iteration result,
          //push the turn around time and waiting time into each_iter_result: Vec<(f64,f64)>,
          for process in processes_list{
            self.each_iter_result.push((process.turnaround_time, process.waiting_time));
          }
        }

        //calculate the Overall average of turnaround time and waiting time
        for (turnaround, waiting) in &self.each_iter_result{
            self.avg_turn_around += turnaround;
            self.avg_waiting_time += waiting;
        }

        self.avg_turn_around /= self.number_of_iteration as f64;
        self.avg_waiting_time /= self.number_of_iteration as f64;

        //calculate the standard deviation

        //calculate each different to the mean
        for (turnaround, waiting) in &self.each_iter_result{
            self.turn_around_std = (self.turn_around_std + self.avg_turn_around - turnaround).abs();
            self.waiting_time_std = (self.waiting_time_std + self.avg_waiting_time - waiting).abs();
        }

        //divide with total element to get the variant
        self.turn_around_std /= self.number_of_iteration as f64;
        self.waiting_time_std /= self.number_of_iteration as f64;

        //get standard deviation from the variant by taking its square root
        self.turn_around_std = self.turn_around_std.sqrt();
        self.waiting_time_std = self.waiting_time_std.sqrt();

        Ok(self)
    }
}