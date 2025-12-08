use crate::monte_carlo::models::{Process, MinHeap, CircularQueue, MaxHeap};
use rand_distr::{Distribution, Exp, Poisson, Uniform};
use rand;
/// Generates a vector of random numbers following a Poisson distribution.
/// The Poisson distribution models the number of events occurring in a fixed 
/// interval of time or space, given a known average rate (lambda).
///
/// # Arguments
/// * `lambda` - The expected number of occurrences in the given interval. 
///              Must be positive (lambda > 0).
/// * `n` - The number of random samples to generate.
///
/// # Returns
/// A `Vec<f64>` where each element represents a count of events (non-negative integer value
/// stored as floating-point number).
///
/// # Panics
/// This function will panic if `lambda` is not positive (via `Poisson::new()`).
///
/// # Example
/// ```
/// // Generate 10 samples with average rate of 3.5 events per interval
/// let samples = generate_poisson_sample(3.5, 10);
/// ```
pub fn generate_poisson_sample(lambda:f64, n: usize) -> Vec<f64>{
    let poi = Poisson::new(lambda).unwrap();
    let poisson_series: Vec<f64> = poi.sample_iter(&mut rand::rng()).take(n).collect();
    return poisson_series;
}

/// Generates a vector of random numbers following a transformed exponential distribution.
/// The exponential distribution models the time between events in a Poisson process,
/// then scales and applies ceiling to convert to discrete values.
///
/// # Arguments
/// * `lambda` - The rate parameter (inverse of the mean time between events).
///              Must be positive (lambda > 0).
/// * `n` - The number of random samples to generate.
///
/// # Returns
/// A `Vec<f64>` where each original exponential value is:
/// 1. Scaled by 100.0 (converting time units or amplifying differences)
/// 2. Rounded up to the nearest integer (via `ceil()`)
/// This transformation is useful for generating discrete wait times or counts
/// based on an underlying continuous process.
///
/// # Panics
/// This function will panic if `lambda` is not positive (via `Exp::new()`).
///
/// # Example
/// ```
/// // Generate 5 samples with rate parameter 0.1, then scale and round
/// let samples = generate_exp_sample(0.1, 5);
/// // Original exponential values ~10.0 become 1000.0 after transformation
/// ```
pub fn generate_exp_sample(lambda:f64, n: usize) -> Vec<f64>{
    let exp = Exp::new(lambda).unwrap();
    exp.sample_iter(&mut rand::rng())
        .take(n)
        .map(|value| (value * 100.0).ceil())
        .collect()
}

pub fn generate_priority(uniform: &Uniform<i32>)->i32{
    uniform.sample(&mut rand::rng())
}


/// Preemptive Shortest Job First (SJF) / Shortest Remaining Time First (SRTF) scheduler.
///
/// This function implements a preemptive SJF scheduling algorithm where the process
/// with the shortest remaining execution time always gets the CPU. If a new process
/// arrives with a shorter burst time than the currently executing process, the
/// current process is preempted.
///
/// # Algorithm Overview
/// 1. Processes are sorted by arrival time
/// 2. A min-heap ready queue prioritizes processes by remaining time
/// 3. At each time step:
///    - New arrivals are added to the ready queue
///    - The ready queue may preempt the current process if it has a shorter job
///    - The shortest job is executed until completion or preemption
/// 4. Process statistics (completion time, turnaround time, waiting time) are updated
///
/// # Key Features
/// - **Preemptive**: Running process can be interrupted by shorter jobs
/// - **Optimal for average waiting time**: SJF provides minimal average waiting time
/// - **Float-based timing**: Supports fractional time units for precise simulation
///
/// # Time Complexity
/// - O(n²) worst-case due to bubble sort (consider using faster sort for large datasets)
/// - O(n log n) for heap operations
/// - O(n) for process updates
///
/// # Arguments
/// * `processes` - Mutable vector of processes to schedule. The vector will be modified
///   in-place with updated timing statistics.
///
/// # Example
/// ```
/// let mut processes = vec![
///     Process::new(1, 0.0, 8.0),
///     Process::new(2, 1.0, 4.0),
///     Process::new(3, 2.0, 9.0),
///     Process::new(4, 3.0, 5.0),
/// ];
///
/// sjf_scheduling(&mut processes);
///
/// // Processes now have updated timing information
/// for p in &processes {
///     println!("P{}: WT={:.2}, TAT={:.2}", 
///              p.id, p.waiting_time, p.turnaround_time);
/// }
/// ```
///
/// # Output
/// The function prints a detailed execution trace showing:
/// - Time of each event
/// - Process ID involved
/// - Action taken (Arrived, Started, Preempted, Completed)
///
/// # Notes
/// - Uses floating-point comparison with epsilon (1e-9) for completion detection
/// - Handles idle CPU time when no processes are ready
/// - Processes are validated during popping to ensure consistency
///
/// # Limitations
/// - Assumes zero-cost context switching
/// - Uses bubble sort which is inefficient for large process lists
/// - No priority or I/O operations considered
pub fn sjf_scheduling(processes: &mut Vec<Process>) {
    let mut current_time = 0.0;
    let mut completed = 0;
    let n = processes.len();

    // Sort processes by arrival time
    sort_processes_by_arrival(processes);

    let mut ready_queue = MinHeap::new();
    let mut current_process: Option<Process> = None;
    let mut i = 0; // Index for processes not yet arrived
    
    //println!("Preemptive SJF Scheduling:");
    //println!("Time\tProcess\tAction");

    while completed < n {
        // Add all processes that have arrived by current_time to ready queue
        while i < n && processes[i].arrival_time <= current_time {
            ready_queue.push(Process::new(
                processes[i].id, 
                processes[i].arrival_time, 
                processes[i].burst_time
            ));
            //println!("{:.2}\tP{}\tArrived", current_time, processes[i].id);
            i += 1;
        }

        // Check for preemption: if ready queue has a shorter job than current process
        if let Some(ref mut current) = current_process {
            if let Some(next_process) = ready_queue.peek() {
                if next_process.remaining_time < current.remaining_time {
                    // Preempt current process - move it back to ready queue
                    ready_queue.push(current.clone());
                    //println!("{:.2}\tP{}\tPreempted", current_time, current.id);
                    current_process = None;
                }
            }
        }

        // If no process is currently executing, get the shortest job from ready queue
        if current_process.is_none() {
            let popped = ready_queue.pop_valid(|popped| {
                processes.iter().any(|p| p.id == popped.id)
            });

            if let Some(process) = popped {
                //println!("{:.2}\tP{}\tStarted", current_time, process.id);
                current_process = Some(process);
            }
        }

        // Execute the current process if one exists
        if let Some(ref mut current) = current_process {
            // Determine execution quantum: until next arrival or completion
            let time_quantum = if let Some(next_arrival) = (i..n)
                .map(|idx| processes[idx].arrival_time)
                .filter(|&at| at > current_time)
                .min_by(|a, b| a.partial_cmp(b).unwrap()) 
            {
                // Execute until next arrival or completion, whichever comes first
                (next_arrival - current_time).min(current.remaining_time)
            } else {
                // No more arrivals, execute until completion
                current.remaining_time
            };
            
            // Execute the process for the calculated quantum
            current.remaining_time -= time_quantum;
            current_time += time_quantum;
            
            // Check if process has completed (with floating-point tolerance)
            if current.remaining_time <= 1e-9 {
                // Calculate final statistics
                current.completion_time = current_time;
                current.turnaround_time = current.completion_time - current.arrival_time;
                current.waiting_time = current.turnaround_time - current.burst_time;
                
                // Update the original process in the vector
                if let Some(original_process) = processes.iter_mut().find(|p| p.id == current.id) {
                    original_process.completion_time = current.completion_time;
                    original_process.turnaround_time = current.turnaround_time;
                    original_process.waiting_time = current.waiting_time;
                }
                
                //println!("{:.2}\tP{}\tCompleted", current_time, current.id);
                completed += 1;
                current_process = None;
            }
        } else {
            // No processes ready - idle CPU until next arrival
            if i < n {
                current_time = processes[i].arrival_time;
            } else {
                break; // All processes have arrived and been scheduled
            }
        }
    }
}

/// Executes a Round Robin CPU scheduling algorithm on a collection of processes.
///
/// This function implements a preemptive scheduling algorithm where each process
/// is assigned a fixed time quantum. Processes execute in a circular fashion,
/// with incomplete processes returning to the end of the ready queue after
/// exhausting their time quantum.
///
/// # Algorithm Characteristics
/// - **Preemptive**: Processes can be interrupted after their time quantum expires
/// - **Fair**: All processes get equal opportunity to execute (same time quantum)
/// - **Circular**: Incomplete processes return to queue end for another turn
/// - **FCFS within quantum**: Processes are initially ordered by arrival time
///
/// # Requirements
/// - All processes must have non-negative arrival times and burst times
/// - Each process must have a defined time quantum (stored in `Process.quantum`)
/// - Processes with equal arrival times are processed in their original order
///   after sorting
///
/// # Preconditions
/// 1. Processes are assumed to have `quantum > 0.0`
/// 2. Process arrival time was randomized by using Poisson Distribution
/// 3. Processes have initialized `remaining_time` equal to their `burst_time`
///
/// # Postconditions
/// Upon completion, each process in the input vector will have:
/// - `remaining_time` set to `0.0`
/// - `completion_time` set to when the process finished execution
/// - `turnaround_time` calculated as `completion_time - arrival_time`
/// - `waiting_time` calculated as `turnaround_time - burst_time`
///
/// # Parameters
/// * `processes` - A mutable vector of `Process` instances to be scheduled.
///   The vector will be modified in-place with scheduling results.
///   **Note**: After calling this function, the caller relinquishes mutable
///   access to individual processes until the function completes, as they are
///   moved into the scheduling queue.
///
/// # Implementation Details
/// 1. **Sorting**: Processes are first sorted by arrival time in ascending order
///    using `sort_processes_by_arrival()`.
/// 2. **Time Initialization**: Current time starts at the earliest arrival time.
/// 3. **Queue Management**: All processes are loaded into a `CircularQueue`
///    which manages execution order.
/// 4. **Execution Loop**: Processes execute until completion:
///    - If `remaining_time <= quantum`: Process completes, metrics calculated
///    - If `remaining_time > quantum`: Process executes for quantum, returns to queue
/// 5. **Termination**: Loop ends when all processes have `remaining_time == 0.0`.
///
/// # Time Complexity
/// - Sorting: O(n log n) where n is number of processes
/// - Scheduling: O(n × q) where q is average number of quanta needed per process
/// - Overall: O(n log n + n × q)
///
/// # Space Complexity
/// - O(n) for the circular queue storage
/// - O(1) additional space beyond input and queue
///
/// # Example
/// ```
/// let arrival_time = generate_poisson_sample(5.0, number_of_process);
/// let burst_time = generate_exp_sample(5.0, number_of_process);
///
/// let mut processes_list: Vec<Process> = Vec::with_capacity(number_of_process);
///
/// for i in 0..number_of_process{
///  processes_list.push(Process::new(i, arrival_time[i], burst_time[i]));
/// }
/// round_robin_scheduling(&mut processes);
/// }
/// ```
///
/// # Edge Cases
/// - **Zero burst time**: Process completes immediately at arrival time
/// - **Equal arrival times**: Order preserved from input after sorting
/// - **Very small quantum**: May cause high context switching overhead
/// - **Large quantum**: Approaches FCFS (First-Come, First-Served) behavior
///
/// # Limitations
/// 1. Assumes all processes fit in memory simultaneously
/// 2. Uses per-process quantum values rather than system-wide quantum
/// 3. No I/O operations or process blocking is simulated
///
/// # Safety
/// - This function is memory safe: all references are bound by lifetime `'a`
/// - No panics on empty input (handles n=0 case)
/// - Floating-point arithmetic may produce rounding errors for very small values
///
/// # Related Algorithms
/// - **FCFS** (First-Come, First-Served): Non-preemptive version with infinite quantum
/// - **SJF** (Shortest Job First): Different prioritization strategy
/// - **Priority Scheduling**: Uses priority levels instead of circular order
/// - **Multilevel Queue**: Multiple queues with different scheduling policies
///
/// # See Also
/// - `CircularQueue`: The data structure managing process execution order
/// - `sort_processes_by_arrival()`: Helper function for initial ordering
pub fn round_robin_scheduling<'a>(processes: &'a mut Vec<Process>){
    let mut current_time: f64;
    let mut completed = 0;
    //number of process
    let n = processes.len();

    sort_processes_by_arrival(processes);

    //this is just to make sure that the initial current time will be equal to the first process arrive
    current_time = processes[0].arrival_time;

    //put all processes into circular queue
    let mut ready_queue: CircularQueue<'a> = CircularQueue::new(n);
    ready_queue.push_back_many(processes);



    while completed < n{
        //take on process from queue
        let current_process = ready_queue.pop_front();
        if let Some(process) = current_process{
            //check whether the remaining burst time is bigger or less than the quantum time
            if process.remaining_time <= process.quantum{
                current_time += process.remaining_time;
                process.remaining_time = 0.0;
                process.completion_time = current_time;
                completed += 1;

                //calculate result
                process.turnaround_time = process.completion_time - process.arrival_time;
                process.waiting_time = process.turnaround_time - process.burst_time;
            }else{// process.remaining_time > process.quantum
                current_time += process.quantum;
                process.remaining_time -= process.quantum;
                ready_queue.push_back(process);
            }
        }
    }
}

/// Performs non-preemptive priority scheduling on a set of processes.
/// 
/// This algorithm schedules processes based on their priority, where **lower
/// priority numbers indicate higher priority** (e.g., priority 0 is higher
/// than priority 5). The scheduling is **non-preemptive**, meaning once a
/// process starts executing, it runs to completion without interruption.
/// 
/// # Algorithm Overview
/// 
/// 1. **Initialization**: Assigns random priorities (0-15) to each process
///    and sorts processes by arrival time.
/// 2. **Ready Queue Management**: Maintains a ready queue of processes that
///    have arrived but not yet executed, ordered by priority.
/// 3. **Execution**: Always executes the highest priority process (lowest
///    priority number) from the ready queue.
/// 4. **Statistics Calculation**: Computes waiting time, turnaround time,
///    and completion time for each process.
/// 5. **Result Update**: Updates the original process structures with
///    calculated statistics.
/// 
/// # Scheduling Characteristics
/// 
/// - **Non-preemptive**: No process interruption once started
/// - **Priority-based**: Lower priority number = higher execution priority
/// - **First-come tie-breaking**: Among processes with equal priority, the
///   one that arrived earlier is executed first (due to arrival time sorting)
/// - **Idle CPU**: CPU remains idle if no processes have arrived
/// 
/// # Parameters
/// 
/// * `processes` - A mutable reference to a vector of `Process` structs to be scheduled.
///   Each process will be updated with its calculated statistics.
/// 
/// # Process Updates
/// 
/// After execution, each `Process` in the input vector will have the following
/// fields updated:
/// 
/// - `priority`: Randomly assigned (0-15, lower = higher priority)
/// - `waiting_time`: Time spent waiting in ready queue before execution
/// - `turnaround_time`: Total time from arrival to completion
/// - `completion_time`: Time when process finished execution
/// - `remaining_time`: Set to 0.0 (process completed)
/// 
/// # Time Complexity
/// 
/// - O(n log n) for sorting processes by arrival time
/// - O(n²) worst-case for priority queue operations (due to linear search in `MaxHeap`)
/// - O(n²) for updating results (matching by process ID)
/// 
/// Where `n` is the number of processes.
/// 
/// # Memory Usage
/// 
/// - O(n) additional space for the ready queue
/// - O(n) additional space for storing execution results
/// 
/// # Example
/// 
/// ```
/// use scheduler::{priority_scheduling, Process};
/// 
/// // Create a list of processes with different arrival times and burst times
/// let mut processes = vec![
///     Process::new(1, 0.0, 5.0),   // Process 1: arrives at 0, needs 5 units
///     Process::new(2, 1.0, 3.0),   // Process 2: arrives at 1, needs 3 units
///     Process::new(3, 2.0, 8.0),   // Process 3: arrives at 2, needs 8 units
/// ];
/// 
/// // Run priority scheduling
/// priority_scheduling(&mut processes);
/// 
/// // Processes now contain scheduling statistics
/// for process in &processes {
///     println!("Process {}: priority={}, waiting={:.2}, turnaround={:.2}",
///              process.id, process.priority, process.waiting_time, process.turnaround_time);
/// }
/// 
/// // Calculate average statistics
/// let avg_waiting: f64 = processes.iter().map(|p| p.waiting_time).sum::<f64>() / processes.len() as f64;
/// let avg_turnaround: f64 = processes.iter().map(|p| p.turnaround_time).sum::<f64>() / processes.len() as f64;
/// 
/// println!("Average waiting time: {:.2}", avg_waiting);
/// println!("Average turnaround time: {:.2}", avg_turnaround);
/// ```
/// 
/// # Algorithm Details
/// 
/// ## Priority Assignment
/// Each process is assigned a random priority between 0 and 15 (inclusive)
/// using a uniform distribution. Priority 0 is the highest priority,
/// priority 15 is the lowest.
/// 
/// ## Tie-breaking Rules
/// 1. If multiple processes have the same priority, the one with earlier
///    arrival time is executed first.
/// 2. If arrival times are also equal, the process with lower ID is
///    executed first (implicit due to initial sorting).
/// 
/// ## CPU Idle Time
/// If no processes have arrived by the current time, the CPU idles until
/// the next process arrival time.
/// 
/// ## Statistics Formulas
/// - `waiting_time = start_time - arrival_time`
/// - `turnaround_time = completion_time - arrival_time`
/// - `completion_time = start_time + burst_time`
/// 
/// # Limitations
/// 
/// 1. **Non-preemptive**: Cannot handle cases where a high-priority process
///    arrives while a lower-priority process is executing.
/// 2. **Random priorities**: Priorities are randomly assigned each time the
///    function is called, which may not be desirable for reproducible results.
/// 3. **Performance**: Uses O(n²) operations due to linear search in `MaxHeap`.
/// 
/// # See Also
/// 
/// - [`Process`] - The process structure containing scheduling data
/// - [`MaxHeap`] - The priority queue implementation used internally
/// - [`priority_scheduling_preemptive`] - For a preemptive version of this algorithm
/// 
/// # Notes
/// 
/// - This function modifies the input vector in-place.
/// - Process IDs must be unique for correct statistics assignment.
/// - The random number generator uses thread-local randomness.
/// - Floating-point comparisons use exact equality for arrival time checks.
pub fn priority_scheduling(processes: &mut Vec<Process>) {
    let n = processes.len();
    if n == 0 {
        return;
    }
    
    // Generate random priorities between 0 and 15 (inclusive)
    // Lower numbers indicate higher priority
    let uniform = Uniform::new_inclusive(0, 15).unwrap();
    for process in processes.iter_mut() {
        process.priority = generate_priority(&uniform);
        process.remaining_time = process.burst_time;
    }
    
    // Sort processes by arrival time in ascending order
    // Processes arriving earlier are considered first
    processes.sort_by(|a, b| a.arrival_time.partial_cmp(&b.arrival_time).unwrap());
    
    // Initialize simulation time to the earliest arrival time
    let mut current_time = processes[0].arrival_time;
    let mut completed = 0;     // Number of processes that have finished execution
    let mut i = 0;            // Index of next process to arrive
    
    // Create ready queue for processes that have arrived but not yet executed
    let mut ready_queue = MaxHeap::new(n);
    
    // Temporary storage for execution results
    // Stores tuples of (process_id, waiting_time, turnaround_time, completion_time)
    let mut execution_results: Vec<(usize, f64, f64, f64)> = Vec::new();
    
    // Main scheduling loop: continues until all processes are completed
    while completed < n {
        // Add all processes that have arrived by the current time to ready queue
        while i < n && processes[i].arrival_time <= current_time {
            ready_queue.push(processes[i].clone());
            i += 1;
        }
        
        if !ready_queue.is_empty() {
            // Execute the process with highest priority (lowest priority number)
            let process = ready_queue.pop_highest_priority().unwrap();
            
            // Calculate scheduling statistics for this process
            let waiting_time = current_time - process.arrival_time;
            current_time += process.burst_time;
            let completion_time = current_time;
            let turnaround_time = completion_time - process.arrival_time;
            
            // Store results for later updating of original processes
            execution_results.push((process.id, waiting_time, turnaround_time, completion_time));
            completed += 1;
            
            // Check for any new processes that arrived during execution
            while i < n && processes[i].arrival_time <= current_time {
                ready_queue.push(processes[i].clone());
                i += 1;
            }
        } else {
            // Ready queue is empty: no processes have arrived yet
            // Advance time to the next process arrival
            if i < n {
                current_time = processes[i].arrival_time;
            }
            // Note: if i >= n, all processes have arrived and loop will terminate
        }
    }
    
    // Update original process structures with calculated statistics
    // This matches processes by their unique ID
    for (id, waiting_time, turnaround_time, completion_time) in execution_results {
        for process in processes.iter_mut() {
            if process.id == id {
                process.waiting_time = waiting_time;
                process.turnaround_time = turnaround_time;
                process.completion_time = completion_time;
                process.remaining_time = 0.0; // Mark process as completed
                break; // Found the process, move to next result
            }
        }
    }
}

pub fn print_results(processes: &[Process]) {
    println!("\nProcess Execution Results:");
    println!("PID\tArrival\tBurst\tCompletion\tTurnaround\tWaiting");
    
    let mut total_turnaround = 0.0;
    let mut total_waiting = 0.0;
    
    for process in processes {
        println!("P{}\t{:.2}\t{:.2}\t{:.2}\t\t{:.2}\t\t{:.2}",
                 process.id, process.arrival_time, process.burst_time,
                 process.completion_time, process.turnaround_time, process.waiting_time);
        
        total_turnaround += process.turnaround_time;
        total_waiting += process.waiting_time;
    }
    
    let n = processes.len() as f64;
    println!("\nAverage Turnaround Time: {:.2}", total_turnaround / n);
    println!("Average Waiting Time: {:.2}", total_waiting / n);
}

/// Sorts processes by arrival time using an efficient stable sort.
///
/// This uses Rust's built-in sorting algorithm (typically TimSort for stable sorts),
/// which has O(n log n) worst-case time complexity, much better than bubble sort's O(n²).
///
/// # Arguments
/// * `processes` - A mutable vector of processes to sort by arrival time.
///
/// # Performance
/// - Time complexity: O(n log n) worst-case, O(n) best-case (already sorted)
/// - Space complexity: O(n) worst-case, O(1) best-case
/// - Stable: Yes (preserves order of equal arrival times)
///
/// # Note on Floating-Point Comparison
/// Uses `ordered_float::OrderedFloat` or `f64::total_cmp` for proper total ordering
/// since f64 doesn't implement Ord due to NaN values.
///
/// # Example
/// ```
/// let mut processes = vec![
///     Process::new(1, 5.5, 10),
///     Process::new(2, 2.3, 5),
///     Process::new(3, 8.1, 3),
/// ];
/// 
/// sort_processes_by_arrival(&mut processes);
/// // processes are now sorted: [pid=2 (2.3), pid=1 (5.5), pid=3 (8.1)]
/// ```
fn sort_processes_by_arrival(processes: &mut [Process]) {
    processes.sort_by(|a, b| a.arrival_time.total_cmp(&b.arrival_time));
}