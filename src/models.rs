use std::collections::VecDeque;

#[derive(Debug, Clone, Default)]
pub struct Process {
    pub id: usize,                    // Unique identifier
    pub arrival_time: f64,            // When process arrives in system
    pub burst_time: f64,              // Total CPU time needed
    pub remaining_time: f64,          // Time left to complete
    pub waiting_time: f64,            // Total time spent waiting
    pub turnaround_time: f64,         // Completion time - arrival time
    pub completion_time: f64, // When process finishes
    pub quantum: f64
    //pub priority: Option<usize>,      // For priority-based scheduling
}

impl Process{
    pub fn new(id: usize, arrival_time: f64, burst_time: f64) -> Self{
        Self { 
            id: id,
            arrival_time: arrival_time, 
            burst_time: burst_time, 
            remaining_time: burst_time,
            waiting_time: 0.0,
            turnaround_time: 0.0,
            completion_time: 0.0,
            quantum: 7.0 
        }
    }
}

/// A priority queue implemented as a binary min-heap for `Process` structs.
///
/// The heap maintains processes in order of their `remaining_time`, where the process
/// with the smallest remaining time is always at the root. This makes it ideal for
/// scheduling algorithms like Shortest Remaining Time First (SRTF).
///
/// # Implementation Details
/// - **Data Structure**: Binary min-heap using a vector for storage
/// - **Key Property**: For any node at index `i`, its children at `2*i+1` and `2*i+2`
///   have `remaining_time >=` the parent's `remaining_time`
/// - **Time Complexity**:
///   - `push()`: O(log n)
///   - `pop()`: O(log n)
///   - `peek()`: O(1)
///
/// # Example
/// ```
/// let mut heap = MinHeap::new();
/// heap.push(Process::new(1, 0.0, 10.0));  // remaining_time: 10.0
/// heap.push(Process::new(2, 1.0, 5.0));   // remaining_time: 5.0
/// heap.push(Process::new(3, 2.0, 8.0));   // remaining_time: 8.0
///
/// // Process 2 (remaining_time = 5.0) will be popped first
/// assert_eq!(heap.pop().unwrap().id, 2);
/// assert_eq!(heap.pop().unwrap().id, 3);
/// assert_eq!(heap.pop().unwrap().id, 1);
/// ```
pub struct MinHeap {
    data: Vec<Process>,
}

impl MinHeap {
    /// Creates a new empty min-heap.
    pub fn new() -> Self {
        MinHeap { data: Vec::new() }
    }

    /// Returns `true` if the heap contains no processes.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the number of processes in the heap.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns a reference to the process with the smallest remaining time,
    /// or `None` if the heap is empty.
    ///
    /// This operation does not remove the process from the heap.
    pub fn peek(&self) -> Option<&Process> {
        self.data.first()
    }

    /// Adds a process to the heap.
    ///
    /// The heap is restructured to maintain the min-heap property.
    pub fn push(&mut self, process: Process) {
        self.data.push(process);
        self.heapify_up(self.data.len() - 1);
    }

    /// Removes and returns the process with the smallest remaining time,
    /// or `None` if the heap is empty.
    ///
    /// After removal, the heap is restructured to maintain the min-heap property.
    pub fn pop(&mut self) -> Option<Process> {
        if self.data.is_empty() {
            return None;
        }

        let last = self.data.len() - 1;
        self.data.swap(0, last);

        let min = self.data.pop();

        if !self.data.is_empty() {
            self.heapify_down(0);
        }

        min
    }

    /// Restores the heap property by moving a process up the heap.
    ///
    /// Starting from `index`, the process is repeatedly swapped with its parent
    /// while its `remaining_time` is less than its parent's.
    fn heapify_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;

            if self.data[index].remaining_time < self.data[parent].remaining_time {
                self.data.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    /// Restores the heap property by moving a process down the heap.
    ///
    /// Starting from `index`, the process is repeatedly swapped with its smallest
    /// child while it has a child with smaller `remaining_time`.
    fn heapify_down(&mut self, mut index: usize) {
        let len = self.data.len();

        loop {
            let left = 2 * index + 1;
            let right = 2 * index + 2;
            let mut smallest = index;

            if left < len &&
                self.data[left].remaining_time < self.data[smallest].remaining_time
            {
                smallest = left;
            }

            if right < len &&
                self.data[right].remaining_time < self.data[smallest].remaining_time
            {
                smallest = right;
            }

            if smallest != index {
                self.data.swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }

    /// Removes and returns the next valid process according to a predicate.
    ///
    /// Repeatedly pops processes from the heap until one satisfies the `is_valid`
    /// predicate. Processes that don't satisfy the predicate are discarded.
    ///
    /// # Arguments
    /// * `is_valid` - A closure that takes a reference to a process and returns
    ///   `true` if the process should be returned, `false` if it should be discarded.
    ///
    /// # Example
    /// ```
    /// // Pop the first process that has arrived (arrival_time <= current_time)
    /// let current_time = 5.0;
    /// let next_process = heap.pop_valid(|p| p.arrival_time <= current_time);
    /// ```
    pub fn pop_valid<F>(&mut self, mut is_valid: F) -> Option<Process>
    where
        F: FnMut(&Process) -> bool,
    {
        loop {
            let p = self.pop()?;
            if is_valid(&p) {
                return Some(p);
            }
            // If not valid, continue popping
        }
    }
}


/// A circular queue for managing processes in a Round Robin scheduling simulation.
///
/// This queue maintains references to `Process` objects, allowing them to be scheduled
/// in a circular fashion. It's designed specifically for process scheduling simulations
/// where processes need to be processed in a FIFO manner with the ability to cycle back.
///
/// # Type Parameters
/// * `'a` - Lifetime parameter ensuring references to processes don't outlive the processes themselves.
///
/// # Safety Considerations
/// The use of mutable references (`&'a mut Process`) ensures exclusive access to each process,
/// preventing data races in concurrent scenarios. However, it also means that once a process
/// is added to the queue, the original owner cannot access it until it's removed.
pub struct CircularQueue<'a> {
    /// Internal double-ended queue storing mutable references to processes.
    ///
    /// Using `VecDeque` provides O(1) amortized operations for both front and back operations,
    /// which is essential for efficient round robin scheduling.
    pub dequeue: VecDeque<&'a mut Process>,
}

impl<'a> CircularQueue<'a> {
    /// Creates a new `CircularQueue` with the specified capacity.
    ///
    /// # Arguments
    /// * `cap` - The initial capacity of the internal buffer.
    ///
    /// # Examples
    /// ```
    /// let queue = CircularQueue::new(10);
    /// ```
    pub fn new(cap: usize) -> Self {
        Self { 
            dequeue: VecDeque::with_capacity(cap) 
        }
    }

    /// Appends multiple processes to the back of the queue in a single operation.
    ///
    /// This method is optimized for efficiently adding multiple processes at once,
    /// particularly useful when initializing the queue with processes that have already
    /// arrived by a given time in the simulation.
    ///
    /// # Precondition
    /// The input slice `processes` should be sorted by arrival time in ascending order.
    /// This ensures that processes are queued in the correct chronological order.
    /// Violating this precondition may lead to incorrect scheduling behavior.
    ///
    /// # Arguments
    /// * `processes` - A mutable slice of `Process` references to be added to the queue.
    ///                 After this call, the caller loses access to these processes
    ///                 until they are removed from the queue via `pop_front()`.
    ///
    /// # Performance
    /// Time complexity: O(n) where n is the length of `processes`.
    /// Space complexity: O(1) additional space (amortized O(1) per element for resizing).
    ///
    /// # Example
    /// ```
    /// let mut queue = CircularQueue::new(5);
    /// let mut processes = vec![
    ///     Process::new(1, 0, 5),  // pid=1, arrival=0, burst=5
    ///     Process::new(2, 0, 3),  // pid=2, arrival=0, burst=3
    ///     Process::new(3, 0, 2),  // pid=3, arrival=0, burst=2
    /// ];
    /// 
    /// // Important: Sort by arrival time before adding
    /// processes.sort_by_key(|p| p.arrival_time);
    /// 
    /// queue.push_back_many(&mut processes);
    /// // Note: Can no longer directly access `processes` elements
    /// // until they're popped from the queue
    /// ```
    ///
    /// # Use in Round Robin Simulation
    /// This method is particularly useful when:
    /// 1. Initializing the ready queue at simulation start
    /// 2. Adding a batch of processes that have arrived at the same time quantum
    /// 3. Bulk-loading processes after sorting by arrival time
    ///
    /// # Alternatives
    /// For adding processes incrementally as they arrive during simulation,
    /// use `push_back()` instead.
    pub fn push_back_many(&mut self, processes: &'a mut [Process]) {
        for process in processes {
            self.dequeue.push_back(process);
        }
    }

    /// Appends a single process to the back of the queue.
    ///
    /// # Arguments
    /// * `process` - A mutable reference to the process to be added.
    pub fn push_back(&mut self, process: &'a mut Process) {
        self.dequeue.push_back(process);
    }

    /// Removes and returns the process at the front of the queue.
    ///
    /// # Returns
    /// * `Some(&'a mut Process)` if the queue was not empty
    /// * `None` if the queue was empty
    ///
    /// # Example
    /// ```
    /// while let Some(next_process) = queue.pop_front() {
    ///     // Execute the process for one time quantum
    ///     execute_process(next_process, time_quantum);
    ///     
    ///     // If process needs more CPU time, push it back to the queue
    ///     if next_process.remaining_time > 0 {
    ///         queue.push_back(next_process);
    ///     }
    /// }
    /// ```
    pub fn pop_front(&mut self) -> Option<&'a mut Process> {
        self.dequeue.pop_front()
    }

    /// Returns the number of processes in the queue.
    pub fn len(&self) -> usize {
        self.dequeue.len()
    }

    /// Returns `true` if the queue contains no processes.
    pub fn is_empty(&self) -> bool {
        self.dequeue.is_empty()
    }
}