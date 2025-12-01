#[derive(Debug, Clone, Default)]
pub struct Process {
    pub id: usize,                    // Unique identifier
    pub arrival_time: f64,            // When process arrives in system
    pub burst_time: f64,              // Total CPU time needed
    pub remaining_time: f64,          // Time left to complete
    pub waiting_time: f64,            // Total time spent waiting
    pub turnaround_time: f64,         // Completion time - arrival time
    //pub response_time: Option<f64>,   // First time getting CPU
    pub completion_time: f64, // When process finishes
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
            completion_time: 0.0 
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