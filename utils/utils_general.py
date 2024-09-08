def create_this_perf(*kwargs):
    this_perf = []
    for value in kwargs:
        cycles = value.getCounter("cycles")
        instructions = value.getCounter("instructions")
        L1_misses = value.getCounter("L1-misses")
        LLC_misses = value.getCounter("LLC-misses")
        L1_accesses = value.getCounter("L1-accesses")
        LLC_accesses = value.getCounter("LLC-accesses")
        branch_misses = value.getCounter("branch-misses")
        task_clock = value.getCounter("task-clock")
        this_perf += [cycles, instructions,
                         L1_misses, LLC_misses,
                         L1_accesses, LLC_accesses,
                         branch_misses, task_clock]
    return this_perf