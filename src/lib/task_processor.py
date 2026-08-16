"""Task execution against GPU clocks.

A task placed by the distributor either completes before its due date
(the GPU clock advances to the finish time) or is killed (deadline miss;
the GPU is not occupied). Completion time is

    max(gpu_time, release) + AP comm + broker comm + BS comm + proc/speed
"""


class TaskProcessor:
    def __init__(self, edge_servers, enable_logging=False):
        self.enable_logging = enable_logging
        self.edge_servers = edge_servers
        self.completed_tasks = []
        self.killed_tasks = []
        self.total_benefit = 0

        # per-class results
        self.completed_tc = []
        self.completed_ntc = []
        self.killed_tc = []
        self.killed_ntc = []

    def log_to_file(self, log_entry, file_path='log.txt'):
        with open(file_path, 'a') as file:
            file.write(log_entry + "\n")

    def process_task(self, task, distributor):
        server, gpu = distributor.select(task)
        arrival = (task['Release Time']
                   + task['Access Point Communication Time']
                   + task['Broker Communication Time']
                   + task[f'Base Station {server.id} Communication Time'])
        start_time = max(gpu.current_time, arrival)
        end_time = start_time + task['Processing Time'] / gpu.speed

        priority = task['PriorityClass']
        if end_time <= task['Due Date']:
            gpu.current_time = end_time
            gpu.completed_tasks.append(task)
            self.completed_tasks.append(task)
            (self.completed_tc if priority == 'TC' else self.completed_ntc).append(task)
            self.total_benefit += task['Benefit']
            status = 'Completed'
        else:
            self.killed_tasks.append(task)
            (self.killed_tc if priority == 'TC' else self.killed_ntc).append(task)
            status = 'Killed'

        if self.enable_logging:
            self.log_to_file(
                f"Task {task['taskid']} ({priority}) -> ES{server.id}/GPU{gpu.id} "
                f"start={start_time:.2f} end={end_time:.2f} "
                f"due={task['Due Date']} status={status}"
            )
        return self.completed_tasks, self.killed_tasks, self.total_benefit
