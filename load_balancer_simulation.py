"""
Script that uses SimPy to simulate and compare three load balancing policies.

The idea is to generate a list of requests beforehand so every simulation uses the same one,
making the comparison fair.

Policies compared:
- RANDOM: Sends the request to a random server.
- ROUND_ROBIN: Sends to each server in order, cyclically.
- SHORTEST_QUEUE: Sends to the server with the shortest queue.

At the end, it prints a table with the results of each one.
"""
import simpy
import random
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Run Load Balancer Simulation.')

# Add the argument for number of servers
parser.add_argument(
    '--nservers', 
    type=int, 
    required=True, 
    help='Number of servers available in the simulation (e.g., --nservers 30)'
)

# Parse arguments
args = parser.parse_args()
NUM_SERVERS = args.nservers

# Simulation Settings
POLICIES_TO_TEST = ['RANDOM', 'ROUND_ROBIN', 'SHORTEST_QUEUE'] # Policies to be tested
REQUEST_ARRIVAL_RATE = 10                                      # Average number of requests arriving per second
SIMULATION_TIME = 100                                          # Total simulation duration in seconds

# Processing time (min, max) for each type of request
PROCESSING_TIMES = {
    'CPU_INTENSIVE': (1, 8),
    'IO_BOUND': (8, 15)
}

class Colors:
    """Color the terminal output."""
    RESET = '\033[0m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'

class Statistics:
    """Class that stores and calculates simulation metrics."""
    def __init__(self):
        """Start the counters."""
        self.completed_requests = 0
        self.total_response_time = 0.0

    def record_completion(self, request):
        """
        Updates the statistics when a request is completed.
        """
        self.completed_requests += 1
        response_time = request.completion_time - request.arrival_time
        self.total_response_time += response_time

    def calculate_throughput(self, sim_time):
        """Calculates the throughput (requests/second)."""
        return self.completed_requests / sim_time if sim_time > 0 else 0

    def calculate_avg_response_time(self):
        """Calculates the average response time for requests."""
        return self.total_response_time / self.completed_requests if self.completed_requests > 0 else 0

class Request:
    """Represents a customer request in the system."""
    def __init__(self, request_id, request_type, arrival_time, processing_time):
        """
        Start a request with its respective characteristics:
        - request_id: Unique ID.
        - request_type: Type of request (ex: 'CPU_INTENSIVE').
        - arrival_time: The moment the reqeuest arrived in the simulation.
        - processing_time: How long it took to be processed.
        """
        self.id = request_id
        self.type = request_type
        self.arrival_time = arrival_time
        self.processing_time = processing_time
        self.completion_time = 0               # It will be modified when the request is completed

class Server:
    """Represents a server that processes one request at a time."""
    def __init__(self, env, server_id, show_logs=True):
        """
        Start a Server.
        - env: SimPy environment.
        - server_id: Unique ID.
        """
        self.env = env
        self.id = server_id
        self.processor = simpy.Resource(env, capacity=1)
        self.queue_len = 0                     # Queue size (including the request being processed).
        self.show_logs = show_logs

    def process_request(self, request):
        """Simulates the time it takes to process the request."""
        # Pause execution for the duration of the request processing time
        yield self.env.timeout(request.processing_time)
        request.completion_time = self.env.now
        if self.show_logs:
            print(f"{Colors.MAGENTA}[Time: {self.env.now:.4f}] Server {self.id}: Request {request.id} finalizado. "
                  f"Response Time: {request.completion_time - request.arrival_time:.4f}{Colors.RESET}")

class LoadBalancer:
    """Distributes incoming requests to the servers."""
    def __init__(self, env, servers, stats, policy='ROUND_ROBIN', show_logs=True):
        """
        Start the LoadBalancer.
        - env: SimPy environment.
        - servers: List of server objects.
        - stats: Statistics object for recording results..
        - policy: Balancing policy to be used.
        """
        self.env = env
        self.servers = servers
        self.stats = stats
        self.policy = policy
        self.rr_counter = 0        # Counter for Round Robin.
        self.show_logs = show_logs

    def handle_request(self, request):
        """Choose a server based on the policy and send the request to it."""
        # Choose a server based on the policy
        if self.policy == 'RANDOM':
            selected_server = random.choice(self.servers)
        elif self.policy == 'ROUND_ROBIN':
            selected_server = self.servers[self.rr_counter]
            self.rr_counter = (self.rr_counter + 1) % len(self.servers)
        elif self.policy == 'SHORTEST_QUEUE':
            loads = [s.queue_len for s in self.servers]
            min_load = min(loads)
            candidates = [s for s, l in zip(self.servers, loads) if l == min_load]
            selected_server = random.choice(candidates)
        else:
            raise ValueError(f"Política desconhecida: {self.policy}")

        # Increase the queue counter and start the server process
        selected_server.queue_len += 1
        if self.show_logs:
            print(f"{Colors.BLUE}[Time: {self.env.now:.4f}] Load Balancer: Request {request.id} sent to the Server {selected_server.id}. "
                  f"Queue length: {selected_server.queue_len}{Colors.RESET}")

        self.env.process(self.server_worker(selected_server, request))

    def server_worker(self, server, request):
        """
        SimPy process that manages the request lifecycle on the server.
        It waits for the server to become available, processes the request, and then saves the results.
        """
        with server.processor.request() as req:
            # Waits until the server becomes available
            yield req
            try:
                # Execute the process
                yield self.env.process(server.process_request(request))
                self.stats.record_completion(request)
            finally:
                # It only removes from the queue when the process is completed
                # Consequently, queue_len >= 1 implies that the server is occupied
                server.queue_len -= 1
                # Sanity check
                if server.queue_len < 0:
                    server.queue_len = 0

def pre_generate_requests(sim_time, request_arrival_rate):
    """
    Creates the request list before the simulation starts.
    This ensures that all policies use the same list, for a fair comparison.
    """
    requests = []
    current_time = 0.0
    # Generates requests until the final simulation time.
    while current_time < sim_time:
        # Calculates when the next request will arrive.
        inter_arrival_time = random.expovariate(request_arrival_rate)
        current_time += inter_arrival_time
        
        if current_time < sim_time:
            # Randomly selects the type and processing time of the request.
            req_type = random.choice(list(PROCESSING_TIMES.keys()))
            min_proc, max_proc = PROCESSING_TIMES[req_type]
            processing_time = random.uniform(min_proc, max_proc)
            
            requests.append({
                'arrival_time': current_time,
                'type': req_type,
                'processing_time': processing_time
            })
    return requests

def request_injector(env, load_balancer, pre_generated_requests, show_logs=True):
    """
    SimPy process that 'injects' requests from the list into the simulation.
    It reads the list and waits for the correct time to create and send the request to the LB.
    """
    for i, req_info in enumerate(pre_generated_requests):
        # Waits until the request arrival time.
        yield env.timeout(max(0, req_info['arrival_time'] - env.now))
        
        # Creates the Request object.
        req = Request(
            request_id=i, 
            request_type=req_info['type'],
            arrival_time=env.now,
            processing_time=req_info['processing_time']
        )
        if show_logs: print(f"{Colors.CYAN}[Time: {env.now:.4f}] Generator: New request {req.type} with ID {req.id} was generated.{Colors.RESET}")
        # Delivers the request to the load balancer.
        load_balancer.handle_request(req)

def run_single_simulation(policy, num_servers, sim_time, pre_generated_requests, show_logs=True):
    """
    Configures and runs the full simulation for one specific policy.
    Returns the average response time and throughput.
    """
    if show_logs:
        print(f"{Colors.BOLD}--- Running the simulation with the policy '{policy}' ---{Colors.RESET}")
    
    # Prepares the simulation environment.
    stats = Statistics()
    env = simpy.Environment()
    servers = [Server(env, i, show_logs=show_logs) for i in range(num_servers)]
    load_balancer = LoadBalancer(env, servers, stats, policy=policy, show_logs=show_logs)
    
    # Starts the process that injects the requests.
    env.process(request_injector(env, load_balancer, pre_generated_requests, show_logs=show_logs))
    
    # Runs the simulation.
    env.run(until=sim_time) 
    
    # Calculates the final results.
    throughput = stats.calculate_throughput(sim_time)
    avg_response_time = stats.calculate_avg_response_time()

    if show_logs:
        print(f"Completed. Average Response Time: {avg_response_time:.4f}, Throughput: {throughput:.4f}")

    return avg_response_time, throughput

if __name__ == "__main__":
    # Sets a seed for random to ensure results are always the same.
    # It ensures that the generated request list will always be identical.
    random.seed(42) 
    
    print("Generating the requests list for the simulation...")
    master_request_list = pre_generate_requests(SIMULATION_TIME, REQUEST_ARRIVAL_RATE)
    print(f"{len(master_request_list)} requests were generated.")

    all_results = []
    
    # Runs a simulation for each policy in the list
    for policy in POLICIES_TO_TEST:
        # Resetting the seed here ensures the 'RANDOM' policy behaves the same way
        # every time we run the script, which ensures reproducibility.
        random.seed(42) 
        
        avg_time, throughput = run_single_simulation(
            policy=policy,
            num_servers=NUM_SERVERS,
            sim_time=SIMULATION_TIME,
            pre_generated_requests=master_request_list,
            show_logs=True
        )
        # Saves the result to show at the end.
        all_results.append({'Policy': policy, 'Avg Response Time (s)': avg_time, 'Throughput (req/s)': throughput})

    # Uses pandas to print the results table.
    results_df = pd.DataFrame(all_results)
    
    # Shows the final table comparing the results.
    print(f"\n{Colors.BOLD}{Colors.GREEN}--- Final Results ---{Colors.RESET}")
    print(results_df.to_string(index=False))