import simpy
import random
import pandas as pd

# --- Simulation Configuration ---
POLICIES_TO_TEST = ['RANDOM', 'ROUND_ROBIN', 'SHORTEST_QUEUE']
NUM_SERVERS = 3
REQUEST_ARRIVAL_RATE = 10 
SIMULATION_TIME = 100
SIMULATE_BURST_PHASE = True 

PROCESSING_TIMES = {
    'CPU_INTENSIVE': (1, 15),
    'IO_BOUND': (1, 15)
}

class Colors:
    """A class to hold ANSI color codes for terminal output."""
    RESET = '\033[0m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'

class Statistics:
    def __init__(self):
        self.completed_requests = 0
        self.total_response_time = 0.0

    def record_completion(self, request):
        self.completed_requests += 1
        response_time = request.completion_time - request.arrival_time
        self.total_response_time += response_time

    def calculate_throughput(self, sim_time):
        return self.completed_requests / sim_time if sim_time > 0 else 0

    def calculate_avg_response_time(self):
        return self.total_response_time / self.completed_requests if self.completed_requests > 0 else 0

class Request:
    def __init__(self, request_id, request_type, arrival_time, processing_time):
        self.id = request_id
        self.type = request_type
        self.arrival_time = arrival_time
        self.processing_time = processing_time
        self.completion_time = 0

class Server:
    def __init__(self, env, server_id):
        self.env = env
        self.id = server_id
        self.processor = simpy.Resource(env, capacity=1)
        self.queue_len = 0

    def process_request(self, request):
        yield self.env.timeout(request.processing_time)
        request.completion_time = self.env.now

class LoadBalancer:
    def __init__(self, env, servers, stats, policy='ROUND_ROBIN'):
        self.env = env
        self.servers = servers
        self.stats = stats
        self.policy = policy
        self.rr_counter = 0

    def handle_request(self, request):
        if self.policy == 'RANDOM':
            selected_server = random.choice(self.servers)
        elif self.policy == 'ROUND_ROBIN':
            selected_server = self.servers[self.rr_counter]
            self.rr_counter = (self.rr_counter + 1) % len(self.servers)
        elif self.policy == 'SHORTEST_QUEUE':
            selected_server = min(self.servers, key=lambda s: s.queue_len)
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        selected_server.queue_len += 1
        self.env.process(self.server_worker(selected_server, request))

    def server_worker(self, server, request):
        with server.processor.request() as req:
            yield req 
            server.queue_len -= 1
            yield self.env.process(server.process_request(request))
            self.stats.record_completion(request)

def pre_generate_requests(sim_time, simulate_burst_phase, request_arrival_rate):
    requests = []
    current_time = 0.0
    while current_time < sim_time:
        inter_arrival_time = random.expovariate(request_arrival_rate)
        current_time += inter_arrival_time
        if current_time < sim_time:
            req_type = random.choice(list(PROCESSING_TIMES.keys()))
            min_proc, max_proc = PROCESSING_TIMES[req_type]
            processing_time = random.uniform(min_proc, max_proc)
            requests.append({
                'arrival_time': current_time,
                'type': req_type,
                'processing_time': processing_time
            })
    return requests

def request_injector(env, load_balancer, pre_generated_requests):
    for i, req_info in enumerate(pre_generated_requests):
        yield env.timeout(max(0, req_info['arrival_time'] - env.now))
        req = Request(
            request_id=i, 
            request_type=req_info['type'],
            arrival_time=env.now,
            processing_time=req_info['processing_time']
        )
        load_balancer.handle_request(req)

def run_single_simulation(policy, num_servers, sim_time, pre_generated_requests, show_logs=False):
    if show_logs:
        print(f"{Colors.CYAN}--- Running Simulation for '{policy}' policy ---{Colors.RESET}")
    
    stats = Statistics()
    env = simpy.Environment()
    servers = [Server(env, i) for i in range(num_servers)]
    load_balancer = LoadBalancer(env, servers, stats, policy=policy)
    env.process(request_injector(env, load_balancer, pre_generated_requests))
    
    # BUG FIX: Changed env.run() to env.run(until=sim_time)
    # This ensures the simulation runs for the full duration, allowing
    # queued requests to be processed and counted.
    env.run(until=sim_time) 
    
    throughput = stats.calculate_throughput(sim_time)
    avg_response_time = stats.calculate_avg_response_time()

    if show_logs:
        print(f"Completed. Avg Time: {avg_response_time:.4f}, Throughput: {throughput:.4f}")

    return avg_response_time, throughput

if __name__ == "__main__":
    random.seed(42) 
    master_request_list = pre_generate_requests(SIMULATION_TIME, SIMULATE_BURST_PHASE, REQUEST_ARRIVAL_RATE)
    all_results = []
    for policy in POLICIES_TO_TEST:
        random.seed(42) 
        avg_time, throughput = run_single_simulation(
            policy=policy,
            num_servers=NUM_SERVERS,
            sim_time=SIMULATION_TIME,
            pre_generated_requests=master_request_list,
            show_logs=True
        )
        all_results.append({'Policy': policy, 'Avg Response Time (s)': avg_time, 'Throughput (req/s)': throughput})
    results_df = pd.DataFrame(all_results)
    print(f"\n{Colors.BOLD}{Colors.GREEN}--- Overall Comparative Results ---{Colors.RESET}")
    print(results_df.to_string(index=False))