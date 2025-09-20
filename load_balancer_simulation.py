import simpy
import random

# Policies: 'RANDOM', 'ROUND_ROBIN', 'SHORTEST_QUEUE'
DEFAULT_POLICY = 'SHORTEST_QUEUE'
NUM_SERVERS = 3
# Average requests per second
REQUEST_ARRIVAL_RATE = 10
SIMULATION_TIME = 100
SIMULATE_BURST_PHASE = True

# Request types and their processing times
PROCESSING_TIMES = {
    # Min/max processing time in seconds
    'CPU_INTENSIVE': (0.1, 0.3),
    'IO_BOUND': (0.4, 0.8)
}

class Colors:
    """A class to hold ANSI color codes for terminal output."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'


class Statistics:
    """A simple class to collect simulation metrics."""
    def __init__(self):
        self.completed_requests = 0
        self.total_response_time = 0.0

    def record_completion(self, request):
        """Records the metrics of a completed request."""
        self.completed_requests += 1
        response_time = request.completion_time - request.arrival_time
        self.total_response_time += response_time

    def calculate_throughput(self, sim_time):
        """Calculates the system throughput."""
        if sim_time == 0:
            return 0
        return self.completed_requests / sim_time

    def calculate_avg_response_time(self):
        """Calculates the average response time."""
        if self.completed_requests == 0:
            return 0
        return self.total_response_time / self.completed_requests


class Request:
    """A simple class to represent a client request."""
    def __init__(self, request_id, request_type, arrival_time):
        self.id = request_id
        self.type = request_type
        self.arrival_time = arrival_time
        self.start_time = 0
        self.completion_time = 0

class Server:
    """A class representing a server with a processing queue."""
    def __init__(self, env, server_id, show_logs=True):
        self.env = env
        self.id = server_id
        self.processor = simpy.Resource(env, capacity=1)
        self.queue_len = 0
        self.show_logs = show_logs

    def process_request(self, request):
        """Simulates the processing of a single request."""
        min_proc, max_proc = PROCESSING_TIMES[request.type]
        processing_time = random.uniform(min_proc, max_proc)
        yield self.env.timeout(processing_time)
        request.completion_time = self.env.now
        if self.show_logs:
            print(f"{Colors.MAGENTA}[Time: {self.env.now:.4f}] Server {self.id}: Finished Request {request.id}. "
                  f"Response Time: {request.completion_time - request.arrival_time:.4f}{Colors.RESET}")

class LoadBalancer:
    """Distributes incoming requests to a set of servers based on a policy."""
    def __init__(self, env, servers, stats, policy='ROUND_ROBIN', show_logs=True):
        self.env = env
        self.servers = servers
        self.stats = stats
        self.policy = policy
        self.rr_counter = 0
        self.show_logs = show_logs

    def handle_request(self, request):
        """Selects a server based on the policy and sends the request."""
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
        request.start_time = self.env.now
        if self.show_logs:
            print(f"{Colors.BLUE}[Time: {self.env.now:.4f}] Load Balancer: Sent Request {request.id} to Server {selected_server.id}. "
                  f"Queue length: {selected_server.queue_len}{Colors.RESET}")
        self.env.process(self.server_worker(selected_server, request))

    def server_worker(self, server, request):
        """A generator that represents the request's lifecycle at the server."""
        with server.processor.request() as req:
            yield req
            server.queue_len -= 1
            yield self.env.process(server.process_request(request))
            self.stats.record_completion(request)


def request_generator(env, load_balancer, sim_time, simulate_burst_phase, request_arrival_rate, show_logs=True):
    """Generates requests based on specified traffic patterns."""
    request_id = 0
    if simulate_burst_phase:
        while env.now < sim_time:
            # Constant flow phase
            if show_logs: print(f"{Colors.BOLD}{Colors.GREEN}--- [Time: {env.now:.4f}] Starting CONSTANT FLOW phase ---{Colors.RESET}")
            flow_duration = random.uniform(10, sim_time / 3)
            phase_end_time = env.now + flow_duration
            while env.now < phase_end_time and env.now < sim_time:
                yield env.timeout(random.expovariate(1.0 / (1.0 / request_arrival_rate)))
                request_type = random.choice(list(PROCESSING_TIMES.keys()))
                req = Request(request_id, request_type, env.now)
                if show_logs: print(f"{Colors.GREEN}[Time: {env.now:.4f}] Generator (Flow): New {req.type} Request {req.id} arrived.{Colors.RESET}")
                load_balancer.handle_request(req)
                request_id += 1
            
            if env.now >= sim_time: break
            
            # Burst phase
            if show_logs: print(f"{Colors.BOLD}{Colors.RED}--- [Time: {env.now:.4f}] Starting BURST phase ---{Colors.RESET}")
            num_burst_requests = random.randint(50, 100)
            for _ in range(num_burst_requests):
                yield env.timeout(random.uniform(0.01, 0.05))
                request_type = random.choice(list(PROCESSING_TIMES.keys()))
                req = Request(request_id, request_type, env.now)
                if show_logs: print(f"{Colors.RED}[Time: {env.now:.4f}] Generator (Burst): New {req.type} Request {req.id} arrived.{Colors.RESET}")
                load_balancer.handle_request(req)
                request_id += 1

            # Cooldown period
            if show_logs: print(f"{Colors.BOLD}{Colors.YELLOW}--- [Time: {env.now:.4f}] Starting COOLDOWN phase ---{Colors.RESET}")
            yield env.timeout(random.uniform(10, 20))
    else:
        while True:
            yield env.timeout(random.expovariate(1.0 / (1.0 / request_arrival_rate)))
            request_type = random.choice(list(PROCESSING_TIMES.keys()))
            req = Request(request_id, request_type, env.now)
            if show_logs: print(f"{Colors.CYAN}[Time: {env.now:.4f}] Generator: New {req.type} Request {req.id} arrived.{Colors.RESET}")
            load_balancer.handle_request(req)
            request_id += 1

def run_simulation(policy, arrival_rate, sim_time, num_servers, burst_phase, show_logs=True):
    """Sets up and runs the simulation, returning key metrics."""
    if show_logs:
        print(f"{Colors.BOLD}--- Starting Simulation with {policy} policy ---{Colors.RESET}\n")
    
    stats = Statistics()
    env = simpy.Environment()
    servers = [Server(env, i, show_logs) for i in range(num_servers)]
    load_balancer = LoadBalancer(env, servers, stats, policy=policy, show_logs=show_logs)
    env.process(request_generator(env, load_balancer, sim_time, burst_phase, arrival_rate, show_logs))
    env.run(until=sim_time)
    
    throughput = stats.calculate_throughput(sim_time)
    avg_response_time = stats.calculate_avg_response_time()

    if show_logs:
        print(f"\n{Colors.BOLD}--- Simulation Finished ---{Colors.RESET}")
        print(f"\n{Colors.BOLD}{Colors.GREEN}--- Results for {policy} ---{Colors.RESET}")
        print(f"{Colors.BOLD}Total simulation time: {sim_time} seconds{Colors.RESET}")
        print(f"{Colors.BOLD}Total requests completed: {stats.completed_requests}{Colors.RESET}")
        print(f"{Colors.BOLD}Throughput: {throughput:.4f} requests/sec{Colors.RESET}")
        print(f"{Colors.BOLD}Average response time: {avg_response_time:.4f} seconds{Colors.RESET}")

    return avg_response_time, throughput

if __name__ == "__main__":
    run_simulation(
        policy=DEFAULT_POLICY,
        arrival_rate=REQUEST_ARRIVAL_RATE,
        sim_time=SIMULATION_TIME,
        num_servers=NUM_SERVERS,
        burst_phase=SIMULATE_BURST_PHASE,
        show_logs=True
    )