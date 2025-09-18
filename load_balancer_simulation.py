import simpy
import random

# Policies: 'RANDOM', 'ROUND_ROBIN', 'SHORTEST_QUEUE'
SELECTED_POLICY = 'SHORTEST_QUEUE' 
NUM_SERVERS = 3
# Average requests per second
REQUEST_ARRIVAL_RATE = 10
SIMULATION_TIME = 5000

# Request types and their processing times
PROCESSING_TIMES = {
    # Min/max processing time in seconds
    'CPU_INTENSIVE': (0.1, 0.3),
    'IO_BOUND': (0.4, 0.8)
}

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
    def __init__(self, env, server_id):
        self.env = env
        self.id = server_id
        # A SimPy resource with a capacity of 1, meaning it can handle one request at a time
        self.processor = simpy.Resource(env, capacity=1)
        self.queue_len = 0

    def process_request(self, request):
        """Simulates the processing of a single request."""
        # Simulate processing delay based on request type
        min_proc, max_proc = PROCESSING_TIMES[request.type]
        processing_time = random.uniform(min_proc, max_proc)
        
        # Handle the passage of time
        yield self.env.timeout(processing_time)
        
        request.completion_time = self.env.now
        print(f"[Time: {self.env.now:.4f}] Server {self.id}: Finished Request {request.id}. "
                f"Response Time: {request.completion_time - request.arrival_time:.4f}")

class LoadBalancer:
    """Distributes incoming requests to a set of servers based on a policy."""
    def __init__(self, env, servers, stats, policy='ROUND_ROBIN'):
        self.env = env
        self.servers = servers
        self.stats = stats
        self.policy = policy
        self.rr_counter = 0 # Counter for Round Robin policy

    def handle_request(self, request):
        """Selects a server based on the policy and sends the request."""
        selected_server = None
        if self.policy == 'RANDOM':
            selected_server = random.choice(self.servers)
        
        elif self.policy == 'ROUND_ROBIN':
            selected_server = self.servers[self.rr_counter]
            self.rr_counter = (self.rr_counter + 1) % len(self.servers)

        elif self.policy == 'SHORTEST_QUEUE':
            # Find server with the minimum queue length
            selected_server = min(self.servers, key=lambda s: s.queue_len)
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        # Send the request to the selected server
        selected_server.queue_len += 1
        request.start_time = self.env.now
        print(f"[Time: {self.env.now:.4f}] Load Balancer: Sent Request {request.id} to Server {selected_server.id}. "
                f"Queue length: {selected_server.queue_len}")

        # Start the processing process on the server
        self.env.process(self.server_worker(selected_server, request))
    
    def server_worker(self, server, request):
        """A generator that represents the request's lifecycle at the server."""
        # This 'with' statement requests the server's processor resource.
        # The request will wait here if the server is busy.
        with server.processor.request() as req:
            # Wait for the processor to become available
            yield req
            server.queue_len -= 1
            # Once available, start processing
            yield self.env.process(server.process_request(request))
            # After the process finishes, we record its duration time
            self.stats.record_completion(request)


def request_generator(env, load_balancer, arrival_rate):
    """Generates requests at random intervals."""
    request_id = 0
    while True:
        # Simulate requests arriving at random intervals
        yield env.timeout(random.expovariate(1.0 / (1.0 / arrival_rate)))

        # Vary request types
        request_type = random.choice(list(PROCESSING_TIMES.keys()))
        
        req = Request(request_id, request_type, env.now)
        print(f"[Time: {env.now:.4f}] Generator: New {req.type} Request {req.id} arrived.")
        
        load_balancer.handle_request(req)
        request_id += 1

def run_simulation():
    """Sets up and runs the simulation."""
    print(f"--- Starting Simulation with {SELECTED_POLICY} policy ---")
    
    # Create the stats collector
    stats = Statistics()

    # Setup the simulation environment
    env = simpy.Environment()
    
    # Create the servers
    servers = [Server(env, i) for i in range(NUM_SERVERS)]
    
    # Create the load balancer
    load_balancer = LoadBalancer(env, servers, stats, policy=SELECTED_POLICY)
    
    # Start the request generator process
    env.process(request_generator(env, load_balancer, REQUEST_ARRIVAL_RATE))
    
    # Run the simulation for a fixed amount of time
    env.run(until=SIMULATION_TIME)
    
    print("--- Simulation Finished ---")

    throughput = stats.calculate_throughput(SIMULATION_TIME)
    avg_response_time = stats.calculate_avg_response_time()
    
    print(f"\n--- Results for {SELECTED_POLICY} ---")
    print(f"Total simulation time: {SIMULATION_TIME} seconds")
    print(f"Total requests completed: {stats.completed_requests}")
    print(f"Throughput: {throughput:.4f} requests/sec")
    print(f"Average response time: {avg_response_time:.4f} seconds")

if __name__ == "__main__":
    run_simulation()