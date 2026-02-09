# Load Balancer Simulation

This project utilizes *SimPy* to simulate a **distributed system** and evaluate the performance of different load balancing algorithms. The simulation models an enviornment of incoming HTTP-like requests and distributes them across a cluster of servers with different techniques, which are thereby evaluated utilizing measured performance metrics such as *Throughput* and *Average Response Time*.

The project was developed as part of the Distributed Systems course at the [University of Campinas](https://unicamp.br/en/) by the pair:
- [Caio Porto](https://github.com/lcaioporto)
- [Joao Guimaraes](https://github.com/JoaoMKlGui)

# Implemented Policies

We implemented and compared three standard load balancing strategies:
- **Random**: Selects a server at random for each new request. This is the simplest approach with zero state overhead, but can lead to imbalances where one server is overloaded while others are idle.
- **Round-Robin**: Distributes requests sequentially (Server 0 $\to$ Server 1 $\to$ ... $\to$ Server N). This ensures a fair distribution of request counts but does not account for the current load or processing time of existing tasks.
- **Shortest-Queue**: Directs new tasks to the server with the fewest active requests. This policy makes smater decisions, but it requires slightly more computational overhead to check the state of all server queues before forwarding an activity.

# Simulation Details
The project modelates the traffic of activities by generating requests with an exponentially distributed inter-arrival times (Poisson process) to simulate real-world traffic fluctuations.

Additionally, two different request types were created to simulate different types of tasks:

- **CPU_INTENSIVE**: Computationally heavy tasks (1–8 seconds).
- **IO_BOUND**: Slower activities, simulating tasks that require access to external resources (8–15 seconds).


To simplify the evaluation of results, we established that each server can process only one request at a time (FIFO queue).

# Main Results

We analyzed the system under varying loads (2 to 20 requests/second) and cluster sizes ($N=3$ vs $N=30$).

- **Small Scale** ($N=3$): Simple policies like Round-Robin performed surprisingly well. The overhead required for Shortest-Queue to poll server states actually slowed its performance slightly compared to the "blind" but fast assignment of Round-Robin.
- **Large Scale** ($N=30$): As the system scaled, Shortest-Queue proved to be superior. The benefit of intelligent load distribution compensated the cost of state checking, resulting in significantly better throughput and lower response times compared to Random or Round-Robin.

# How to Run
## Prerequisites
The project was recently tested with Python 3.14.2.

Initially, clone the repository in your machine:
```
git clone https://github.com/lcaioporto/Load-Balancer-Simulation.git
```

Then, install the project dependecies
```
pip install -r requirements.txt
```

## Running the Experiment
To run the full simulation and generate performance graphs (saved under `stats/`), run:
```
python plot_stats.py --nservers 3
```
You can freely modify the number of servers to test different scenarios.

## Running a Single Simulation
To watch the load balancer in action with real-time logs in the terminal, run:
```
python load_balancer_simulation.py --nservers 3
```

# Project Structure

- `load_balancer_simulation.py`: Core logic containing the Server, LoadBalancer, and Request classes.
- `plot_stats.py`: Automation script that varies the arrival rate, runs multiple iterations, and plots the comparison metrics.