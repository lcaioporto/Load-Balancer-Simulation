import numpy as np
import matplotlib.pyplot as plt
import load_balancer_simulation as lbs
import os

POLICIES = ['RANDOM', 'ROUND_ROBIN', 'SHORTEST_QUEUE']
# Define the range of request arrival rates to test
ARRIVAL_RATES = np.arange(2, 22, 2)
# Simulation time for each individual run
SIM_TIME_PER_RUN = 2000
NUM_SERVERS_FOR_PLOT = 3
# Set to False to compare policies under constant flow
SIMULATE_BURST_PHASE_FOR_PLOT = False
# Output directory
OUTPUT_DIR = 'stats'


def run_experiment():
    """
    Runs the simulation for each policy across all specified arrival rates
    and collects the results.
    """
    results = {policy: {'avg_response_times': [], 'throughputs': []} for policy in POLICIES}

    print("--- Starting Experiment ---")
    print(f"Testing Policies: {POLICIES}")
    print(f"Testing Arrival Rates: {ARRIVAL_RATES}\n")

    for policy in POLICIES:
        print(f"--- Running simulations for '{policy}' policy ---")
        for rate in ARRIVAL_RATES:
            print(f"  Testing arrival rate: {rate} req/s")
            avg_time, throughput = lbs.run_simulation(
                policy=policy,
                arrival_rate=rate,
                sim_time=SIM_TIME_PER_RUN,
                num_servers=NUM_SERVERS_FOR_PLOT,
                burst_phase=SIMULATE_BURST_PHASE_FOR_PLOT,
                show_logs=False
            )
            results[policy]['avg_response_times'].append(avg_time)
            results[policy]['throughputs'].append(throughput)
    
    print("\n--- Experiment Finished ---")
    return results

def plot_and_save_results(results):
    """
    Generates, saves, and calculates the area under the curve for two plots:
    1. Average Response Time vs. Arrival Rate
    2. Throughput vs. Arrival Rate
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Graphic 1: Average Response Time vs. Arrival Rate
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    # Include relative areas
    areas = [np.trapezoid(results[policy]['avg_response_times'], ARRIVAL_RATES) for policy in POLICIES]
    max_area = max(areas)
    for idx, policy in enumerate(POLICIES):
        area = areas[idx]
        rel = max_area / area if area != 0 else 0
        label_text = f'{policy} (Relative Area: {rel:.4f})'
        ax1.plot(ARRIVAL_RATES, results[policy]['avg_response_times'], marker='o', linestyle='-', label=label_text)
    
    ax1.set_title('Load Balancer Performance: Average Response Time', fontsize=16, pad=20)
    ax1.set_xlabel('Request Arrival Rate (requests/sec)', fontsize=12)
    ax1.set_ylabel('Average Response Time (seconds)', fontsize=12)
    legend = ax1.legend(title='Policy', fontsize=10)
    plt.setp(legend.get_title(), fontsize='11')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save Fig 1
    fig1_path = os.path.join(OUTPUT_DIR, 'average_response_time.png')
    fig1.savefig(fig1_path, bbox_inches='tight', dpi=150)
    plt.close(fig1)
    print(f"Plot saved successfully to: {fig1_path}")

    # Graphic 2: Throughput vs. Arrival Rate
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    # Include relative areas
    areas = [np.trapezoid(results[policy]['throughputs'], ARRIVAL_RATES) for policy in POLICIES]
    max_area = max(areas)
    for idx, policy in enumerate(POLICIES):
        area = areas[idx]
        rel = area / max_area if max_area != 0 else 0
        label_text = f'{policy} (Relative Area: {rel:.4f})'
        ax2.plot(ARRIVAL_RATES, results[policy]['throughputs'], marker='s', linestyle='--', label=label_text)

    ax2.set_title('Load Balancer Performance: Throughput', fontsize=16, pad=20)
    ax2.set_xlabel('Request Arrival Rate (requests/sec)', fontsize=12)
    ax2.set_ylabel('Throughput (completed requests/sec)', fontsize=12)
    legend = ax2.legend(title='Policy', fontsize=10)
    plt.setp(legend.get_title(), fontsize='11')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save Fig 2
    fig2_path = os.path.join(OUTPUT_DIR, 'throughput.png')
    fig2.savefig(fig2_path, bbox_inches='tight', dpi=150)
    plt.close(fig2)
    print(f"Plot saved successfully to: {fig2_path}")

if __name__ == "__main__":
    experiment_results = run_experiment()
    plot_and_save_results(experiment_results)