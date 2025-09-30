import numpy as np
import matplotlib.pyplot as plt
import load_balancer_simulation as lbs
import os
import random

POLICIES = ['RANDOM', 'ROUND_ROBIN', 'SHORTEST_QUEUE']
ARRIVAL_RATES = np.arange(0.5, 22, 0.5)
SIM_TIME_PER_RUN = 200
NUM_SERVERS_FOR_PLOT = 3
OUTPUT_DIR = 'stats'

def run_experiment():
    """
    Executa simulações para cada política em todas as taxas de chegada
    """
    results = {p: {'avg_response_times': [], 'throughputs': []} for p in POLICIES}

    print("--- Starting Experiment ---")
    for rate in ARRIVAL_RATES:
        print(f"\n--- Testing Arrival Rate: {rate} req/s ---")
        random.seed(34)
        request_list_for_rate = lbs.pre_generate_requests(
            sim_time=SIM_TIME_PER_RUN,
            request_arrival_rate=rate
        )
        
        for policy in POLICIES:
            print(f"  Running policy: '{policy}'...")
            random.seed(34)
            avg_time, throughput = lbs.run_single_simulation(
                policy=policy,
                num_servers=NUM_SERVERS_FOR_PLOT,
                sim_time=SIM_TIME_PER_RUN,
                pre_generated_requests=request_list_for_rate,
                show_logs=False
            )
            results[policy]['avg_response_times'].append(avg_time)
            results[policy]['throughputs'].append(throughput)
    
    print("\n--- Experiment Finished ---")
    return results

def plot_and_save_results(results):
    """
    Gera e salva gráficos, incluindo as métricas de área sob a curva.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Plot 1: Average Response Time ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    # Para tempo de resposta, uma área menor é melhor.
    areas = {p: np.trapezoid(results[p]['avg_response_times'], ARRIVAL_RATES) for p in POLICIES}
    # Encontre a política com a área mínima (melhor desempenho)
    min_area = min(areas.values())

    for policy in POLICIES:
        area = areas[policy]
        # Desempenho em relação à melhor política (quanto menor, melhor)
        rel_perf = area / min_area if min_area > 0 else 0
        label_text = f'{policy} (Best Perf. Ratio: {rel_perf:.2f})'
        ax1.plot(ARRIVAL_RATES, results[policy]['avg_response_times'], marker='o', linestyle='-', label=label_text)
    
    ax1.set_title('Load Balancer: Average Response Time vs. Arrival Rate', fontsize=16)
    ax1.set_xlabel('Request Arrival Rate (requests/sec)', fontsize=12)
    ax1.set_ylabel('Average Response Time (seconds)', fontsize=12)
    ax1.legend(title='Policy')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig1_path = os.path.join(OUTPUT_DIR, 'average_response_time.png')
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)
    print(f"Plot saved to: {fig1_path}")

    # --- Plot 2: Throughput ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    # Para throughput, uma área maior é melhor
    areas = {p: np.trapezoid(results[p]['throughputs'], ARRIVAL_RATES) for p in POLICIES}
    # Encontre a política com a área máxima (melhor desempenho)
    max_area = max(areas.values())

    for policy in POLICIES:
        area = areas[policy]
        # Desempenho em relação à melhor política (quanto maior, melhor)
        rel_perf = area / max_area if max_area > 0 else 0
        label_text = f'{policy} (Best Perf. Ratio: {rel_perf:.2f})'
        ax2.plot(ARRIVAL_RATES, results[policy]['throughputs'], marker='s', linestyle='--', label=label_text)

    ax2.set_title('Load Balancer: Throughput vs. Arrival Rate', fontsize=16)
    ax2.set_xlabel('Request Arrival Rate (requests/sec)', fontsize=12)
    ax2.set_ylabel('Throughput (completed requests/sec)', fontsize=12)
    ax2.legend(title='Policy')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig2_path = os.path.join(OUTPUT_DIR, 'throughput.png')
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f"Plot saved to: {fig2_path}")

if __name__ == "__main__":
    experiment_results = run_experiment()
    plot_and_save_results(experiment_results)