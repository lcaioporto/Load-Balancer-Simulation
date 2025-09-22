"""
Script que usa SimPy pra simular e comparar umas políticas de load balancing.

A ideia é gerar uma lista de requests antes pra todo mundo usar a mesma,
aí a comparação fica justa.

Políticas comparadas:
- RANDOM: Manda a request pra um servidor aleatório.
- ROUND_ROBIN: Manda pra cada servidor em ordem, ciclicamente.
- SHORTEST_QUEUE: Manda pro servidor que tiver a menor fila.

No final, ele printa uma tabela com os resultados de cada uma.
"""
import simpy
import random
import pandas as pd

# --- Configs da Simulação ---
POLICIES_TO_TEST = ['RANDOM', 'ROUND_ROBIN', 'SHORTEST_QUEUE'] # Políticas que vamos testar.
NUM_SERVERS = 3                      # Quantidade de servidores disponíveis.
REQUEST_ARRIVAL_RATE = 10            # Média de requests chegando por segundo.
SIMULATION_TIME = 100                # Duração total da simulação em segundos.

# Tempo de processamento (min, max) pra cada tipo de request.
PROCESSING_TIMES = {
    'CPU_INTENSIVE': (1, 8),
    'IO_BOUND': (8, 15)
}

class Colors:
    """Classe pra deixar o print no terminal colorido."""
    RESET = '\033[0m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'

class Statistics:
    """Classe que guarda e calcula as métricas da simulação."""
    def __init__(self):
        """Inicia os contadores."""
        self.completed_requests = 0
        self.total_response_time = 0.0

    def record_completion(self, request):
        """
        Atualiza as stats quando uma request termina.
        Recebe a request que acabou de ser processada.
        """
        self.completed_requests += 1
        response_time = request.completion_time - request.arrival_time
        self.total_response_time += response_time

    def calculate_throughput(self, sim_time):
        """Calcula o throughput (requests por segundo)."""
        return self.completed_requests / sim_time if sim_time > 0 else 0

    def calculate_avg_response_time(self):
        """Calcula o tempo médio de resposta das requests."""
        return self.total_response_time / self.completed_requests if self.completed_requests > 0 else 0

class Request:
    """Representa uma request de um cliente no sistema."""
    def __init__(self, request_id, request_type, arrival_time, processing_time):
        """
        Inicia a request com suas infos.
        - request_id: ID único.
        - request_type: Tipo da request (ex: 'CPU_INTENSIVE').
        - arrival_time: Momento em que ela chegou na simulação.
        - processing_time: Tempo que ela leva pra ser processada.
        """
        self.id = request_id
        self.type = request_type
        self.arrival_time = arrival_time
        self.processing_time = processing_time
        self.completion_time = 0 # Vai ser preenchido quando ela terminar.

class Server:
    """Representa um servidor que processa uma request por vez."""
    def __init__(self, env, server_id, show_logs=True):
        """
        Inicia um Servidor.
        - env: Ambiente do SimPy.
        - server_id: ID único do servidor.
        """
        self.env = env
        self.id = server_id
        self.processor = simpy.Resource(env, capacity=1)
        self.queue_len = 0 # Tamanho da fila (contando a request em processamento).
        self.show_logs = show_logs

    def process_request(self, request):
        """Simula o tempo que leva pra processar a request."""
        # Pausa a execução pelo tempo de processamento da request.
        yield self.env.timeout(request.processing_time)
        request.completion_time = self.env.now
        if self.show_logs:
            print(f"{Colors.MAGENTA}[Time: {self.env.now:.4f}] Server {self.id}: Request {request.id} finalizado. "
                  f"Response Time: {request.completion_time - request.arrival_time:.4f}{Colors.RESET}")

class LoadBalancer:
    """Distribui as requests que chegam para os servidores."""
    def __init__(self, env, servers, stats, policy='ROUND_ROBIN', show_logs=True):
        """
        Inicia o LoadBalancer.
        - env: Ambiente do SimPy.
        - servers: Lista com os objetos de servidores.
        - stats: Objeto de estatísticas pra gravar os resultados.
        - policy: Política de balanceamento a ser usada.
        """
        self.env = env
        self.servers = servers
        self.stats = stats
        self.policy = policy
        self.rr_counter = 0 # Contador pro Round Robin.
        self.show_logs = show_logs

    def handle_request(self, request):
        """Escolhe um servidor com base na política e manda a request pra ele."""
        # Escolhe o servidor de acordo com a política.
        if self.policy == 'RANDOM':
            selected_server = random.choice(self.servers)
        elif self.policy == 'ROUND_ROBIN':
            selected_server = self.servers[self.rr_counter]
            self.rr_counter = (self.rr_counter + 1) % len(self.servers)
        elif self.policy == 'SHORTEST_QUEUE':
            selected_server = min(self.servers, key=lambda s: s.queue_len)
        else:
            raise ValueError(f"Política desconhecida: {self.policy}")

        # Aumenta o contador da fila e inicia o processo do servidor.
        selected_server.queue_len += 1
        if self.show_logs:
            print(f"{Colors.BLUE}[Time: {self.env.now:.4f}] Load Balancer: Request {request.id} enviado para o Server {selected_server.id}. "
                  f"Tamanho da fila: {selected_server.queue_len}{Colors.RESET}")

        self.env.process(self.server_worker(selected_server, request))

    def server_worker(self, server, request):
        """
        Processo do SimPy que cuida do ciclo de vida da request no servidor.
        Ele espera o servidor ficar livre, processa a request e depois salva os resultados.
        """
        with server.processor.request() as req:
            # Espera o servidor ficar disponível.
            yield req
            # Quando começa a processar, o item "sai da fila de espera".
            server.queue_len -= 1
            # Inicia o processamento de fato.
            yield self.env.process(server.process_request(request))
            # Grava as estatísticas quando termina.
            self.stats.record_completion(request)

def pre_generate_requests(sim_time, request_arrival_rate):
    """
    Cria a lista de requests ANTES da simulação começar.
    Isso garante que todas as políticas usem a mesma lista, pra comparação ser justa.
    """
    requests = []
    current_time = 0.0
    # Gera requests até o tempo final da simulação.
    while current_time < sim_time:
        # Calcula quando a próxima request vai chegar.
        inter_arrival_time = random.expovariate(request_arrival_rate)
        current_time += inter_arrival_time
        
        if current_time < sim_time:
            # Sorteia o tipo e o tempo de processamento da request.
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
    Processo do SimPy que vai 'injetando' as requests da lista na simulação.
    Ele lê a lista e espera o tempo certo pra criar e enviar a request pro LB.
    """
    for i, req_info in enumerate(pre_generated_requests):
        # Espera até o tempo de chegada da request.
        yield env.timeout(max(0, req_info['arrival_time'] - env.now))
        
        # Cria o objeto da Request.
        req = Request(
            request_id=i, 
            request_type=req_info['type'],
            arrival_time=env.now,
            processing_time=req_info['processing_time']
        )
        if show_logs: print(f"{Colors.CYAN}[Time: {env.now:.4f}] Generator: Novo request {req.type} com ID {req.id} foi gerado.{Colors.RESET}")
        # Entrega a request pro load balancer.
        load_balancer.handle_request(req)

def run_single_simulation(policy, num_servers, sim_time, pre_generated_requests, show_logs=True):
    """
    Configura e roda a simulação completa pra UMA política específica.
    Retorna o tempo médio de resposta e o throughput.
    """
    if show_logs:
        print(f"{Colors.BOLD}--- Rodando simulação para a política '{policy}' ---{Colors.RESET}")
    
    # Prepara o ambiente da simulação.
    stats = Statistics()
    env = simpy.Environment()
    servers = [Server(env, i, show_logs=show_logs) for i in range(num_servers)]
    load_balancer = LoadBalancer(env, servers, stats, policy=policy, show_logs=show_logs)
    
    # Inicia o processo que injeta as requests.
    env.process(request_injector(env, load_balancer, pre_generated_requests, show_logs=show_logs))
    
    # Roda a simulação.
    env.run(until=sim_time) 
    
    # Calcula os resultados finais.
    throughput = stats.calculate_throughput(sim_time)
    avg_response_time = stats.calculate_avg_response_time()

    if show_logs:
        print(f"Finalizado. Tempo Médio: {avg_response_time:.4f}, Throughput: {throughput:.4f}")

    return avg_response_time, throughput

if __name__ == "__main__":
    # Define uma seed pro random pra garantir que os resultados sejam sempre os mesmos.
    # Assim, a lista de requests gerada vai ser sempre idêntica.
    random.seed(42) 
    
    print("Gerando lista de requests para a simulação...")
    master_request_list = pre_generate_requests(SIMULATION_TIME, REQUEST_ARRIVAL_RATE)
    print(f"{len(master_request_list)} requests geradas.")

    all_results = []
    
    # Roda uma simulação para cada política da lista.
    for policy in POLICIES_TO_TEST:
        # Resetar a seed aqui garante que a política 'RANDOM' se comporte igual
        # toda vez que rodamos o script, pra ajudar na reprodutibilidade.
        random.seed(42) 
        
        avg_time, throughput = run_single_simulation(
            policy=policy,
            num_servers=NUM_SERVERS,
            sim_time=SIMULATION_TIME,
            pre_generated_requests=master_request_list,
            show_logs=True
        )
        # Guarda o resultado pra mostrar no final.
        all_results.append({'Policy': policy, 'Avg Response Time (s)': avg_time, 'Throughput (req/s)': throughput})

    # Usa pandas pra printar a tabela de resultados bonitinha.
    results_df = pd.DataFrame(all_results)
    
    # Mostra a tabela final comparando tudo.
    print(f"\n{Colors.BOLD}{Colors.GREEN}--- Resultados Finais Comparativos ---{Colors.RESET}")
    print(results_df.to_string(index=False))