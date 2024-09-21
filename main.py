import threading
from neuromorphic_network import NeuromorphicNetwork
from global_workspace import GlobalWorkspace
from evolutionary_dynamics import EvolutionaryDynamics
from llm_integration import LLMIntegration
from agent import AgentManager

def main():
    # Initialize components
    global_workspace = GlobalWorkspace()
    neuromorphic_network = NeuromorphicNetwork(global_workspace)
    evolutionary_dynamics = EvolutionaryDynamics(global_workspace)
    llm_integration = LLMIntegration(global_workspace)
    agent_manager = AgentManager(global_workspace)

    # Start threads for each component
    threads = []

    # Neuromorphic Network Thread
    t1 = threading.Thread(target=neuromorphic_network.run)
    threads.append(t1)

    # Evolutionary Dynamics Thread
    t2 = threading.Thread(target=evolutionary_dynamics.run)
    threads.append(t2)

    # LLM Integration Thread
    t3 = threading.Thread(target=llm_integration.run)
    threads.append(t3)

    # Agent Manager Thread
    t4 = threading.Thread(target=agent_manager.run)
    threads.append(t4)

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
