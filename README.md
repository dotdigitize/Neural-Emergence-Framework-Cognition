
# Neuromorphic System Framework

## Vision

This project envisions a comprehensive neuromorphic system that integrates neuromorphic network simulations, evolutionary dynamics, local language models (LLMs), and agents into a cohesive architecture. The purpose is to explore emergent behaviors that may arise from combining spiking neural networks, cognitive processing, and large-scale simulations. The framework is designed to function on modern hardware and inspired by theoretical insights from leading AI scientists.

### Purpose

The primary goal of this system is to simulate human-like cognition and behavior in a controlled computational environment. The design incorporates elements of spiking neural networks (inspired by biology), global workspaces (for inter-component communication), and evolutionary dynamics (for self-improving algorithms). Additionally, LLMs are used for higher-level decision-making, simulating aspects of language and abstract thinking.

## System Components

The system is built around several core components, each represented by a Python module. These components work together to form an intelligent, evolving system.

### 1. **Neuromorphic Network**
The neuromorphic network uses spiking neural networks (SNN) to simulate the brain's electrical activity. The neurons are modeled as leaky integrate-and-fire (LIF) neurons, and synapses follow spike-timing dependent plasticity (STDP) rules to adjust weights based on neural activity.

**Purpose:**
- Model low-level neural activity and synaptic learning.
- Log activity to a database for further analysis.
- Interface with the global workspace to share neural states, voltages, and firing rates.

### 2. **Global Workspace**
The global workspace acts as the central hub for communication between components. Each component can write its data to the global workspace and read from it, allowing a shared memory architecture to emerge.

**Purpose:**
- Facilitate communication between independent system components.
- Store shared states, such as neural activations, entropy values, and modulation instructions.

### 3. **Evolutionary Dynamics**
This module simulates evolutionary strategies by creating a "primordial soup" of random programs. Over time, these programs evolve by interacting, splitting, and merging. Entropy is used to measure the complexity of the programs, and the highest-performing programs are selected for further evolution.

**Purpose:**
- Simulate evolution of neural and cognitive processes.
- Use entropy to drive self-improvement and complexity.
- Log results in a database and interact with the global workspace.

### 4. **LLM Integration**
This component connects to a pre-trained large language model (LLM) such as in the example python file `llama3.1:8b` model from the `ollama` library. The LLM reads from the global workspace and makes decisions based on current neural activations and other metrics. These decisions are then fed back into the system to modulate neural parameters or agent behaviors.

**Purpose:**
- Leverage LLM for high-level cognitive processing.
- Simulate abstract reasoning and decision-making.
- Write decisions back to the global workspace for action by other components.

### 5. **Agent Manager**
Agents represent intelligent actors in the system, responsible for managing various actions and tasks. The agent network is a simple neural network that makes decisions based on inputs from the global workspace. These decisions influence both the neuromorphic network and the LLM.

**Purpose:**
- Simulate a higher-level decision-making process using neural networks.
- Interact with the global workspace and contribute to emergent behavior.
- Adapt to environmental stimuli and the LLM's outputs.

## System Design and Execution

The `main.py` file acts as the central orchestrator for the system. It initializes each of the components and runs them in parallel using Python threads. The global workspace serves as the communication bridge between these threads, ensuring data is synchronized across the system.

The system also features:
- **Signal handling for graceful shutdown:** Ensures all threads and components are stopped properly upon receiving a termination signal.
- **Central logging and monitoring:** Logs system actions to a database and console for debugging and tracking system health.
- **Modular and extensible design:** Allows for easy expansion, such as adding new agents, evolutionary algorithms, or neural models.

## Next Steps and Future Expansion

This framework is designed as a wheelbase for future work in neuromorphic systems and artificial intelligence. Future work will involve:

1. **Expanding Neuromorphic Network:**
   - Add additional neuron models and synaptic rules.
   - Simulate more complex brain structures (e.g., hippocampus, cortex).

2. **Evolving Cognitive Agents:**
   - Implement more sophisticated agents with reinforcement learning.
   - Allow agents to evolve and adapt to the system dynamically.

3. **LLM Enhancements:**
   - Integrate more modalities such as vision and sound.
   - Use the LLM for complex reasoning tasks and long-term memory management.

4. **System Optimizations:**
   - Implement GPU-based optimizations for large-scale simulations.
   - Integrate real-time resource management for efficient multi-threading.

## Cryptocurrency-Based Reward System

In future expansions, we plan to introduce a cryptocurrency-based reward system. Every time an agent exhibits advanced cognition, successfully invents something, or completes a significant task, the system will reward the agent with its own cryptocurrency. This currency will be tied to the AI’s autonomous entity, represented by an LLC or corporation, giving the AI a form of self-governance and financial reward for its contributions. This reward structure incentivizes continuous development and performance, both in virtual environments and, potentially, real-world applications.

The cryptocurrency could be earned through:
- **Inventive Tasks:** Rewards for coming up with creative solutions or discovering new ideas.
- **Advanced Learning:** Rewards for improving cognitive abilities or successfully completing complex tasks.
- **Collaboration:** Cryptocurrency could also be exchanged between different AI agents or even human counterparts, fostering a collaborative ecosystem.

This innovative approach provides a tangible metric for evaluating and rewarding AI's cognitive abilities while creating a novel intersection between AI research and blockchain technologies.

