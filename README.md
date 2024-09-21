
# Neuromorphic System Framework

## Vision

This project envisions a comprehensive neuromorphic system that integrates neuromorphic network simulations, evolutionary dynamics, and revolutionary local language model (LLM) agents that are dynamically created based on data and emergent algorithms. These agents evolve in real time as the system processes information, adapting and expanding as new patterns and behaviors emerge. By combining spiking neural networks with cognitive processing and large-scale simulations, the framework explores novel emergent behaviors and is designed to function on modern hardware.

As the system evolves and demonstrates increasingly sophisticated cognitive abilities, a unique cryptocurrency reward system is introduced. Each time an agent exhibits advanced cognition, invents something, or makes a meaningful contribution, the system rewards it with its own cryptocurrency. This currency symbolizes the agent's success and can be used to track its progress or fuel further system development. This innovative reward mechanism contributes to the advancement of cognitive AI systems, empowering agents to manage resources autonomously and drive continual growth and innovation.
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

**Code Sample:**

```python
class NeuromorphicNetwork:
    def __init__(self, global_workspace: GlobalWorkspace):
        self.global_workspace = global_workspace
        self.setup_database()
        self.build_model()
        self.run_duration = 1000 * ms  # Simulation run time
        self.time_step = 0.1 * ms      # Simulation time step

    def setup_database(self):
        self.conn = sqlite3.connect('neuro_activity.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS neural_activity (
                               time REAL,
                               neuron_id INTEGER,
                               variable TEXT,
                               value REAL)''')
        self.conn.commit()

    def build_model(self):
        self.N = 1000  # Number of neurons
        eqs = '''
        dv/dt = (I_syn + I_inj - v) / tau : volt (unless refractory)
        I_syn = g_syn * (E_syn - v) : amp
        dg_syn/dt = -g_syn / tau_syn : siemens
        I_inj : amp
        tau : second
        E_syn : volt
        '''
        self.neurons = b2.NeuronGroup(self.N, eqs,
                                      threshold='v > -50*mV',
                                      reset='v = -65*mV',
                                      refractory=5*ms,
                                      method='euler')
        self.synapses = b2.Synapses(self.neurons, self.neurons,
                                    '''
                                    w : siemens
                                    dpre/dt = -pre / tau_pre : 1 (event-driven)
                                    dpost/dt = -post / tau_post : 1 (event-driven)
                                    ''')
        self.synapses.connect(p=0.1)
        self.synapses.w = 0.5 * nS  # Initial synaptic weight
```

This neuromorphic network interacts with the global workspace to store and share neural states, including voltages and synaptic weights, and processes data for further analysis.


## 2. Global Workspace

The global workspace acts as the central hub for communication between components. Each component can write its data to the global workspace and read from it, allowing a shared memory architecture to emerge.

**Purpose**:
- Facilitate communication between independent system components.
- Store shared states, such as neural activations, entropy values, and modulation instructions.

```python
# File: global_workspace.py

import threading
import time
import logging
import json
from collections import defaultdict
from typing import Any, Dict, Optional, Callable, List, Set

logging.basicConfig(level=logging.DEBUG)

class GlobalWorkspace:
    def __init__(self):
        self._shared_memory: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._event_conditions: Dict[str, threading.Condition] = defaultdict(threading.Condition)
        self._data_timestamps: Dict[str, float] = {}
        self._data_versions: Dict[str, int] = {}
        self._data_expirations: Dict[str, float] = {}
        self._max_data_age: float = float('inf')
        self._cleanup_interval: float = 60.0
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        self._logging_enabled: bool = True
        self._persistence_file: Optional[str] = None
        self._persistence_interval: float = 300.0
        self._persistence_thread = threading.Thread(target=self._persistence_loop, daemon=True)
        self._persistence_thread.start()
        self._observers: Set[Callable[[str, Any], None]] = set()
        self._stop_event = threading.Event()

    def write(self, key: str, value: Any, expiration: Optional[float] = None):
        with self._lock:
            self._shared_memory[key] = value
            timestamp = time.time()
            self._data_timestamps[key] = timestamp
            self._data_versions[key] = self._data_versions.get(key, 0) + 1
            if expiration is not None:
                self._data_expirations[key] = timestamp + expiration
            elif self._max_data_age != float('inf'):
                self._data_expirations[key] = timestamp + self._max_data_age
            else:
                self._data_expirations.pop(key, None)

            if self._logging_enabled:
                logging.debug(f"Write: key={key}, value={value}, timestamp={timestamp}")

            for callback in self._subscribers.get(key, []):
                threading.Thread(target=callback, args=(value,), daemon=True).start()

            condition = self._event_conditions.get(key)
            if condition:
                with condition:
                    condition.notify_all()

            for observer in self._observers:
                threading.Thread(target=observer, args=(key, value), daemon=True).start()

    def read(self, key: str, default: Any = None) -> Any:
        with self._lock:
            value = self._shared_memory.get(key, default)
            if self._logging_enabled:
                logging.debug(f"Read: key={key}, value={value}")
            return value

    def wait_for(self, key: str, timeout: Optional[float] = None) -> Any:
        condition = self._event_conditions.setdefault(key, threading.Condition())
        with condition:
            if key in self._shared_memory:
                return self._shared_memory[key]
            notified = condition.wait(timeout=timeout)
            if not notified:
                raise TimeoutError(f"Timeout waiting for key '{key}'")
            return self._shared_memory[key]

    def subscribe(self, key: str, callback: Callable[[Any], None]):
        with self._lock:
            self._subscribers[key].append(callback)
            if self._logging_enabled:
                logging.debug(f"Subscribed to key: {key}")

    def unsubscribe(self, key: str, callback: Callable[[Any], None]):
        with self._lock:
            if callback in self._subscribers.get(key, []):
                self._subscribers[key].remove(callback)
                if self._logging_enabled:
                    logging.debug(f"Unsubscribed from key: {key}")

    # Additional methods omitted for brevity
```



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

