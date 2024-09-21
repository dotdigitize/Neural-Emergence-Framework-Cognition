# File: agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading
from global_workspace import GlobalWorkspace
import ollama  # Import Ollama library for LLM integration, code here might not work as it for ollama, adjustments needed based on version
import random
import numpy as np
import sqlite3
import logging
from typing import List, Dict, Any
import re
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

class Agent(nn.Module):
    """
    LLM-based Agent that makes decisions based on inputs from the global workspace.
    Inspired by Greg Brockman's work on OpenAI Gym and RL.
    """
    def __init__(self, input_size, hidden_size, output_size, llm_model_name):
        super(Agent, self).__init__()
        # Neural network architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        # Initialize LLM
        self.llama = ollama.Model(llm_model_name)

    def forward(self, x):
        return self.model(x)

    def interact_with_llm(self, prompt):
        """
        Use the LLM to generate responses based on the current state.
        """
        try:
            llm_response = self.llama.generate(prompt)
            response_text = llm_response.strip()
            return response_text
        except Exception as e:
            logging.error(f"Error during LLM interaction: {e}")
            return None

class AgentManager:
    """
    Manages the Agent, interacts with the global workspace, handles LLM interactions,
    and updates the agent based on rewards.
    """
    def __init__(self, global_workspace):
        self.global_workspace = global_workspace
        self.agent = Agent(input_size=10, hidden_size=128, output_size=3, llm_model_name='llama3.1:8b')  # Example sizes
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent.to(self.device)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.setup_database()
        self.context_window = []
        self.max_context_length = 5  # Max number of previous exchanges to keep
        self.logging_enabled = True
        self.command_thread = threading.Thread(target=self.handle_commands)
        self.command_thread.daemon = True
        self.visualization_thread = threading.Thread(target=self.visualize_performance)
        self.visualization_thread.daemon = True
        self.performance_data = {
            'rewards': [],
            'timestamps': []
        }

    def setup_database(self):
        """
        Set up the SQLite database for logging agent interactions.
        """
        self.conn = sqlite3.connect('agent.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS agent_interactions (
                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                               time REAL,
                               state TEXT,
                               action INTEGER,
                               reward REAL,
                               llm_response TEXT)''')
        self.conn.commit()
        logging.debug("Database initialized.")

    def select_action(self, state):
        """
        Select an action based on the current state using the agent's policy.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        output = self.agent(state_tensor)
        action_prob = output.detach().cpu().numpy()
        action = np.random.choice(len(action_prob), p=action_prob)
        return action, output

    def run(self):
        """
        Main loop for the agent's operation.
        """
        self.performance_data['start_time'] = time.time()
        for step in range(1000):  # Expanded loop
            if self.stop_event.is_set():
                break
            # Read data from global workspace
            data = self.global_workspace.read('neural_activations')
            avg_entropy = self.global_workspace.read('avg_entropy')
            if data is not None and avg_entropy is not None:
                # Prepare state
                state = data[:10]  # Simplify to first 10 activations
                # Select action
                action, output = self.select_action(state)
                # Interact with environment (here, simulate reward)
                reward = self.compute_reward(action, avg_entropy)
                # Update agent
                target = torch.tensor([action], dtype=torch.long).to(self.device)
                loss = self.criterion(output.unsqueeze(0), target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Write decisions back to global workspace
                self.global_workspace.write('agent_decision', action)

                # Log interaction
                timestamp = time.time()
                self.cursor.execute('INSERT INTO agent_interactions (time, state, action, reward, llm_response) VALUES (?, ?, ?, ?, ?)',
                                    (timestamp, str(state), action, reward, None))
                self.conn.commit()

                # Update performance data
                self.performance_data['rewards'].append(reward)
                self.performance_data['timestamps'].append(timestamp - self.performance_data['start_time'])

                # Interact with LLM
                self.interact_with_llm(state, action, reward)

                if self.logging_enabled:
                    logging.info(f"Step {step}: Action {action}, Reward {reward}")

            else:
                logging.debug("Data not available in global workspace.")
            time.sleep(1)

        self.conn.close()
        logging.info("AgentManager run completed.")

    def compute_reward(self, action, avg_entropy):
        """
        Compute the reward for a given action.
        """
        # Placeholder reward function
        if action == 0:
            reward = -avg_entropy
        elif action == 1:
            reward = avg_entropy
        else:
            reward = 0
        return reward

    def interact_with_llm(self, state, action, reward):
        """
        Interact with the LLM to get feedback or suggestions based on the current state, action, and reward.
        """
        # Prepare prompt for LLM
        prompt = f"State: {state}\nAction taken: {action}\nReward: {reward}\nProvide feedback or suggestions:"
        # Append context
        context = '\n'.join(self.context_window)
        if context:
            full_prompt = f"{context}\n{prompt}"
        else:
            full_prompt = prompt

        # Generate response from LLM
        try:
            llm_response = self.agent.interact_with_llm(full_prompt)
            if llm_response:
                # Update context
                self.context_window.append(f"Agent: {prompt}")
                self.context_window.append(f"LLM: {llm_response}")
                if len(self.context_window) > self.max_context_length * 2:
                    self.context_window = self.context_window[-self.max_context_length * 2:]
                # Process LLM response
                self.process_llm_response(llm_response)
                # Update database with LLM response
                timestamp = time.time()
                self.cursor.execute('UPDATE agent_interactions SET llm_response = ? WHERE time = ?',
                                    (llm_response, timestamp))
                self.conn.commit()
                logging.debug(f"LLM interaction recorded at {timestamp}")
        except Exception as e:
            logging.error(f"Error during LLM interaction: {e}")

    def process_llm_response(self, response_text):
        """
        Process the LLM response to adjust the agent's parameters or strategies.
        """
        if 'increase exploration' in response_text.lower():
            self.adjust_exploration_rate(1.1)
        elif 'decrease exploration' in response_text.lower():
            self.adjust_exploration_rate(0.9)
        elif 'adjust learning rate' in response_text.lower():
            # Extract suggested learning rate
            suggested_lr = self.extract_learning_rate(response_text)
            if suggested_lr:
                self.set_learning_rate(suggested_lr)
        elif 'change strategy to' in response_text.lower():
            new_strategy = self.extract_strategy(response_text)
            if new_strategy:
                self.change_strategy(new_strategy)
        else:
            logging.info("No actionable instruction found in LLM response.")

    def adjust_exploration_rate(self, factor):
        """
        Adjust the agent's exploration rate.
        """
        # Placeholder: Adjust agent's exploration rate
        logging.info(f"Adjusting exploration rate by a factor of {factor}")
        # In a real implementation, adjust the epsilon in an epsilon-greedy policy

    def extract_learning_rate(self, text):
        """
        Extract learning rate value from the LLM response.
        """
        match = re.search(r'learning rate to (\d+\.\d+)', text)
        if match:
            lr = float(match.group(1))
            return lr
        return None

    def set_learning_rate(self, lr):
        """
        Set the agent's learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        logging.info(f"Learning rate set to {lr}")

    def extract_strategy(self, text):
        """
        Extract a new strategy from the LLM response.
        """
        match = re.search(r'change strategy to (\w+)', text)
        if match:
            strategy = match.group(1)
            return strategy
        return None

    def change_strategy(self, strategy):
        """
        Change the agent's strategy.
        """
        # Placeholder for changing the agent's strategy
        logging.info(f"Changing strategy to {strategy}")

    def stop(self):
        """
        Stop the agent's main loop and threads.
        """
        self.stop_event.set()
        self.thread.join()
        self.command_thread.join()
        self.visualization_thread.join()
        logging.info("AgentManager stopped.")

    def start(self):
        """
        Start the agent's main loop and threads.
        """
        self.thread.start()
        self.command_thread.start()
        self.visualization_thread.start()
        logging.info("AgentManager started.")

    def save_model(self, filename='agent_model.pth'):
        """
        Save the agent's model to a file.
        """
        torch.save(self.agent.state_dict(), filename)
        logging.info(f"Agent model saved to {filename}")

    def load_model(self, filename='agent_model.pth'):
        """
        Load the agent's model from a file.
        """
        self.agent.load_state_dict(torch.load(filename))
        self.agent.to(self.device)
        logging.info(f"Agent model loaded from {filename}")

    def get_action_distribution(self):
        """
        Retrieve the distribution of actions taken by the agent.
        """
        self.cursor.execute('SELECT action, COUNT(*) FROM agent_interactions GROUP BY action')
        rows = self.cursor.fetchall()
        action_counts = {row[0]: row[1] for row in rows}
        total_actions = sum(action_counts.values())
        action_distribution = {action: count / total_actions for action, count in action_counts.items()}
        logging.info(f"Action distribution: {action_distribution}")
        return action_distribution

    def analyze_performance(self):
        """
        Analyze the agent's performance over time.
        """
        rewards = self.performance_data['rewards']
        timestamps = self.performance_data['timestamps']
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            logging.info(f"Average reward: {avg_reward}")
            return avg_reward
        else:
            logging.info("No rewards to analyze.")
            return 0

    def update_from_global_workspace(self):
        """
        Update the agent's parameters based on commands from the global workspace.
        """
        command = self.global_workspace.read('agent_command')
        if command:
            if command == 'save_model':
                self.save_model()
            elif command == 'load_model':
                self.load_model()
            elif command == 'increase_learning_rate':
                self.adjust_learning_rate(1.1)
            elif command == 'decrease_learning_rate':
                self.adjust_learning_rate(0.9)
            elif command == 'export_performance':
                self.export_performance_data()
            # Clear the command after processing
            self.global_workspace.write('agent_command', None)
            logging.info(f"Processed command from global workspace: {command}")

    def adjust_learning_rate(self, factor):
        """
        Adjust the learning rate by a given factor.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
        logging.info(f"Learning rate adjusted by a factor of {factor}")

    def handle_commands(self):
        """
        Handle commands from the global workspace.
        """
        while not self.stop_event.is_set():
            self.update_from_global_workspace()
            time.sleep(1)

    def visualize_performance(self):
        """
        Visualize the agent's performance over time.
        """
        plt.ion()
        fig, ax = plt.subplots()
        while not self.stop_event.is_set():
            if self.performance_data['rewards']:
                ax.clear()
                ax.plot(self.performance_data['timestamps'], self.performance_data['rewards'], label='Rewards')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Reward')
                ax.set_title('Agent Performance Over Time')
                ax.legend()
                plt.pause(0.1)
            time.sleep(1)
        plt.ioff()
        plt.show()
        logging.info("Performance visualization stopped.")

    def export_performance_data(self, filename='agent_performance.npz'):
        """
        Export the agent's performance data to a file.
        """
        np.savez(filename, rewards=self.performance_data['rewards'], timestamps=self.performance_data['timestamps'])
        logging.info(f"Performance data exported to {filename}")

    def import_performance_data(self, filename='agent_performance.npz'):
        """
        Import performance data from a file.
        """
        data = np.load(filename)
        self.performance_data['rewards'] = data['rewards'].tolist()
        self.performance_data['timestamps'] = data['timestamps'].tolist()
        logging.info(f"Performance data imported from {filename}")

    def adjust_parameters(self):
        """
        Adjust internal parameters based on performance metrics.
        """
        avg_reward = self.analyze_performance()
        if avg_reward < 0:
            self.adjust_learning_rate(1.1)
            logging.info("Adjusted learning rate due to low performance.")

    def backup_database(self, filename='agent_backup.db'):
        """
        Backup the agent's database.
        """
        self.conn.close()
        import shutil
        shutil.copy('agent.db', filename)
        self.conn = sqlite3.connect('agent.db')
        self.cursor = self.conn.cursor()
        logging.info(f"Database backed up to {filename}")

    def restore_database(self, filename='agent_backup.db'):
        """
        Restore the agent's database from a backup.
        """
        self.conn.close()
        import shutil
        shutil.copy(filename, 'agent.db')
        self.conn = sqlite3.connect('agent.db')
        self.cursor = self.conn.cursor()
        logging.info(f"Database restored from {filename}")

    def monitor_health(self):
        """
        Monitor the health of the agent and take corrective actions if necessary.
        """
        while not self.stop_event.is_set():
            # Placeholder for health monitoring logic
            time.sleep(5)
            logging.debug("Agent health check performed.")

    def start_health_monitor(self):
        """
        Start the health monitoring thread.
        """
        self.health_thread = threading.Thread(target=self.monitor_health)
        self.health_thread.daemon = True
        self.health_thread.start()
        logging.info("Agent health monitor started.")

    def synchronize_with_external_systems(self):
        """
        Synchronize data with external systems or services.
        """
        # Placeholder for synchronization logic
        logging.info("Synchronizing with external systems.")

    def handle_shutdown(self):
        """
        Handle shutdown procedures gracefully.
        """
        self.stop()
        self.save_model()
        self.backup_database()
        self.export_performance_data()
        logging.info("AgentManager shutdown complete.")

    def train_custom_model(self):
        """
        Train a custom model for specific tasks or improvements.
        """
        # Placeholder for training logic
        logging.info("Training custom model.")

    def perform_system_check(self):
        """
        Perform a system check to ensure all components are functioning.
        """
        # Placeholder for system check logic
        logging.info("System check performed.")

    def send_feedback_to_llm(self, feedback):
        """
        Send feedback to the LLM to improve future interactions.
        """
        # Placeholder for feedback mechanism
        logging.info(f"Feedback sent to LLM: {feedback}")

    def simulate_user_input(self):
        """
        Simulate user input for testing purposes.
        """
        # Placeholder for simulation logic
        logging.info("Simulating user input.")

    def dynamic_parameter_adjustment(self):
        """
        Dynamically adjust agent parameters based on performance.
        """
        avg_reward = self.analyze_performance()
        if avg_reward > 0.5:
            self.adjust_learning_rate(0.9)
            logging.info("Decreased learning rate due to high performance.")
        elif avg_reward < -0.5:
            self.adjust_learning_rate(1.1)
            logging.info("Increased learning rate due to low performance.")

    def process_custom_commands(self, command):
        """
        Process custom commands received from the global workspace.
        """
        if command == 'export_performance':
            self.export_performance_data()
        elif command == 'analyze_performance':
            self.analyze_performance()
        elif command == 'visualize_performance':
            self.visualize_performance()
        else:
            logging.warning(f"Unknown custom command: {command}")

    def schedule_tasks(self):
        """
        Schedule tasks to be run at specific times or intervals.
        """
        # Placeholder for task scheduling logic
        logging.info("Scheduled tasks executed.")

    def periodic_task(self, interval=60):
        """
        Perform a periodic task, such as adjusting parameters.
        """
        while not self.stop_event.is_set():
            self.dynamic_parameter_adjustment()
            time.sleep(interval)

    def start_periodic_task(self, interval=60):
        """
        Start the periodic task in a separate thread.
        """
        self.periodic_thread = threading.Thread(target=self.periodic_task, args=(interval,))
        self.periodic_thread.daemon = True
        self.periodic_thread.start()
        logging.info(f"Periodic task started with interval {interval} seconds.")

    def stop_periodic_task(self):
        """
        Stop the periodic task.
        """
        self.stop_event.set()
        self.periodic_thread.join()
        logging.info("Periodic task stopped.")

    def summarize_performance(self):
        """
        Summarize the agent's performance data.
        """
        avg_reward = self.analyze_performance()
        total_steps = len(self.performance_data['rewards'])
        summary = f"Total steps: {total_steps}, Average reward: {avg_reward}"
        logging.info(f"Performance summary: {summary}")
        return summary

    def generate_report(self):
        """
        Generate a report based on interactions and performance.
        """
        report = self.summarize_performance()
        # Additional report generation logic can be added here
        logging.info("Report generated.")
        return report

    def send_report(self):
        """
        Send the generated report to a specified recipient or system.
        """
        report = self.generate_report()
        # Placeholder for sending the report
        logging.info(f"Report sent: {report}")

    def adaptive_learning(self):
        """
        Adapt the agent's learning strategy based on environment changes.
        """
        # Placeholder for adaptive learning logic
        logging.info("Adaptive learning executed.")

    def integrate_with_monitoring_tools(self):
        """
        Integrate with monitoring tools for better observability.
        """
        # Placeholder for integration logic
        logging.info("Integrated with monitoring tools.")

    def enable_logging(self, enabled):
        """
        Enable or disable logging.
        """
        self.logging_enabled = enabled
        if enabled:
            logging.info("Logging enabled.")
        else:
            logging.info("Logging disabled.")

    def reset_agent(self):
        """
        Reset the agent's parameters and state.
        """
        self.agent = Agent(input_size=10, hidden_size=128, output_size=3, llm_model_name='llama3.1:8b')
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001)
        self.agent.to(self.device)
        logging.info("Agent reset to initial state.")

    def manage_resources(self):
        """
        Manage system resources for optimal performance.
        """
        # Placeholder for resource management logic
        logging.info("Resources managed.")

    def custom_llm_prompt(self, custom_prompt):
        """
        Send a custom prompt to the LLM and process the response.
        """
        try:
            llm_response = self.agent.interact_with_llm(custom_prompt)
            if llm_response:
                self.process_llm_response(llm_response)
                logging.info(f"Custom LLM prompt processed: {custom_prompt}")
        except Exception as e:
            logging.error(f"Error during custom LLM interaction: {e}")

# Example usage:
# global_workspace = GlobalWorkspace()
# agent_manager = AgentManager(global_workspace)
# agent_manager.start()
# agent_manager.start_health_monitor()
# agent_manager.start_periodic_task()
# # Run for some time...
# agent_manager.handle_shutdown()
