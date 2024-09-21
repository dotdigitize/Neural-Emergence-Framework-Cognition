# File: llm_integration.py

import threading
import time
import random
from global_workspace import GlobalWorkspace
import sqlite3
import logging
import requests
from typing import Optional, List, Dict, Any
import json

logging.basicConfig(level=logging.DEBUG)

class LLMIntegration:
    """
    This class handles the integration of a Local Large Language Model (LLM) into the system.
    It interacts with the GlobalWorkspace, processes data from other components, and generates
    responses or actions based on LLM outputs. This version also includes functionality to
    dynamically create and manage agents based on system outputs.

    Key Features:
    - Connects to a local LLM API or interface.
    - Generates prompts based on data from the GlobalWorkspace.
    - Parses LLM outputs and translates them into system actions.
    - Logs interactions for analysis and debugging.
    - Implements agent spawning based on LLM interactions or workspace commands.
    - Implements error handling and retries for robustness.
    """

    def __init__(self, global_workspace: GlobalWorkspace):
        self.global_workspace = global_workspace
        self.setup_database()
        self.llm_api_url = "http://localhost:XXXX"  # Placeholder URL for local LLM API
        self.api_timeout = 10  # Timeout for API requests
        self.retry_attempts = 3  # Number of retry attempts for API calls
        self.retry_delay = 2  # Delay between retries
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        self.prompt_templates = self.load_prompt_templates()
        self.context_window = []  # For maintaining conversation context
        self.max_context_length = 5  # Max number of previous exchanges to keep
        self.logging_enabled = True
        self.agents = []  # List to manage dynamic agents

    def setup_database(self):
        self.conn = sqlite3.connect('llm_interactions.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS llm_interactions (
                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                               time REAL,
                               input TEXT,
                               output TEXT,
                               metadata TEXT)''')
        self.conn.commit()

    def load_prompt_templates(self) -> Dict[str, str]:
        """
        Load prompt templates from a file or define them here.
        """
        templates = {
            'default': "Based on the neural activations and entropy, provide guidance.",
            'modulation': "Neural activations are {activations}. The average entropy is {entropy}. What should be the modulation command?",
        }
        logging.debug("Prompt templates loaded.")
        return templates

    def format_prompt(self, template_name: str, data: Dict[str, Any]) -> str:
        """
        Format the prompt using the specified template and data.
        """
        template = self.prompt_templates.get(template_name)
        if not template:
            template = self.prompt_templates['default']
        prompt = template.format(**data)
        return prompt

    def ollama_generate_response(self, prompt: str) -> Optional[str]:
        """
        Send the prompt to the LLM API and return the response.
        """
        for attempt in range(self.retry_attempts):
            try:
                payload = {'prompt': prompt}
                headers = {'Content-Type': 'application/json'}
                response = requests.post(self.llm_api_url, json=payload, headers=headers, timeout=self.api_timeout)
                if response.status_code == 200:
                    data = response.json()
                    llm_output = data.get('text', '').strip()
                    logging.debug(f"LLM response received: {llm_output}")
                    return llm_output
                else:
                    logging.error(f"LLM API error: Status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                logging.error(f"LLM API request failed: {e}")
                time.sleep(self.retry_delay)
        logging.error("Failed to get response from LLM after retries.")
        return None

    def parse_llm_output(self, output: str) -> Optional[str]:
        """
        Parse the LLM output to extract actionable commands.
        """
        if 'increase activity' in output.lower():
            return 'increase'
        elif 'decrease activity' in output.lower():
            return 'decrease'
        elif 'maintain activity' in output.lower():
            return 'maintain'
        elif 'spawn new agent' in output.lower():
            # Trigger to create a new agent
            self.spawn_new_agent("Agent X", "Strategist", "A strategic AI agent", "Methodical", "llama3.1:8b")
            return "spawned_agent"
        else:
            return None

    def run(self):
        while not self.stop_event.is_set():
            # Read from global workspace
            neural_activations = self.global_workspace.read('neural_activations')
            avg_entropy = self.global_workspace.read('avg_entropy')
            if neural_activations is not None and avg_entropy is not None:
                # Prepare prompt based on data
                data = {
                    'activations': neural_activations[:10],
                    'entropy': avg_entropy
                }
                prompt = self.format_prompt('modulation', data)
                # Append context
                context = '\n'.join(self.context_window)
                if context:
                    full_prompt = f"{context}\n{prompt}"
                else:
                    full_prompt = prompt

                # Send prompt to LLM
                output_text = self.ollama_generate_response(full_prompt)
                if output_text is not None:
                    # Update context
                    self.context_window.append(f"User: {prompt}")
                    self.context_window.append(f"LLM: {output_text}")
                    if len(self.context_window) > self.max_context_length * 2:
                        self.context_window = self.context_window[-self.max_context_length * 2:]
                    # Parse LLM output
                    command = self.parse_llm_output(output_text)
                    if command:
                        # Write command to global workspace
                        self.global_workspace.write('modulation', command)
                        # Log interaction
                        timestamp = time.time()
                        metadata = json.dumps({'context': context})
                        self.cursor.execute('INSERT INTO llm_interactions (time, input, output, metadata) VALUES (?, ?, ?, ?)',
                                            (timestamp, prompt, output_text, metadata))
                        self.conn.commit()
                        if self.logging_enabled:
                            logging.info(f"LLM interaction logged at {timestamp}")
                    else:
                        logging.warning("No actionable command found in LLM output.")
                else:
                    logging.error("LLM output is None.")
            else:
                logging.debug("Required data not available in global workspace.")
            time.sleep(1)
        self.conn.close()

    def spawn_new_agent(self, agent_name: str, role: str, backstory: str, style: str, model="llama3.1:8b"):
        """
        Dynamically creates a new agent and adds it to the list of active agents.
        """
        new_agent = {
            "name": agent_name,
            "role": role,
            "backstory": backstory,
            "style": style,
            "model": model
        }
        # Add the agent to the list of agents
        self.agents.append(new_agent)
        logging.info(f"New agent '{agent_name}' spawned with role '{role}' and style '{style}'.")

    def stop(self):
        """
        Stop the LLM integration thread and close resources.
        """
        self.stop_event.set()
        self.thread.join()
        self.conn.close()
        logging.info("LLMIntegration stopped.")

    def set_logging(self, enabled: bool):
        """
        Enable or disable logging.
        """
        self.logging_enabled = enabled

    def update_prompt_templates(self, templates: Dict[str, str]):
        """
        Update the prompt templates.
        """
        self.prompt_templates.update(templates)
        logging.info("Prompt templates updated.")

    def set_context_length(self, length: int):
        """
        Set the maximum context length.
        """
        self.max_context_length = length
        logging.info(f"Context length set to {length}")

    def clear_context(self):
        """
        Clear the conversation context.
        """
        self.context_window.clear()
        logging.info("Context window cleared.")

    def send_custom_prompt(self, prompt: str) -> Optional[str]:
        """
        Send a custom prompt to the LLM and return the response.
        """
        output_text = self.ollama_generate_response(prompt)
        if output_text is not None:
            # Update context
            self.context_window.append(f"User: {prompt}")
            self.context_window.append(f"LLM: {output_text}")
            if len(self.context_window) > self.max_context_length * 2:
                self.context_window = self.context_window[-self.max_context_length * 2:]
            # Log interaction
            timestamp = time.time()
            metadata = json.dumps({'custom_prompt': True})
            self.cursor.execute('INSERT INTO llm_interactions (time, input, output, metadata) VALUES (?, ?, ?, ?)',
                                (timestamp, prompt, output_text, metadata))
            self.conn.commit()
            if self.logging_enabled:
                logging.info(f"Custom LLM interaction logged at {timestamp}")
            return output_text
        else:
            logging.error("LLM output is None.")
            return None

    def handle_commands(self):
        """
        Check the global workspace for commands directed at the LLMIntegration component.
        """
        while not self.stop_event.is_set():
            command = self.global_workspace.read('llm_command')
            if command:
                if command == 'clear_context':
                    self.clear_context()
                elif command == 'disable_logging':
                    self.set_logging(False)
                elif command == 'enable_logging':
                    self.set_logging(True)
                # Clear the command after processing
                self.global_workspace.write('llm_command', None)
                logging.info(f"Processed command: {command}")
            time.sleep(1)

    def start(self):
        """
        Start the LLM integration component.
        """
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        self.command_thread = threading.Thread(target=self.handle_commands)
        self.command_thread.daemon = True
        self.command_thread.start()
        logging.info("LLMIntegration started.")

    def generate_modulation_command(self):
        """
        Generate a modulation command based on the current state.
        """
        neural_activations = self.global_workspace.read('neural_activations')
        avg_entropy = self.global_workspace.read('avg_entropy')
        if neural_activations is not None and avg_entropy is not None:
            # Prepare prompt based on data
            data = {
                'activations': neural_activations[:10],
                'entropy': avg_entropy
            }
            prompt = self.format_prompt('modulation', data)
            # Send prompt to LLM
            output_text = self.ollama_generate_response(prompt)
            if output_text is not None:
                command = self.parse_llm_output(output_text)
                if command:
                    # Write command to global workspace
                    self.global_workspace.write('modulation', command)
                    # Log interaction
                    timestamp = time.time()
                    metadata = json.dumps({'function': 'generate_modulation_command'})
                    self.cursor.execute('INSERT INTO llm_interactions (time, input, output, metadata) VALUES (?, ?, ?, ?)',
                                        (timestamp, prompt, output_text, metadata))
                    self.conn.commit()
                    if self.logging_enabled:
                        logging.info(f"Modulation command generated at {timestamp}")
                    return command
                else:
                    logging.warning("No actionable command found in LLM output.")
            else:
                logging.error("LLM output is None.")
        else:
            logging.debug("Required data not available in global workspace.")
        return None

    def periodic_task(self, interval: int):
        """
        Perform a periodic task, such as generating modulation commands at regular intervals.
        """
        while not self.stop_event.is_set():
            self.generate_modulation_command()
            time.sleep(interval)

    def start_periodic_task(self, interval: int = 10):
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

    def update_llm_api_url(self, url: str):
        """
        Update the URL of the LLM API endpoint.
        """
        self.llm_api_url = url
        logging.info(f"LLM API URL updated to {url}")

    def set_retry_policy(self, attempts: int, delay: int):
        """
        Set the retry policy for API calls.
        """
        self.retry_attempts = attempts
        self.retry_delay = delay
        logging.info(f"Retry policy updated: attempts={attempts}, delay={delay}s")

    def get_interaction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve the interaction history from the database.
        """
        self.cursor.execute('SELECT time, input, output, metadata FROM llm_interactions ORDER BY id DESC LIMIT ?', (limit,))
        rows = self.cursor.fetchall()
        history = []
        for row in rows:
            entry = {
                'time': row[0],
                'input': row[1],
                'output': row[2],
                'metadata': json.loads(row[3])
            }
            history.append(entry)
        return history

    def export_interaction_history(self, filename: str = 'llm_interactions.json'):
        """
        Export the interaction history to a JSON file.
        """
        history = self.get_interaction_history(limit=1000)
        with open(filename, 'w') as f:
            json.dump(history, f, indent=4)
        logging.info(f"Interaction history exported to {filename}")

    def analyze_interactions(self):
        """
        Analyze interactions to extract insights or patterns.
        """
        history = self.get_interaction_history(limit=1000)
        total_interactions = len(history)
        commands = {'increase': 0, 'decrease': 0, 'maintain': 0, 'spawned_agent': 0}
        for entry in history:
            output = entry['output'].lower()
            command = self.parse_llm_output(output)
            if command:
                commands[command] += 1
        logging.info(f"Total interactions: {total_interactions}")
        logging.info(f"Command counts: {commands}")

    def backup_database(self, filename: str = 'llm_interactions_backup.db'):
        """
        Backup the SQLite database.
        """
        self.conn.close()
        import shutil
        shutil.copy('llm_interactions.db', filename)
        self.conn = sqlite3.connect('llm_interactions.db')
        self.cursor = self.conn.cursor()
        logging.info(f"Database backed up to {filename}")

    def restore_database(self, filename: str = 'llm_interactions_backup.db'):
        """
        Restore the SQLite database from a backup.
        """
        self.conn.close()
        import shutil
        shutil.copy(filename, 'llm_interactions.db')
        self.conn = sqlite3.connect('llm_interactions.db')
        self.cursor = self.conn.cursor()
        logging.info(f"Database restored from {filename}")

    def handle_shutdown(self):
        """
        Handle shutdown procedures gracefully.
        """
        self.stop()
        self.export_interaction_history()
        self.backup_database()
        logging.info("LLMIntegration shutdown complete.")
