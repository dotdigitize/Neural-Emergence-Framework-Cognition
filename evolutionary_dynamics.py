# File: evolutionary_dynamics.py

import random
import threading
import time
import math
import zlib
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from global_workspace import GlobalWorkspace
from typing import List, Dict, Tuple, Any
import logging

logging.basicConfig(level=logging.DEBUG)

class EvolutionaryDynamics:
    """
    This class simulates an evolutionary system inspired by genetic algorithms,
    artificial life, and emergent behavior theories. It integrates with the GlobalWorkspace
    to share data with other system components.

    Key Features:
    - Simulates a population of programs (organisms) that evolve over time.
    - Uses concepts like mutation, crossover, and selection to evolve programs.
    - Calculates entropy and complexity measures to guide evolution.
    - Stores data in a SQLite database for persistence and analysis.
    - Visualizes evolutionary metrics in real-time.
    """

    def __init__(self, global_workspace: GlobalWorkspace):
        self.global_workspace = global_workspace
        self.SOUP_SIZE = 1000  # Increased size for complexity
        self.PROGRAM_LENGTH = 100  # Length of each program
        self.MAX_STEPS = 1000  # Max execution steps for programs
        self.THRESHOLD = 0.1  # Threshold for certain conditions
        self.MUTATION_RATE = 0.01  # Mutation rate for genetic operations
        self.CROSSOVER_RATE = 0.7  # Crossover rate for breeding
        self.EPOCHS = 100  # Number of evolutionary epochs
        self.soup = self.create_soup(self.SOUP_SIZE, self.PROGRAM_LENGTH)
        self.setup_database()
        self.current_epoch = 0
        self.lock = threading.Lock()
        self.statistics = {
            'avg_entropy': [],
            'max_entropy': [],
            'min_entropy': [],
            'diversity_index': [],
            'fitness_scores': []
        }
        self.stop_event = threading.Event()
        self.visualization_thread = threading.Thread(target=self.visualize_statistics)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()

    def setup_database(self):
        self.conn = sqlite3.connect('evolution.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS interactions (
                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                               epoch INTEGER,
                               program_a TEXT, program_b TEXT,
                               offspring_a TEXT, offspring_b TEXT,
                               entropy_a REAL, entropy_b REAL,
                               avg_entropy REAL,
                               fitness_a REAL, fitness_b REAL,
                               timestamp REAL)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS population (
                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                               epoch INTEGER,
                               program TEXT,
                               entropy REAL,
                               fitness REAL,
                               timestamp REAL)''')
        self.conn.commit()

    def generate_random_program(self, length: int) -> str:
        symbols = ['>', '<', '+', '-', '.', ',', '[', ']']
        return ''.join(random.choice(symbols) for _ in range(length))

    def create_soup(self, size: int, length: int) -> List[str]:
        return [self.generate_random_program(length) for _ in range(size)]

    def high_order_entropy(self, program: str) -> float:
        shannon_entropy = self.calculate_shannon_entropy(program)
        kolmogorov_complexity = self.estimate_kolmogorov_complexity(program)
        return shannon_entropy - (kolmogorov_complexity / len(program) if len(program) > 0 else 0)

    def calculate_shannon_entropy(self, data: str) -> float:
        if not data:
            return 0.0
        frequencies = {char: data.count(char) for char in set(data)}
        total = len(data)
        entropy = -sum((freq / total) * math.log2(freq / total) for freq in frequencies.values())
        return entropy

    def estimate_kolmogorov_complexity(self, data: str) -> int:
        compressed = zlib.compress(data.encode('utf-8'))
        return len(compressed)

    def execute_program(self, program: str) -> Any:
        # Placeholder for program execution logic
        # In a real implementation, this would execute the program and return the result
        return None

    def fitness_function(self, program: str) -> float:
        # Define a fitness function to evaluate programs
        # Placeholder implementation: higher entropy means higher fitness
        return self.high_order_entropy(program)

    def mutate(self, program: str) -> str:
        program_list = list(program)
        for i in range(len(program_list)):
            if random.random() < self.MUTATION_RATE:
                program_list[i] = random.choice(['>', '<', '+', '-', '.', ',', '[', ']'])
        mutated_program = ''.join(program_list)
        logging.debug(f"Mutated Program: {mutated_program}")
        return mutated_program

    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        if random.random() > self.CROSSOVER_RATE:
            return parent1, parent2
        point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        logging.debug(f"Crossover at point {point}: Offspring1: {offspring1}, Offspring2: {offspring2}")
        return offspring1, offspring2

    def selection(self, population: List[str], fitness_scores: List[float]) -> List[str]:
        # Roulette Wheel Selection
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        selected = random.choices(population, weights=selection_probs, k=len(population))
        return selected

    def calculate_diversity(self, population: List[str]) -> float:
        unique_programs = set(population)
        diversity = len(unique_programs) / len(population)
        logging.debug(f"Diversity Index: {diversity}")
        return diversity

    def run_epoch(self):
        with self.lock:
            logging.info(f"Starting epoch {self.current_epoch}")
            fitness_scores = [self.fitness_function(prog) for prog in self.soup]
            max_fitness = max(fitness_scores)
            min_fitness = min(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            entropies = [self.high_order_entropy(prog) for prog in self.soup]
            avg_entropy = sum(entropies) / len(entropies)
            max_entropy = max(entropies)
            min_entropy = min(entropies)
            diversity = self.calculate_diversity(self.soup)

            # Log statistics
            self.statistics['avg_entropy'].append(avg_entropy)
            self.statistics['max_entropy'].append(max_entropy)
            self.statistics['min_entropy'].append(min_entropy)
            self.statistics['diversity_index'].append(diversity)
            self.statistics['fitness_scores'].append(avg_fitness)

            # Write to global workspace
            self.global_workspace.write('avg_entropy', avg_entropy)
            self.global_workspace.write('diversity_index', diversity)

            # Selection
            selected_population = self.selection(self.soup, fitness_scores)

            # Generate next generation
            next_generation = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1 if i + 1 < len(selected_population) else 0]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                next_generation.extend([offspring1, offspring2])

                # Save interaction data to database
                timestamp = time.time()
                entropy_a = self.high_order_entropy(offspring1)
                entropy_b = self.high_order_entropy(offspring2)
                fitness_a = self.fitness_function(offspring1)
                fitness_b = self.fitness_function(offspring2)
                avg_entropy_offspring = (entropy_a + entropy_b) / 2

                self.cursor.execute('''
                    INSERT INTO interactions (
                        epoch, program_a, program_b, offspring_a, offspring_b,
                        entropy_a, entropy_b, avg_entropy, fitness_a, fitness_b, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.current_epoch, parent1, parent2, offspring1, offspring2,
                    entropy_a, entropy_b, avg_entropy_offspring, fitness_a, fitness_b, timestamp
                ))
                self.conn.commit()

            # Update soup with the next generation
            self.soup = next_generation[:self.SOUP_SIZE]

            # Save population data
            timestamp = time.time()
            for prog in self.soup:
                entropy = self.high_order_entropy(prog)
                fitness = self.fitness_function(prog)
                self.cursor.execute('''
                    INSERT INTO population (epoch, program, entropy, fitness, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (self.current_epoch, prog, entropy, fitness, timestamp))
            self.conn.commit()

            # Increment epoch
            self.current_epoch += 1

    def run(self):
        while not self.stop_event.is_set() and self.current_epoch < self.EPOCHS:
            self.run_epoch()
            # Sleep to simulate real-time processing
            time.sleep(1)

        # Close database connection
        self.conn.close()
        logging.info("Evolutionary Dynamics simulation completed.")

    def stop(self):
        self.stop_event.set()
        self.visualization_thread.join()

    def visualize_statistics(self):
        plt.ion()
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        while not self.stop_event.is_set():
            if self.statistics['avg_entropy']:
                axs[0, 0].clear()
                axs[0, 0].plot(self.statistics['avg_entropy'], label='Average Entropy')
                axs[0, 0].set_title('Average Entropy Over Epochs')
                axs[0, 0].legend()

                axs[0, 1].clear()
                axs[0, 1].plot(self.statistics['diversity_index'], label='Diversity Index', color='orange')
                axs[0, 1].set_title('Diversity Index Over Epochs')
                axs[0, 1].legend()

                axs[1, 0].clear()
                axs[1, 0].plot(self.statistics['fitness_scores'], label='Average Fitness', color='green')
                axs[1, 0].set_title('Average Fitness Over Epochs')
                axs[1, 0].legend()

                axs[1, 1].clear()
                axs[1, 1].plot(self.statistics['max_entropy'], label='Max Entropy', color='red')
                axs[1, 1].plot(self.statistics['min_entropy'], label='Min Entropy', color='purple')
                axs[1, 1].set_title('Entropy Extremes Over Epochs')
                axs[1, 1].legend()

                plt.pause(0.1)

            time.sleep(1)

        plt.ioff()
        plt.show()

    def save_statistics(self, filename: str = 'evolution_stats.npz'):
        np.savez(filename,
                 avg_entropy=self.statistics['avg_entropy'],
                 max_entropy=self.statistics['max_entropy'],
                 min_entropy=self.statistics['min_entropy'],
                 diversity_index=self.statistics['diversity_index'],
                 fitness_scores=self.statistics['fitness_scores'])
        logging.info(f"Statistics saved to {filename}")

    def load_statistics(self, filename: str = 'evolution_stats.npz'):
        data = np.load(filename)
        self.statistics['avg_entropy'] = data['avg_entropy'].tolist()
        self.statistics['max_entropy'] = data['max_entropy'].tolist()
        self.statistics['min_entropy'] = data['min_entropy'].tolist()
        self.statistics['diversity_index'] = data['diversity_index'].tolist()
        self.statistics['fitness_scores'] = data['fitness_scores'].tolist()
        logging.info(f"Statistics loaded from {filename}")

    def export_population(self, filename: str = 'population.txt'):
        with open(filename, 'w') as f:
            for prog in self.soup:
                f.write(f"{prog}\n")
        logging.info(f"Population exported to {filename}")

    def import_population(self, filename: str = 'population.txt'):
        with open(filename, 'r') as f:
            self.soup = [line.strip() for line in f.readlines()]
        logging.info(f"Population imported from {filename}")

    def analyze_population(self):
        # Additional analysis can be added here
        pass

    def get_top_performers(self, top_n: int = 10) -> List[str]:
        fitness_scores = [(prog, self.fitness_function(prog)) for prog in self.soup]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        top_performers = [prog for prog, _ in fitness_scores[:top_n]]
        logging.debug(f"Top {top_n} performers: {top_performers}")
        return top_performers

    def inject_new_programs(self, num_programs: int = 10):
        new_programs = self.create_soup(num_programs, self.PROGRAM_LENGTH)
        self.soup.extend(new_programs)
        if len(self.soup) > self.SOUP_SIZE:
            self.soup = self.soup[:self.SOUP_SIZE]
        logging.info(f"Injected {num_programs} new programs into the population.")

    def adjust_mutation_rate(self, factor: float):
        self.MUTATION_RATE *= factor
        logging.info(f"Mutation rate adjusted to {self.MUTATION_RATE}")

    def adjust_crossover_rate(self, factor: float):
        self.CROSSOVER_RATE *= factor
        logging.info(f"Crossover rate adjusted to {self.CROSSOVER_RATE}")

    def receive_commands(self):
        while not self.stop_event.is_set():
            # Check for commands from the global workspace
            command = self.global_workspace.read('evolution_command')
            if command:
                if command == 'increase_mutation':
                    self.adjust_mutation_rate(1.1)
                elif command == 'decrease_mutation':
                    self.adjust_mutation_rate(0.9)
                elif command == 'inject_programs':
                    self.inject_new_programs(50)
                elif command == 'export_population':
                    self.export_population()
                elif command == 'import_population':
                    self.import_population()
                # Clear the command after processing
                self.global_workspace.write('evolution_command', None)
            time.sleep(1)

    def start(self):
        self.main_thread = threading.Thread(target=self.run)
        self.command_thread = threading.Thread(target=self.receive_commands)
        self.main_thread.start()
        self.command_thread.start()

    def join(self):
        self.main_thread.join()
        self.command_thread.join()

# Example usage:
# global_workspace = GlobalWorkspace()
# evolution = EvolutionaryDynamics(global_workspace)
# evolution.start()
# # Run for some time...
# evolution.stop()
# evolution.save_statistics()
