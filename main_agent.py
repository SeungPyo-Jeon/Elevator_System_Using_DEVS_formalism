"""
Reinforcement Learning based Elevator Control System
Using DQN (Deep Q-Network) with DEVS Simulation

Author: SeungPyo
Description: 강화학습을 이용한 엘리베이터 그룹 제어 시스템
"""

from pypdevs.DEVS import *
from pypdevs.infinity import INFINITY
from pypdevs.simulator import Simulator
import random
import csv
import math
import numpy as np
from collections import deque, defaultdict
import copy
import os
from datetime import datetime
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Simulation parameters
    FLOOR_MAX = 8
    NUM_ELEVATORS = 3
    ELEVATOR_CAPACITY = 4
    SIMULATION_TIME = 500
    HOURLY_RATES = [700, 500, 300, 200, 400, 500, 600, 200]
    
    # RL parameters
    STATE_DIM = None  # Will be calculated
    ACTION_DIM = 3 ** 3  # 3 actions (UP, DOWN, IDLE) per elevator, 3 elevators = 27 combinations
    
    # DQN Hyperparameters
    LEARNING_RATE = 1e-4
    GAMMA = 0.99  # Discount factor
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BATCH_SIZE = 64
    MEMORY_SIZE = 100000
    TARGET_UPDATE_FREQ = 100  # Update target network every N episodes
    
    # Training parameters
    NUM_EPISODES = 1000
    SAVE_FREQ = 100  # Save model every N episodes
    
    # Reward weights
    REWARD_PASSENGER_SERVED = 10.0
    REWARD_WAITING_PENALTY = -0.1  # Per passenger per second
    REWARD_ENERGY_PENALTY = -0.05  # Per floor moved
    REWARD_IDLE_BONUS = 0.01  # Small bonus for being idle when no passengers


# ============================================================================
# CSV Event Logger (Same as original)
# ============================================================================
class EventLogger:
    def __init__(self, filename="simulation_events.csv"):
        self.filename = filename
        self.file = None
        self.writer = None

    def open(self):
        self.file = open(self.filename, 'w', newline='', encoding='utf-8')
        fieldnames = ['time', 'event_type', 'floor', 'destination', 'direction', 'elevator_id', 'details']
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log_event(self, time, event_type, floor=None, destination=None, direction=None, elevator_id=None, details=None):
        if self.writer:
            self.writer.writerow({
                'time': f"{time:.2f}",
                'event_type': event_type,
                'floor': floor if floor is not None else '',
                'destination': destination if destination is not None else '',
                'direction': direction if direction else '',
                'elevator_id': elevator_id if elevator_id else '',
                'details': details if details else ''
            })
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# ============================================================================
# Global States
# ============================================================================
elevator_states = {}
totalbuffer_state = None
event_logger = None
rl_agent = None  # RL Agent reference
metrics_collector = None  # For collecting training metrics


# ============================================================================
# Metrics Collector for RL Training
# ============================================================================
class MetricsCollector:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.passengers_served = 0
        self.total_waiting_time = 0.0
        self.total_travel_time = 0.0
        self.total_energy = 0  # Floor movements
        self.passenger_timestamps = {}  # passenger_id -> generation_time
        self.step_rewards = []
        self.decisions_made = 0
    
    def record_passenger_generated(self, passenger_id, timestamp):
        self.passenger_timestamps[passenger_id] = {
            'generated': timestamp,
            'boarded': None,
            'alighted': None
        }
    
    def record_passenger_boarded(self, passenger_id, timestamp):
        if passenger_id in self.passenger_timestamps:
            self.passenger_timestamps[passenger_id]['boarded'] = timestamp
            wait_time = timestamp - self.passenger_timestamps[passenger_id]['generated']
            self.total_waiting_time += wait_time
    
    def record_passenger_alighted(self, passenger_id, timestamp):
        if passenger_id in self.passenger_timestamps:
            self.passenger_timestamps[passenger_id]['alighted'] = timestamp
            if self.passenger_timestamps[passenger_id]['boarded']:
                travel_time = timestamp - self.passenger_timestamps[passenger_id]['boarded']
                self.total_travel_time += travel_time
            self.passengers_served += 1
    
    def record_energy(self, floors_moved):
        self.total_energy += abs(floors_moved)
    
    def record_step_reward(self, reward):
        self.step_rewards.append(reward)
        self.decisions_made += 1
    
    def get_episode_stats(self):
        return {
            'passengers_served': self.passengers_served,
            'avg_waiting_time': self.total_waiting_time / max(1, self.passengers_served),
            'avg_travel_time': self.total_travel_time / max(1, self.passengers_served),
            'total_energy': self.total_energy,
            'total_reward': sum(self.step_rewards),
            'avg_reward': sum(self.step_rewards) / max(1, len(self.step_rewards)),
            'decisions_made': self.decisions_made
        }


# ============================================================================
# Data Structures (Same as original with minor additions)
# ============================================================================
class TowerState:
    def __init__(self, floor_max):
        self.state = {}
        self.floor_max = floor_max
        for idx in range(floor_max):
            self.state[idx] = {"up": [], "down": []}

    def add_passenger(self, passenger):
        floor_id = passenger[0]
        passenger_id = passenger[1]
        destination_floor = passenger[2]
        timestamp = passenger[3]
        direction = "up" if destination_floor > floor_id else "down"
        self.state[floor_id][direction].append([destination_floor, timestamp,passenger_id])

    def pop_passengers(self, floor_id, available_space, direction):
        if direction == "IDLE":
            return []
        dir_key = "up" if direction == "UP" else "down"
        pop_list = []
        queue = self.state[floor_id][dir_key]
        while queue and len(pop_list) < available_space:
            pop_list.append(queue.pop(0))
        return pop_list

    def get_waiting_count(self, floor_id, direction):
        if direction == "IDLE":
            return 0
        dir_key = "up" if direction == "UP" else "down"
        return len(self.state[floor_id][dir_key])

    def get_total_waiting(self, floor_id):
        return len(self.state[floor_id]["up"]) + len(self.state[floor_id]["down"])
    
    def get_total_system_waiting(self):
        total = 0
        for floor_id in range(self.floor_max):
            total += len(self.state[floor_id]["up"]) + len(self.state[floor_id]["down"])
        return total
    
    def get_all_calls(self):
        calls = []
        for floor_id in range(self.floor_max):
            for dir_key in ["up", "down"]:
                queue = self.state[floor_id][dir_key]
                if len(queue) > 0:
                    direction = "UP" if dir_key == "up" else "DOWN"
                    oldest_timestamp = min(p[1] for p in queue)
                    calls.append((floor_id, direction, len(queue), oldest_timestamp))
        return calls
    
    def get_state_vector(self, current_time):
        """
        Convert tower state to a vector for RL
        Returns: numpy array of shape (floor_max * 4,)
        - For each floor: [up_count, down_count, up_avg_wait, down_avg_wait]
        """
        vector = []
        for floor_id in range(self.floor_max):
            up_queue = self.state[floor_id]["up"]
            down_queue = self.state[floor_id]["down"]
            
            up_count = len(up_queue)
            down_count = len(down_queue)
            
            up_avg_wait = 0.0
            if up_count > 0:
                up_avg_wait = sum(current_time - p[1] for p in up_queue) / up_count
            
            down_avg_wait = 0.0
            if down_count > 0:
                down_avg_wait = sum(current_time - p[1] for p in down_queue) / down_count
            
            # Normalize
            vector.extend([
                up_count / Config.ELEVATOR_CAPACITY,  # Normalized by capacity
                down_count / Config.ELEVATOR_CAPACITY,
                up_avg_wait / 60.0,  # Normalized by 60 seconds
                down_avg_wait / 60.0
            ])
        
        return np.array(vector, dtype=np.float32)


class ElevatorState:
    def __init__(self, capacity):
        self.capacity = capacity
        self.passengers = []
        self.current_floor = 0
        self.direction = "IDLE"

    def available_space(self):
        return self.capacity - len(self.passengers)
    
    def get_destinations(self):
        return [p[0] for p in self.passengers]
    
    def get_state_vector(self, floor_max):
        """
        Convert elevator state to a vector for RL
        Returns: numpy array
        """
        # One-hot encode current floor
        floor_one_hot = np.zeros(floor_max, dtype=np.float32)
        floor_one_hot[self.current_floor] = 1.0
        
        # Direction encoding: UP=1, IDLE=0, DOWN=-1
        direction_map = {"UP": 1.0, "IDLE": 0.0, "DOWN": -1.0}
        direction_val = direction_map.get(self.direction, 0.0)
        
        # Passenger count (normalized)
        passenger_ratio = len(self.passengers) / self.capacity
        
        # Destination distribution
        dest_dist = np.zeros(floor_max, dtype=np.float32)
        for p in self.passengers:
            dest_dist[p[0]] += 1.0 / self.capacity
        
        return np.concatenate([
            floor_one_hot,
            [direction_val],
            [passenger_ratio],
            dest_dist
        ])


# ============================================================================
# Deep Q-Network
# ============================================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# Replay Buffer
# ============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# RL Agent (DQN)
# ============================================================================
class ElevatorRLAgent:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LEARNING_RATE)
        
        # Replay buffer
        self.memory = ReplayBuffer(Config.MEMORY_SIZE)
        
        # Exploration
        self.epsilon = Config.EPSILON_START
        
        # Action mapping: index -> (elevator1_action, elevator2_action, elevator3_action)
        # Each action is 0=UP, 1=DOWN, 2=IDLE
        self.action_map = self._build_action_map()
        
        # Training state
        self.training_mode = True
        self.last_state = None
        self.last_action = None
        self.total_steps = 0
    
    def _build_action_map(self):
        """Build mapping from action index to individual elevator actions"""
        actions = ["UP", "DOWN", "IDLE"]
        action_map = {}
        idx = 0
        for a1 in actions:
            for a2 in actions:
                for a3 in actions:
                    action_map[idx] = (a1, a2, a3)
                    idx += 1
        return action_map
    
    def get_state(self, current_time):
        """
        Construct state vector from global states
        """
        global elevator_states, totalbuffer_state
        
        # Tower state vector
        tower_vec = totalbuffer_state.get_state_vector(current_time)
        
        # Elevator state vectors
        elevator_vecs = []
        for i in range(1, Config.NUM_ELEVATORS + 1):
            elev_name = f"Elevator_{i}"
            if elev_name in elevator_states:
                elev_vec = elevator_states[elev_name].get_state_vector(Config.FLOOR_MAX)
                elevator_vecs.append(elev_vec)
        
        # Concatenate all
        if elevator_vecs:
            full_state = np.concatenate([tower_vec] + elevator_vecs)
        else:
            full_state = tower_vec
        
        return full_state
    
    def select_action(self, state, valid_actions_mask=None):
        """
        Select action using epsilon-greedy policy
        """
        if self.training_mode and random.random() < self.epsilon:
            # Random action
            action = random.randint(0, self.action_dim - 1)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                if valid_actions_mask is not None:
                    # Mask invalid actions
                    mask = torch.FloatTensor(valid_actions_mask).to(self.device)
                    q_values = q_values * mask - 1e9 * (1 - mask)
                
                action = q_values.argmax(dim=1).item()
        
        return action
    
    def action_to_commands(self, action_idx):
        """
        Convert action index to elevator commands dict
        """
        actions = self.action_map[action_idx]
        commands = {}
        for i, act in enumerate(actions):
            commands[f"Elevator_{i+1}"] = act
        return commands
    
    def calculate_reward(self, prev_state, action, current_time):
        """
        Calculate reward based on current state
        """
        global totalbuffer_state, elevator_states, metrics_collector
        
        reward = 0.0
        
        # Penalty for waiting passengers
        total_waiting = totalbuffer_state.get_total_system_waiting()
        reward += Config.REWARD_WAITING_PENALTY * total_waiting
        
        # Energy penalty (count non-IDLE actions)
        commands = self.action_to_commands(action)
        for elev_name, cmd in commands.items():
            if cmd != "IDLE":
                reward += Config.REWARD_ENERGY_PENALTY
            else:
                # Small bonus for being idle when appropriate
                if total_waiting == 0:
                    reward += Config.REWARD_IDLE_BONUS
        
        return reward
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        """
        Perform one step of optimization
        """
        if len(self.memory) < Config.BATCH_SIZE:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(Config.BATCH_SIZE)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            # Select best action using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Evaluate using target network
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + Config.GAMMA * next_q * (1 - dones.unsqueeze(1))
        
        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.total_steps += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Copy policy network weights to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.total_steps = checkpoint['total_steps']
            print(f"Model loaded from {filepath}")
            return True
        return False
    
    def set_training_mode(self, mode):
        """Set training or evaluation mode"""
        self.training_mode = mode
        if mode:
            self.policy_net.train()
        else:
            self.policy_net.eval()
            self.epsilon = 0.0  # No exploration during evaluation


# ============================================================================
# DEVS Atomic Models
# ============================================================================
class Floor(AtomicDEVS):
    def __init__(self, floor_id, floor_max, hourly_rate=5):
        super().__init__(f"Floor_{floor_id}")
        self.floor_id = floor_id
        self.floor_max = floor_max
        self.hourly_rate = hourly_rate
        self.floor_outport = self.addOutPort("floor_outport")
        self.state = "GEN"

    def timeAdvance(self):
        if self.state == "GEN":
            return random.expovariate(self.hourly_rate / 60) * 60
        return INFINITY

    def intTransition(self):
        return "GEN"

    def outputFnc(self):
        destination_floor = random.randint(0, self.floor_max - 1)
        while destination_floor == self.floor_id:
            destination_floor = random.randint(0, self.floor_max - 1)
        
        timestamp = self.time_next[0]
        passenger_id = f"P_{self.floor_id}_{int(timestamp*100)}"
        
        out = [self.floor_id, passenger_id, destination_floor, timestamp]
        return {self.floor_outport: out}


class TotalBuffer(AtomicDEVS):
    def __init__(self, floor_max):
        super().__init__("TotalBuffer")
        global totalbuffer_state
        totalbuffer_state = TowerState(floor_max)
        self.floor_input = self.addInPort("floor_input")

    def timeAdvance(self):
        return INFINITY

    def extTransition(self, inputs):
        global totalbuffer_state, event_logger, metrics_collector
        if self.floor_input in inputs:
            passenger = inputs[self.floor_input]
            totalbuffer_state.add_passenger(passenger)
            
            floor_id, passenger_id, destination_floor, timestamp = passenger
            direction = "UP" if destination_floor > floor_id else "DOWN"
            
            # Track metrics
            if metrics_collector:
                metrics_collector.record_passenger_generated(passenger_id, timestamp)
            
            if event_logger:
                event_logger.log_event(
                    time=timestamp,
                    event_type='passenger_generated',
                    floor=floor_id + 1,
                    destination=destination_floor + 1,
                    direction=direction,
                    details=f"Passenger {passenger_id}"
                )
        return self.state


class RLController(AtomicDEVS):
    """
    RL-based Controller that uses the trained agent to make decisions
    """
    def __init__(self, num_elevators, floor_max):
        super().__init__("RLController")
        self.num_elevators = num_elevators
        self.floor_max = floor_max
        self.elevator_input = self.addInPort("elevator_input")
        self.elevator_output = self.addOutPort("elevator_output")
        
        self.state = {"count": 0}
        self.last_decision_time = 0
        self.cumulative_reward = 0

    def timeAdvance(self):
        if self.state["count"] == self.num_elevators:
            return 0.01
        return INFINITY

    def extTransition(self, inputs):
        if self.elevator_input in inputs:
            self.state["count"] += 1
        return self.state

    def outputFnc(self):
        global elevator_states, totalbuffer_state, rl_agent, metrics_collector
        
        if self.state["count"] != self.num_elevators:
            return {}
        
        current_time = self.time_last[0] if hasattr(self, 'time_last') else 0
        
        # Get current state for RL
        current_state = rl_agent.get_state(current_time)
        
        # Store transition from previous decision
        if rl_agent.last_state is not None:
            reward = rl_agent.calculate_reward(
                rl_agent.last_state, 
                rl_agent.last_action,
                current_time
            )
            if metrics_collector:
                metrics_collector.record_step_reward(reward)
            
            # Add reward for passengers served since last decision
            # (This will be added in elevator's passenger_alighted event)
            if rl_agent.training_mode:
                rl_agent.store_transition(
                    rl_agent.last_state,
                    rl_agent.last_action,
                    reward,
                    current_state,
                    False  # Not done (episode continues)
                )
            
            
                # Perform learning update
                loss = rl_agent.update()
        
        # Select action
        action = rl_agent.select_action(current_state)
        commands = rl_agent.action_to_commands(action)
        
        # Apply constraints (boundary checks)
        commands = self._apply_constraints(commands)
        
        # Store for next transition
        rl_agent.last_state = current_state
        rl_agent.last_action = action
        
        # Debug output
        total_waiting = totalbuffer_state.get_total_system_waiting()
        for name in sorted(commands.keys()):
            e_state = elevator_states[name]
            #print(f"[RL Controller] {name} @ F{e_state.current_floor}: "
            #      f"Load={len(e_state.passengers)}, Cmd={commands[name]}")
        
        if total_waiting > 0:
            #print(f"[RL Controller] Total waiting: {total_waiting}, Epsilon: {rl_agent.epsilon:.3f}")
            pass
        # Create command list
        command_list = []
        for i in range(1, self.num_elevators + 1):
            elevator_name = f"Elevator_{i}"
            command_list.append(commands[elevator_name])
        
        return {self.elevator_output: command_list}

    def _apply_constraints(self, commands):
        """Apply physical constraints to commands"""
        global elevator_states
        
        for name, cmd in commands.items():
            e_state = elevator_states[name]
            
            # Boundary checks
            if e_state.current_floor == 0 and cmd == "DOWN":
                commands[name] = "IDLE"
            if e_state.current_floor == self.floor_max - 1 and cmd == "UP":
                commands[name] = "IDLE"
            
            # If passengers onboard, override with required direction
            if len(e_state.passengers) > 0:
                destinations = e_state.get_destinations()
                has_up = any(d > e_state.current_floor for d in destinations)
                has_down = any(d < e_state.current_floor for d in destinations)
                
                if has_up and not has_down:
                    commands[name] = "UP"
                elif has_down and not has_up:
                    commands[name] = "DOWN"
        
        return commands

    def intTransition(self):
        self.state["count"] = 0
        return self.state


class Elevator(AtomicDEVS):
    def __init__(self, name, capacity, init_delay=0.0):
        super().__init__(name)
        self.name = name
        self.init_delay = init_delay
        global elevator_states
        elevator_states[self.name] = ElevatorState(capacity)
        
        self.ctrl_in = self.addInPort("ctrl_in")
        self.status_out = self.addOutPort("status_out")
        
        self.state = "UNLOAD"
        self.boarded_passengers = []  # Track passenger IDs for metrics

    def timeAdvance(self):
        if self.state == "UNLOAD":
            return self.init_delay
        elif self.state == "WAIT_CMD":
            return INFINITY
        elif self.state == "MOVE":
            return 1.0
        return INFINITY

    def outputFnc(self):
        if self.state == "UNLOAD":
            return {self.status_out: (self.name,)}
        return {}

    def intTransition(self):
        global elevator_states, event_logger, metrics_collector, rl_agent
        est = elevator_states[self.name]

        if self.state == "UNLOAD":
            self.state = "WAIT_CMD"

        elif self.state == "MOVE":
            current_time = self.time_last[0]
            old_floor = est.current_floor
            
            # Move
            if est.direction == "UP":
                est.current_floor += 1
            elif est.direction == "DOWN":
                est.current_floor -= 1
            
            # Track energy
            if metrics_collector and est.direction != "IDLE":
                metrics_collector.record_energy(1)
            
            # Log
            if event_logger:
                event_logger.log_event(
                    time=current_time,
                    event_type='elevator_moved',
                    floor=est.current_floor + 1,
                    elevator_id=self.name,
                    details=f"Direction: {est.direction}, From: F{old_floor+1}, Passengers: {len(est.passengers)}"
                )

            # Alight passengers
            staying_passengers = []
            alighted_passengers = []
            for i, p in enumerate(est.passengers):
                if p[0] == est.current_floor:
                    alighted_passengers.append(p)
                else:
                    staying_passengers.append(p)
            
            alighted_count = len(alighted_passengers)
            est.passengers = staying_passengers

            if alighted_count > 0:
                #print(f"[{self.name} @ F{est.current_floor}] Alighted {alighted_count}. Load: {len(est.passengers)}")
                
                # Record metrics and add reward for each alighted passenger
                for p in alighted_passengers:
                    passenger_id = p[2]
                    if metrics_collector:
                        # We need passenger ID - using a composite key
                        passenger_key = f"{p[0]}_{p[1]}"  # destination_timestamp
                        metrics_collector.record_passenger_alighted(passenger_id, current_time)
                    
                    if event_logger:
                        event_logger.log_event(
                            time=current_time,
                            event_type='passenger_alighted',
                            floor=est.current_floor + 1,
                            elevator_id=self.name,
                            details=f"Remaining load: {len(est.passengers)}"
                        )

            self.state = "UNLOAD"

        return self.state

    def extTransition(self, inputs):
        global elevator_states, totalbuffer_state, event_logger, metrics_collector

        if self.state == "WAIT_CMD" and self.ctrl_in in inputs:
            msg = inputs[self.ctrl_in]
            
            if msg:
                my_cmd = msg[int(self.name.split("_")[1]) - 1]
                est = elevator_states[self.name]
                est.direction = my_cmd

                # Board passengers
                available = est.available_space()
                if available > 0 and est.direction != "IDLE":
                    new_passengers = totalbuffer_state.pop_passengers(
                        est.current_floor, available, est.direction
                    )
                    if new_passengers:
                        est.passengers.extend(new_passengers)
                        #print(f"[{self.name} @ F{est.current_floor}] Boarded {len(new_passengers)} ({est.direction}). Load: {len(est.passengers)}")

                        current_time = self.time_last[0]
                        
                        for passenger in new_passengers:
                            passenger_id = passenger[2]
                            passenger_key = f"{passenger[0]}_{passenger[1]}"
                            if metrics_collector:
                                metrics_collector.record_passenger_boarded(passenger_id, current_time)
                            
                            if event_logger:
                                event_logger.log_event(
                                    time=current_time,
                                    event_type='passenger_boarded',
                                    floor=est.current_floor + 1,
                                    destination=passenger[0] + 1,
                                    direction=est.direction,
                                    elevator_id=self.name,
                                    details=f"Current load: {len(est.passengers)}"
                                )

                self.state = "MOVE"

        return self.state

    def confTransition(self, inputs):
        self.intTransition()
        return self.extTransition(inputs)


class Building(CoupledDEVS):
    def __init__(self, floor_max, num_elevators, hourly_rates, use_rl=True):
        super().__init__("Building")

        self.totalbuffer = TotalBuffer(floor_max)
        
        if use_rl:
            self.controller = RLController(num_elevators, floor_max)
        else:
            # Could add traditional controller here for comparison
            self.controller = RLController(num_elevators, floor_max)
        
        self.addSubModel(self.totalbuffer)
        self.addSubModel(self.controller)

        self.elevators = []
        for i in range(num_elevators):
            elev = Elevator(f"Elevator_{i+1}", capacity=Config.ELEVATOR_CAPACITY, 
                          init_delay=0.01 * (i + 1))
            self.addSubModel(elev)
            self.elevators.append(elev)

            self.connectPorts(elev.status_out, self.controller.elevator_input)
            self.connectPorts(self.controller.elevator_output, elev.ctrl_in)

        self.floor_models = []
        for floor_id in range(floor_max):
            floor_model = Floor(floor_id, floor_max, hourly_rates[floor_id])
            self.addSubModel(floor_model)
            self.floor_models.append(floor_model)
            self.connectPorts(floor_model.floor_outport, self.totalbuffer.floor_input)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================
def calculate_state_dim():
    """Calculate the state dimension for the RL agent"""
    # Tower state: floor_max * 4 (up_count, down_count, up_wait, down_wait)
    tower_dim = Config.FLOOR_MAX * 4
    
    # Each elevator: floor_max (one-hot) + 1 (direction) + 1 (passenger ratio) + floor_max (dest dist)
    elevator_dim = Config.FLOOR_MAX + 1 + 1 + Config.FLOOR_MAX
    
    total_dim = tower_dim + Config.NUM_ELEVATORS * elevator_dim
    return total_dim


def reset_simulation():
    """Reset global states for new episode"""
    global elevator_states, totalbuffer_state, metrics_collector
    elevator_states = {}
    totalbuffer_state = None
    if metrics_collector:
        metrics_collector.reset()


def run_episode(episode_num, training=True, log_events=False):
    """Run a single simulation episode"""
    global event_logger, elevator_states, totalbuffer_state, rl_agent, metrics_collector
    
    # Reset
    reset_simulation()
    metrics_collector = MetricsCollector()
    
    # Setup logging
    if log_events:
        log_filename = f"simulation_events_ep{episode_num}.csv"
        event_logger = EventLogger(log_filename)
        event_logger.open()
    else:
        event_logger = None
    
    # Set agent mode
    rl_agent.set_training_mode(training)
    rl_agent.last_state = None
    rl_agent.last_action = None
    
    try:
        # Create and run simulation
        building = Building(
            floor_max=Config.FLOOR_MAX,
            num_elevators=Config.NUM_ELEVATORS,
            hourly_rates=Config.HOURLY_RATES,
            use_rl=True
        )
        
        sim = Simulator(building)
        sim.setTerminationTime(Config.SIMULATION_TIME)
        sim.simulate()
        
        # Get episode stats
        stats = metrics_collector.get_episode_stats()
        
        # Final state handling
        if training and rl_agent.last_state is not None:
            # Terminal transition
            final_state = rl_agent.get_state(Config.SIMULATION_TIME)
            final_reward = rl_agent.calculate_reward(
                rl_agent.last_state,
                rl_agent.last_action,
                Config.SIMULATION_TIME
            )
            # Add bonus for completing episode
            final_reward += stats['passengers_served'] * Config.REWARD_PASSENGER_SERVED
            
            rl_agent.store_transition(
                rl_agent.last_state,
                rl_agent.last_action,
                final_reward,
                final_state,
                True  # Done
            )
        
        return stats
        
    finally:
        if event_logger:
            event_logger.close()


def train(num_episodes=None, save_dir="checkpoints"):
    """Train the RL agent"""
    global rl_agent
    
    if num_episodes is None:
        num_episodes = Config.NUM_EPISODES
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate state dimension and create agent
    state_dim = calculate_state_dim()
    Config.STATE_DIM = state_dim
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {Config.ACTION_DIM}")
    
    rl_agent = ElevatorRLAgent(state_dim, Config.ACTION_DIM, device)
    
    # Try to load existing model
    model_path = os.path.join(save_dir, "latest_model.pt")
    rl_agent.load(model_path)
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for episode in tqdm(range(1, num_episodes + 1)):
        # Run episode
        stats = run_episode(episode, training=True, log_events=False)
        
        episode_rewards.append(stats['total_reward'])
        
        # Decay epsilon
        rl_agent.decay_epsilon()
        
        # Update target network
        if episode % Config.TARGET_UPDATE_FREQ == 0:
            rl_agent.update_target_network()
        
        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Reward: {stats['total_reward']:.2f} (Avg100: {avg_reward:.2f})")
            print(f"  Passengers Served: {stats['passengers_served']}")
            print(f"  Avg Wait Time: {stats['avg_waiting_time']:.2f}s")
            print(f"  Epsilon: {rl_agent.epsilon:.3f}")
            print()
        
        # Save model
        if episode % Config.SAVE_FREQ == 0:
            rl_agent.save(os.path.join(save_dir, f"model_ep{episode}.pt"))
            rl_agent.save(model_path)
        
        # Save best model
        if stats['total_reward'] > best_reward:
            best_reward = stats['total_reward']
            rl_agent.save(os.path.join(save_dir, "best_model.pt"))
    
    # Final save
    rl_agent.save(model_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Reward: {best_reward:.2f}")
    print("="*60)
    
    return episode_rewards


def evaluate(model_path, num_episodes=10, log_events=True):
    """Evaluate a trained model"""
    global rl_agent
    
    # Setup agent
    state_dim = calculate_state_dim()
    Config.STATE_DIM = state_dim
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rl_agent = ElevatorRLAgent(state_dim, Config.ACTION_DIM, device)
    
    if not rl_agent.load(model_path):
        print(f"Error: Could not load model from {model_path}")
        return None
    
    rl_agent.set_training_mode(False)
    
    print("\n" + "="*60)
    print("Starting Evaluation")
    print("="*60)
    
    all_stats = []
    
    for episode in range(1, num_episodes + 1):
        print(f"\nEvaluation Episode {episode}/{num_episodes}")
        
        stats = run_episode(
            episode, 
            training=False, 
            log_events=log_events and episode == 1  # Log first episode only
        )
        
        all_stats.append(stats)
        
        print(f"  Passengers Served: {stats['passengers_served']}")
        print(f"  Avg Wait Time: {stats['avg_waiting_time']:.2f}s")
        print(f"  Avg Travel Time: {stats['avg_travel_time']:.2f}s")
        print(f"  Total Energy: {stats['total_energy']}")
        print(f"  Total Reward: {stats['total_reward']:.2f}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    avg_passengers = np.mean([s['passengers_served'] for s in all_stats])
    avg_wait = np.mean([s['avg_waiting_time'] for s in all_stats])
    avg_travel = np.mean([s['avg_travel_time'] for s in all_stats])
    avg_energy = np.mean([s['total_energy'] for s in all_stats])
    avg_reward = np.mean([s['total_reward'] for s in all_stats])
    
    print(f"Average Passengers Served: {avg_passengers:.1f}")
    print(f"Average Wait Time: {avg_wait:.2f}s")
    print(f"Average Travel Time: {avg_travel:.2f}s")
    print(f"Average Energy: {avg_energy:.1f}")
    print(f"Average Reward: {avg_reward:.2f}")
    
    return all_stats

def plot_training_results(rewards, window_size=50, save_dir='checkpoints'):
    """
    학습 결과(Reward)를 시각화하여 저장합니다.
    - rewards: 에피소드별 보상 리스트
    - window_size: 이동 평균을 계산할 윈도우 크기
    """
    plt.figure(figsize=(12, 6))
    
    # 1. Raw Data (회색, 투명하게) - 변동성을 확인하기 위함
    plt.plot(rewards, alpha=0.3, color='gray', label='Raw Reward')
    
    # 2. Moving Average (붉은색, 진하게) - 전체적인 추세를 확인하기 위함
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        # x축 위치를 맞추기 위해 range 조정
        plt.plot(range(window_size-1, len(rewards)), moving_avg, color='red', 
                 linewidth=2, label=f'Moving Avg ({window_size})')
    
    plt.title('Training Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 파일 저장
    save_path = os.path.join(save_dir, 'learning_curve.png')
    plt.savefig(save_path)
    print(f"학습 곡선이 저장되었습니다: {save_path}")
    plt.close()
# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RL-based Elevator Control System')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'demo'],
                       help='Mode: train, eval, or demo')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes (default: 1000 for train, 10 for eval)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                       help='Model path for evaluation')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        episodes = args.episodes if args.episodes else Config.NUM_EPISODES
        episode_rewards = train(num_episodes=episodes, save_dir=args.save_dir)
        plot_training_results(episode_rewards, save_dir=args.save_dir)
        
    elif args.mode == 'eval':
        episodes = args.episodes if args.episodes else 100
        evaluate(args.model, num_episodes=episodes, log_events=True)
        
    elif args.mode == 'demo':
        # Quick demo with minimal training
        print("Running demo with 50 training episodes...")
        train(num_episodes=50, save_dir=args.save_dir)
        print("\nEvaluating trained model...")
        evaluate(os.path.join(args.save_dir, 'latest_model.pt'), num_episodes=3, log_events=True)
