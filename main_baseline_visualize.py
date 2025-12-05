from pypdevs.DEVS import *
from pypdevs.infinity import INFINITY
from pypdevs.simulator import Simulator
import random
import csv
import math
import numpy as np
# --- CSV Event Logger ---
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

# --- Global States ---
elevator_states = {}
totalbuffer_state = None
event_logger = None 
metrics_collector = None

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

# --- Data Structures ---
class TowerState():
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
        self.state[floor_id][direction].append([destination_floor, timestamp, passenger_id])

    def pop_passengers(self, floor_id, available_space, direction):
        """승객 탑승: 해당 층의 해당 방향 대기열에서 승객 제거 및 반환"""
        if direction == "IDLE":
            return []

        dir_key = "up" if direction == "UP" else "down"
        pop_list = []

        queue = self.state[floor_id][dir_key]
        while queue and len(pop_list) < available_space:
            pop_list.append(queue.pop(0))

        return pop_list

    def get_waiting_count(self, floor_id, direction):
        """특정 층의 특정 방향 대기 승객 수"""
        if direction == "IDLE":
            return 0
        dir_key = "up" if direction == "UP" else "down"
        return len(self.state[floor_id][dir_key])

    def get_total_waiting(self, floor_id):
        """특정 층의 모든 대기 승객 수"""
        return len(self.state[floor_id]["up"]) + len(self.state[floor_id]["down"])
    
    def get_total_system_waiting(self):
        """전체 시스템의 대기 승객 수"""
        total = 0
        for floor_id in range(self.floor_max):
            total += len(self.state[floor_id]["up"]) + len(self.state[floor_id]["down"])
        return total
    
    def get_all_calls(self):
        """
        모든 대기 호출 목록 반환
        Returns: [(floor, direction, count, oldest_timestamp), ...]
        """
        calls = []
        for floor_id in range(self.floor_max):
            for dir_key in ["up", "down"]:
                queue = self.state[floor_id][dir_key]
                if len(queue) > 0:
                    direction = "UP" if dir_key == "up" else "DOWN"
                    oldest_timestamp = min(p[1] for p in queue)
                    calls.append((floor_id, direction, len(queue), oldest_timestamp))
        return calls


class ElevatorState():
    def __init__(self, capacity):
        self.capacity = capacity
        self.passengers = []  # [destination, timestamp]
        self.current_floor = 0
        self.direction = "IDLE"

    def available_space(self):
        return self.capacity - len(self.passengers)
    
    def get_destinations(self):
        """탑승객 목적지 리스트"""
        return [p[0] for p in self.passengers]


# --- Atomic Models ---

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

            # timestamp: 승객 데이터에 포함된 생성 시간 사용 (정확한 시간)
            if event_logger:
                event_logger.log_event(
                    time=timestamp,  # 승객 생성 시간 사용
                    event_type='passenger_generated',
                    floor=floor_id + 1,
                    destination=destination_floor + 1,
                    direction=direction,
                    details=f"Passenger {passenger_id}"
                )
        return self.state


class Controller(AtomicDEVS):
    def __init__(self, num_elevators, floor_max):
        super().__init__("Controller")
        self.num_elevators = num_elevators
        self.floor_max = floor_max
        self.elevator_input = self.addInPort("elevator_input")
        self.elevator_output = self.addOutPort("elevator_output")

        self.state = {"count": 0}

    def timeAdvance(self):
        if self.state["count"] == self.num_elevators:
            return 0.01
        return INFINITY

    def extTransition(self, inputs):
        if self.elevator_input in inputs:
            self.state["count"] += 1
        return self.state

    def calculate_eta(self, elevator_name, target_floor, target_direction):
        """
        엘리베이터가 특정 호출(층, 방향)에 도달하여 서비스할 수 있는 예상 시간
        
        핵심: 엘리베이터가 해당 층에 도착했을 때 올바른 방향이어야 탑승 가능
        """
        e_state = elevator_states[elevator_name]
        current_floor = e_state.current_floor
        passengers = e_state.passengers
        
        # 빈 엘리베이터: 직접 이동 후 해당 방향으로 서비스
        if len(passengers) == 0:
            return abs(target_floor - current_floor)
        
        # 탑승객이 있는 경우
        destinations = e_state.get_destinations()
        
        # 현재 진행 방향 결정
        has_up_dest = any(d > current_floor for d in destinations)
        has_down_dest = any(d < current_floor for d in destinations)
        
        if has_up_dest and not has_down_dest:
            current_direction = "UP"
        elif has_down_dest and not has_up_dest:
            current_direction = "DOWN"
        else:
            # 양방향 목적지: 현재 저장된 방향 사용
            current_direction = e_state.direction if e_state.direction != "IDLE" else "UP"
        
        # ETA 계산
        if current_direction == "UP":
            max_dest = max(destinations)
            if target_direction == "UP" and target_floor >= current_floor:
                # 올라가면서 서비스 가능
                return target_floor - current_floor
            elif target_direction == "UP" and target_floor < current_floor:
                # 위로 갔다가 아래로 내려왔다가 다시 올라와야 함
                return (max_dest - current_floor) + (max_dest - 0) + target_floor + 10
            elif target_direction == "DOWN" and target_floor > current_floor:
                # 위로 올라가서 해당 층 지나친 후 DOWN 서비스
                return (max(max_dest, target_floor) - current_floor) + 2
            else:  # target_direction == "DOWN" and target_floor <= current_floor
                # 위로 끝까지 갔다가 내려오면서 서비스
                return (max_dest - current_floor) + (max_dest - target_floor)
        
        else:  # current_direction == "DOWN"
            min_dest = min(destinations)
            if target_direction == "DOWN" and target_floor <= current_floor:
                # 내려가면서 서비스 가능
                return current_floor - target_floor
            elif target_direction == "DOWN" and target_floor > current_floor:
                # 아래로 갔다가 위로 올라갔다가 다시 내려와야 함
                return (current_floor - min_dest) + (self.floor_max - 1 - min_dest) + (self.floor_max - 1 - target_floor) + 10
            elif target_direction == "UP" and target_floor < current_floor:
                # 아래로 내려가서 해당 층 지나친 후 UP 서비스
                return (current_floor - min(min_dest, target_floor)) + 2
            else:  # target_direction == "UP" and target_floor >= current_floor
                # 아래로 끝까지 갔다가 올라오면서 서비스
                return (current_floor - min_dest) + (target_floor - min_dest)

    def assign_calls_to_elevators(self, current_time):
        """
        모든 호출을 엘리베이터에 할당하고 각 엘리베이터의 다음 방향 결정
        
        알고리즘:
        1. 탑승객이 있는 엘리베이터: 탑승객 목적지 방향으로 이동
        2. 빈 엘리베이터: 가장 효율적인 호출에 할당
        3. 같은 호출에 여러 엘리베이터 할당 방지 (승객 수 고려)
        """
        global totalbuffer_state
        commands = {}
        
        # 모든 호출 수집
        all_calls = totalbuffer_state.get_all_calls()
        
        # 호출별 할당된 엘리베이터 용량 추적
        # key: (floor, direction), value: 할당된 총 용량
        assigned_capacity = {}
        
        # === 1단계: 탑승객이 있는 엘리베이터 처리 ===
        for name, e_state in elevator_states.items():
            if len(e_state.passengers) > 0:
                destinations = e_state.get_destinations()
                current_floor = e_state.current_floor
                
                has_up = any(d > current_floor for d in destinations)
                has_down = any(d < current_floor for d in destinations)
                
                if has_up and not has_down:
                    direction = "UP"
                elif has_down and not has_up:
                    direction = "DOWN"
                elif has_up and has_down:
                    # 현재 방향 유지, IDLE이면 가까운 쪽
                    if e_state.direction in ["UP", "DOWN"]:
                        direction = e_state.direction
                    else:
                        closest_up = min(d for d in destinations if d > current_floor)
                        closest_down = max(d for d in destinations if d < current_floor)
                        direction = "UP" if (closest_up - current_floor) <= (current_floor - closest_down) else "DOWN"
                else:
                    # 모든 목적지가 현재 층 (하차 예정)
                    direction = "IDLE"
                
                # 층 경계 체크
                if current_floor == 0 and direction == "DOWN":
                    direction = "UP" if has_up else "IDLE"
                if current_floor == self.floor_max - 1 and direction == "UP":
                    direction = "DOWN" if has_down else "IDLE"
                
                commands[name] = direction
                
                # 이 엘리베이터가 지나가는 경로의 호출들에 용량 할당
                if direction == "UP":
                    for call_floor, call_dir, count, _ in all_calls:
                        if call_dir == "UP" and current_floor <= call_floor:
                            key = (call_floor, call_dir)
                            assigned_capacity[key] = assigned_capacity.get(key, 0) + e_state.available_space()
                elif direction == "DOWN":
                    for call_floor, call_dir, count, _ in all_calls:
                        if call_dir == "DOWN" and call_floor <= current_floor:
                            key = (call_floor, call_dir)
                            assigned_capacity[key] = assigned_capacity.get(key, 0) + e_state.available_space()
        
        # === 2단계: 빈 엘리베이터에 호출 할당 ===
        empty_elevators = [name for name in elevator_states.keys() if name not in commands]
        
        # 호출이 없으면 IDLE
        if not all_calls:
            for name in empty_elevators:
                commands[name] = "IDLE"
            return commands
        
        # 각 빈 엘리베이터에 최적 호출 할당
        for name in empty_elevators:
            e_state = elevator_states[name]
            current_floor = e_state.current_floor
            capacity = e_state.capacity
            
            # 현재 층에 대기 승객 확인 (우선 처리)
            up_waiting = totalbuffer_state.get_waiting_count(current_floor, "UP")
            down_waiting = totalbuffer_state.get_waiting_count(current_floor, "DOWN")
            
            # 현재 층에 승객이 있고, 아직 충분히 할당되지 않았으면 처리
            if up_waiting > 0:
                key = (current_floor, "UP")
                already_assigned = assigned_capacity.get(key, 0)
                if already_assigned < up_waiting:
                    commands[name] = "UP"
                    assigned_capacity[key] = already_assigned + capacity
                    continue
            
            if down_waiting > 0:
                key = (current_floor, "DOWN")
                already_assigned = assigned_capacity.get(key, 0)
                if already_assigned < down_waiting:
                    commands[name] = "DOWN"
                    assigned_capacity[key] = already_assigned + capacity
                    continue
            
            # 가장 효율적인 호출 찾기
            best_call = None
            best_score = float('inf')
            
            for call_floor, call_dir, count, oldest_ts in all_calls:
                key = (call_floor, call_dir)
                already_assigned = assigned_capacity.get(key, 0)
                
                # 이미 충분히 할당되었으면 스킵
                if already_assigned >= count:
                    continue
                
                # ETA 계산
                eta = self.calculate_eta(name, call_floor, call_dir)
                
                # 대기 시간 고려 (오래 기다린 승객 우선)
                wait_time = current_time - oldest_ts
                
                # 점수 = ETA - 대기시간 보너스
                score = eta - (wait_time * 0.5)
                
                if score < best_score:
                    best_score = score
                    best_call = (call_floor, call_dir)
            
            if best_call:
                call_floor, call_dir = best_call
                key = best_call
                assigned_capacity[key] = assigned_capacity.get(key, 0) + capacity
                
                # 이동 방향 결정
                if call_floor > current_floor:
                    direction = "UP"
                elif call_floor < current_floor:
                    direction = "DOWN"
                else:
                    # 같은 층: 호출 방향
                    direction = call_dir
                
                # 층 경계 체크
                if current_floor == 0 and direction == "DOWN":
                    # 0층에서 DOWN은 불가, 호출이 위에 있으면 UP
                    direction = "UP" if call_floor > 0 or any(c[0] > 0 for c in all_calls) else "IDLE"
                if current_floor == self.floor_max - 1 and direction == "UP":
                    direction = "DOWN" if call_floor < self.floor_max - 1 or any(c[0] < self.floor_max - 1 for c in all_calls) else "IDLE"
                
                commands[name] = direction
            else:
                commands[name] = "IDLE"
        
        return commands

    def outputFnc(self):
        global elevator_states, totalbuffer_state
        
        if self.state["count"] != self.num_elevators:
            return {}
        
        current_time = self.time_last[0] if hasattr(self, 'time_last') else 0
        
        # 명령 생성
        commands = self.assign_calls_to_elevators(current_time)
        
        # 디버그 출력
        total_waiting = totalbuffer_state.get_total_system_waiting()
        
        for name in sorted(commands.keys()):
            e_state = elevator_states[name]
            #print(f"[Controller] {name} @ F{e_state.current_floor}: "
            #      f"Load={len(e_state.passengers)}, Cmd={commands[name]}")
        
        if total_waiting > 0:
            #print(f"[Controller] Total waiting: {total_waiting}")
            for floor in range(self.floor_max):
                up = totalbuffer_state.get_waiting_count(floor, "UP")
                down = totalbuffer_state.get_waiting_count(floor, "DOWN")
                if up > 0 or down > 0:
                    #print(f"  Floor {floor}: UP={up}, DOWN={down}")
                    pass

        # 명령 리스트 생성
        command_list = []
        for i in range(1, self.num_elevators + 1):
            elevator_name = f"Elevator_{i}"
            command_list.append(commands[elevator_name])
        
        return {self.elevator_output: command_list}

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
        global elevator_states, event_logger, metrics_collector
        est = elevator_states[self.name]

        if self.state == "UNLOAD":
            self.state = "WAIT_CMD"

        elif self.state == "MOVE":
            current_time = self.time_last[0]
            
            # 1. 이동 (IDLE이면 제자리)
            old_floor = est.current_floor
            if est.direction == "UP":
                est.current_floor += 1
            elif est.direction == "DOWN":
                est.current_floor -= 1

            if metrics_collector and est.direction != "IDLE":
                metrics_collector.record_energy(1) 
            
            # 로그 기록
            if event_logger:
                event_logger.log_event(
                    time=current_time,
                    event_type='elevator_moved',
                    floor=est.current_floor + 1,
                    elevator_id=self.name,
                    details=f"Direction: {est.direction}, From: F{old_floor+1}, Passengers: {len(est.passengers)}"
                )
            
               
            # 2. 하차
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
                for p in alighted_passengers:
                    passenger_id = p[2]
                    if metrics_collector:
                        metrics_collector.record_passenger_alighted(passenger_id, current_time)

                    if event_logger:
                        event_logger.log_event(
                            time=current_time,
                            event_type='passenger_alighted',
                            floor=est.current_floor + 1,
                            elevator_id=self.name,
                            details=f"Remaining load: {len(est.passengers)}"
                        )

            # 3. UNLOAD 상태로 전환
            self.state = "UNLOAD"

        return self.state

    def extTransition(self, inputs):
        global elevator_states, totalbuffer_state, event_logger, metrics_collector

        if self.state == "WAIT_CMD" and self.ctrl_in in inputs:
            msg = inputs[self.ctrl_in]
            #print(f"[{self.name}] Received: {msg}")
            
            if msg:
                my_cmd = msg[int(self.name.split("_")[1]) - 1]
                est = elevator_states[self.name]

                # 1. 방향 설정
                est.direction = my_cmd

                # 2. 탑승 (현재 층, 해당 방향 승객)
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
                            if metrics_collector:
                                metrics_collector.record_passenger_boarded(passenger_id, self.time_last[0])
                            
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

                # 3. 이동 상태로 전환
                self.state = "MOVE"

        return self.state

    def confTransition(self, inputs):
        self.intTransition()
        return self.extTransition(inputs)


class Building(CoupledDEVS):
    def __init__(self, floor_max, num_elevators, hourly_rate=5):
        super().__init__("Building")

        self.totalbuffer = TotalBuffer(floor_max)
        self.controller = Controller(num_elevators, floor_max)
        self.addSubModel(self.totalbuffer)
        self.addSubModel(self.controller)

        self.elevators = []
        for i in range(num_elevators):
            elev = Elevator(f"Elevator_{i+1}", capacity=4, init_delay=0.01 * (i + 1))
            self.addSubModel(elev)
            self.elevators.append(elev)

            self.connectPorts(elev.status_out, self.controller.elevator_input)
            self.connectPorts(self.controller.elevator_output, elev.ctrl_in)

        self.floor_models = []
        self.hourly_rate = [700, 500, 300, 200, 400, 500, 600, 200 ]
        for floor_id in range(floor_max):
            floor_model = Floor(floor_id, floor_max, self.hourly_rate[floor_id])
            self.addSubModel(floor_model)
            self.floor_models.append(floor_model)
            self.connectPorts(floor_model.floor_outport, self.totalbuffer.floor_input)

def reset_simulation():
    """Reset global states for new episode"""
    global event_logger,elevator_states, totalbuffer_state, metrics_collector
    elevator_states = {}
    totalbuffer_state = None
    if metrics_collector:
        metrics_collector.reset()

def simulate(episode, log_events=False):
    global event_logger, elevator_states, totalbuffer_state, metrics_collector
    reset_simulation()
    metrics_collector = MetricsCollector()
    building = Building(floor_max=8, num_elevators=3, hourly_rate=60)
    sim = Simulator(building)
    sim.setTerminationTime(500)
    sim.simulate()

    #print("\n=== Simulation Complete ===")
    #print(f"Final Elevator States:")
    #for name, state in elevator_states.items():
    #    print(f"  {name}: Floor {state.current_floor}, Direction {state.direction}, Load {len(state.passengers)}")
#
    #print(f"\nRemaining Waiting Passengers:")
    #total_remaining = 0
    #for floor in range(8):
    #    up_count = len(totalbuffer_state.state[floor]["up"])
    #    down_count = len(totalbuffer_state.state[floor]["down"])
    #    if up_count > 0 or down_count > 0:
    #        print(f"  Floor {floor}: UP={up_count}, DOWN={down_count}")
    #        total_remaining += up_count + down_count
    #
    #if total_remaining == 0:
    #    print("  None - All passengers served!")
    #else:
    #    print(f"\n  WARNING: {total_remaining} passenger(s) still waiting!")
    stats = metrics_collector.get_episode_stats()
    return stats

if __name__ == "__main__":
    event_logger = EventLogger("simulation_events.csv")
    event_logger.open()
    print("Event logging started: simulation_events_v2.csv")
    all_stats = []
    log_events  = True
    for episode in range(1, 100 + 1):
        print(f"\nEvaluation Episode {episode}/{100}")
        stats = simulate(
            episode, 
            log_events= log_events and episode == 1  # Log first episode only
        )
        all_stats.append(stats)
        print(f"  Passengers Served: {stats['passengers_served']}")
        print(f"  Avg Wait Time: {stats['avg_waiting_time']:.2f}s")
        print(f"  Avg Travel Time: {stats['avg_travel_time']:.2f}s")
        print(f"  Total Energy: {stats['total_energy']}")

            # Summary statistics
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    avg_passengers = np.mean([s['passengers_served'] for s in all_stats])
    avg_wait = np.mean([s['avg_waiting_time'] for s in all_stats])
    avg_travel = np.mean([s['avg_travel_time'] for s in all_stats])
    avg_energy = np.mean([s['total_energy'] for s in all_stats])
    
    print(f"Average Passengers Served: {avg_passengers:.1f}")
    print(f"Average Wait Time: {avg_wait:.2f}s")
    print(f"Average Travel Time: {avg_travel:.2f}s")
    print(f"Average Energy: {avg_energy:.1f}")
    event_logger.close()