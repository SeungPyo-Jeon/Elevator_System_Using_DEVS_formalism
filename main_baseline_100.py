from pypdevs.DEVS import *
from pypdevs.infinity import INFINITY
from pypdevs.simulator import Simulator
import random
import csv
import math

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

# --- Data Structures ---
class TowerState():
    def __init__(self, floor_max):
        self.state = {}
        self.floor_max = floor_max
        for idx in range(floor_max):
            self.state[idx] = {"up": [], "down": []}

    def add_passenger(self, passenger):
        floor_id = passenger[0]
        destination_floor = passenger[2]
        timestamp = passenger[3]

        direction = "up" if destination_floor > floor_id else "down"
        self.state[floor_id][direction].append([destination_floor, timestamp])

    def pop_passengers(self, floor_id, available_space, direction):
        """�밴컼 �묒듅: �대떦 痢듭쓽 �대떦 諛⑺뼢 ��湲곗뿴�먯꽌 �밴컼 �쒓굅 諛� 諛섑솚"""
        if direction == "IDLE":
            return []

        dir_key = "up" if direction == "UP" else "down"
        pop_list = []

        queue = self.state[floor_id][dir_key]
        while queue and len(pop_list) < available_space:
            pop_list.append(queue.pop(0))

        return pop_list

    def get_waiting_count(self, floor_id, direction):
        """�뱀젙 痢듭쓽 �뱀젙 諛⑺뼢 ��湲� �밴컼 ��"""
        if direction == "IDLE":
            return 0
        dir_key = "up" if direction == "UP" else "down"
        return len(self.state[floor_id][dir_key])

    def get_total_waiting(self, floor_id):
        """�뱀젙 痢듭쓽 紐⑤뱺 ��湲� �밴컼 ��"""
        return len(self.state[floor_id]["up"]) + len(self.state[floor_id]["down"])
    
    def get_total_system_waiting(self):
        """�꾩껜 �쒖뒪�쒖쓽 ��湲� �밴컼 ��"""
        total = 0
        for floor_id in range(self.floor_max):
            total += len(self.state[floor_id]["up"]) + len(self.state[floor_id]["down"])
        return total
    
    def get_all_calls(self):
        """
        紐⑤뱺 ��湲� �몄텧 紐⑸줉 諛섑솚
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
        """�묒듅媛� 紐⑹쟻吏� 由ъ뒪��"""
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
        global totalbuffer_state, event_logger
        if self.floor_input in inputs:
            passenger = inputs[self.floor_input]
            totalbuffer_state.add_passenger(passenger)

            floor_id, passenger_id, destination_floor, timestamp = passenger
            direction = "UP" if destination_floor > floor_id else "DOWN"
            # timestamp: �밴컼 �곗씠�곗뿉 �ы븿�� �앹꽦 �쒓컙 �ъ슜 (�뺥솗�� �쒓컙)

            if event_logger:
                event_logger.log_event(
                    time=timestamp,  # �밴컼 �앹꽦 �쒓컙 �ъ슜
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
        �섎━踰좎씠�곌� �뱀젙 �몄텧(痢�, 諛⑺뼢)�� �꾨떖�섏뿬 �쒕퉬�ㅽ븷 �� �덈뒗 �덉긽 �쒓컙
        
        �듭떖: �섎━踰좎씠�곌� �대떦 痢듭뿉 �꾩갑�덉쓣 �� �щ컮瑜� 諛⑺뼢�댁뼱�� �묒듅 媛���
        """
        e_state = elevator_states[elevator_name]
        current_floor = e_state.current_floor
        passengers = e_state.passengers
        
        # 鍮� �섎━踰좎씠��: 吏곸젒 �대룞 �� �대떦 諛⑺뼢�쇰줈 �쒕퉬��
        if len(passengers) == 0:
            return abs(target_floor - current_floor)
        
        # �묒듅媛앹씠 �덈뒗 寃쎌슦
        destinations = e_state.get_destinations()
        
        # �꾩옱 吏꾪뻾 諛⑺뼢 寃곗젙
        has_up_dest = any(d > current_floor for d in destinations)
        has_down_dest = any(d < current_floor for d in destinations)
        
        if has_up_dest and not has_down_dest:
            current_direction = "UP"
        elif has_down_dest and not has_up_dest:
            current_direction = "DOWN"
        else:
            # �묐갑�� 紐⑹쟻吏�: �꾩옱 ���λ맂 諛⑺뼢 �ъ슜
            current_direction = e_state.direction if e_state.direction != "IDLE" else "UP"
        
        # ETA 怨꾩궛
        if current_direction == "UP":
            max_dest = max(destinations)
            if target_direction == "UP" and target_floor >= current_floor:
                # �щ씪媛�硫댁꽌 �쒕퉬�� 媛���
                return target_floor - current_floor
            elif target_direction == "UP" and target_floor < current_floor:
                # �꾨줈 媛붾떎媛� �꾨옒濡� �대젮�붾떎媛� �ㅼ떆 �щ씪���� ��
                return (max_dest - current_floor) + (max_dest - 0) + target_floor + 10
            elif target_direction == "DOWN" and target_floor > current_floor:
                # �꾨줈 �щ씪媛��� �대떦 痢� 吏��섏튇 �� DOWN �쒕퉬��
                return (max(max_dest, target_floor) - current_floor) + 2
            else:  # target_direction == "DOWN" and target_floor <= current_floor
                # �꾨줈 �앷퉴吏� 媛붾떎媛� �대젮�ㅻ㈃�� �쒕퉬��
                return (max_dest - current_floor) + (max_dest - target_floor)
        
        else:  # current_direction == "DOWN"
            min_dest = min(destinations)
            if target_direction == "DOWN" and target_floor <= current_floor:
                # �대젮媛�硫댁꽌 �쒕퉬�� 媛���
                return current_floor - target_floor
            elif target_direction == "DOWN" and target_floor > current_floor:
                # �꾨옒濡� 媛붾떎媛� �꾨줈 �щ씪媛붾떎媛� �ㅼ떆 �대젮���� ��
                return (current_floor - min_dest) + (self.floor_max - 1 - min_dest) + (self.floor_max - 1 - target_floor) + 10
            elif target_direction == "UP" and target_floor < current_floor:
                # �꾨옒濡� �대젮媛��� �대떦 痢� 吏��섏튇 �� UP �쒕퉬��
                return (current_floor - min(min_dest, target_floor)) + 2
            else:  # target_direction == "UP" and target_floor >= current_floor
                # �꾨옒濡� �앷퉴吏� 媛붾떎媛� �щ씪�ㅻ㈃�� �쒕퉬��
                return (current_floor - min_dest) + (target_floor - min_dest)

    def assign_calls_to_elevators(self, current_time):
        """
        紐⑤뱺 �몄텧�� �섎━踰좎씠�곗뿉 �좊떦�섍퀬 媛� �섎━踰좎씠�곗쓽 �ㅼ쓬 諛⑺뼢 寃곗젙
        
        �뚭퀬由ъ쬁:
        1. �묒듅媛앹씠 �덈뒗 �섎━踰좎씠��: �묒듅媛� 紐⑹쟻吏� 諛⑺뼢�쇰줈 �대룞
        2. 鍮� �섎━踰좎씠��: 媛��� �⑥쑉�곸씤 �몄텧�� �좊떦
        3. 媛숈� �몄텧�� �щ윭 �섎━踰좎씠�� �좊떦 諛⑹� (�밴컼 �� 怨좊젮)
        """
        global totalbuffer_state
        commands = {}
        
        # 紐⑤뱺 �몄텧 �섏쭛
        all_calls = totalbuffer_state.get_all_calls()
        
        # �몄텧蹂� �좊떦�� �섎━踰좎씠�� �⑸웾 異붿쟻
        # key: (floor, direction), value: �좊떦�� 珥� �⑸웾
        assigned_capacity = {}
        
        # === 1�④퀎: �묒듅媛앹씠 �덈뒗 �섎━踰좎씠�� 泥섎━ ===
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
                    # �꾩옱 諛⑺뼢 �좎�, IDLE�대㈃ 媛�源뚯슫 履�
                    if e_state.direction in ["UP", "DOWN"]:
                        direction = e_state.direction
                    else:
                        closest_up = min(d for d in destinations if d > current_floor)
                        closest_down = max(d for d in destinations if d < current_floor)
                        direction = "UP" if (closest_up - current_floor) <= (current_floor - closest_down) else "DOWN"
                else:
                    # 紐⑤뱺 紐⑹쟻吏�媛� �꾩옱 痢� (�섏감 �덉젙)
                    direction = "IDLE"
                
                # 痢� 寃쎄퀎 泥댄겕
                if current_floor == 0 and direction == "DOWN":
                    direction = "UP" if has_up else "IDLE"
                if current_floor == self.floor_max - 1 and direction == "UP":
                    direction = "DOWN" if has_down else "IDLE"
                
                commands[name] = direction
                
                # �� �섎━踰좎씠�곌� 吏��섍��� 寃쎈줈�� �몄텧�ㅼ뿉 �⑸웾 �좊떦
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
        
        # === 2�④퀎: 鍮� �섎━踰좎씠�곗뿉 �몄텧 �좊떦 ===
        empty_elevators = [name for name in elevator_states.keys() if name not in commands]
        
        # �몄텧�� �놁쑝硫� IDLE
        if not all_calls:
            for name in empty_elevators:
                commands[name] = "IDLE"
            return commands
        
        # 媛� 鍮� �섎━踰좎씠�곗뿉 理쒖쟻 �몄텧 �좊떦
        for name in empty_elevators:
            e_state = elevator_states[name]
            current_floor = e_state.current_floor
            capacity = e_state.capacity
            
            # �꾩옱 痢듭뿉 ��湲� �밴컼 �뺤씤 (�곗꽑 泥섎━)
            up_waiting = totalbuffer_state.get_waiting_count(current_floor, "UP")
            down_waiting = totalbuffer_state.get_waiting_count(current_floor, "DOWN")
            
            # �꾩옱 痢듭뿉 �밴컼�� �덇퀬, �꾩쭅 異⑸텇�� �좊떦�섏� �딆븯�쇰㈃ 泥섎━
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
            
            # 媛��� �⑥쑉�곸씤 �몄텧 李얘린
            best_call = None
            best_score = float('inf')
            
            for call_floor, call_dir, count, oldest_ts in all_calls:
                key = (call_floor, call_dir)
                already_assigned = assigned_capacity.get(key, 0)
                
                # �대� 異⑸텇�� �좊떦�섏뿀�쇰㈃ �ㅽ궢
                if already_assigned >= count:
                    continue
                
                # ETA 怨꾩궛
                eta = self.calculate_eta(name, call_floor, call_dir)
                
                # ��湲� �쒓컙 怨좊젮 (�ㅻ옒 湲곕떎由� �밴컼 �곗꽑)
                wait_time = current_time - oldest_ts
                
                # �먯닔 = ETA - ��湲곗떆媛� 蹂대꼫��
                score = eta - (wait_time * 0.5)
                
                if score < best_score:
                    best_score = score
                    best_call = (call_floor, call_dir)
            
            if best_call:
                call_floor, call_dir = best_call
                key = best_call
                assigned_capacity[key] = assigned_capacity.get(key, 0) + capacity
                
                # �대룞 諛⑺뼢 寃곗젙
                if call_floor > current_floor:
                    direction = "UP"
                elif call_floor < current_floor:
                    direction = "DOWN"
                else:
                    # 媛숈� 痢�: �몄텧 諛⑺뼢
                    direction = call_dir
                
                # 痢� 寃쎄퀎 泥댄겕
                if current_floor == 0 and direction == "DOWN":
                    # 0痢듭뿉�� DOWN�� 遺덇�, �몄텧�� �꾩뿉 �덉쑝硫� UP
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
        
        # 紐낅졊 �앹꽦
        commands = self.assign_calls_to_elevators(current_time)
        
        # �붾쾭洹� 異쒕젰
        total_waiting = totalbuffer_state.get_total_system_waiting()
        
        for name in sorted(commands.keys()):
            e_state = elevator_states[name]
            print(f"[Controller] {name} @ F{e_state.current_floor}: "
                  f"Load={len(e_state.passengers)}, Cmd={commands[name]}")
        
        if total_waiting > 0:
            print(f"[Controller] Total waiting: {total_waiting}")
            for floor in range(self.floor_max):
                up = totalbuffer_state.get_waiting_count(floor, "UP")
                down = totalbuffer_state.get_waiting_count(floor, "DOWN")
                if up > 0 or down > 0:
                    print(f"  Floor {floor}: UP={up}, DOWN={down}")
        
        # 紐낅졊 由ъ뒪�� �앹꽦
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
        global elevator_states, event_logger
        est = elevator_states[self.name]

        if self.state == "UNLOAD":
            self.state = "WAIT_CMD"

        elif self.state == "MOVE":
            current_time = self.time_last[0]
            
            # 1. �대룞 (IDLE�대㈃ �쒖옄由�)
            old_floor = est.current_floor
            if est.direction == "UP":
                est.current_floor += 1
            elif est.direction == "DOWN":
                est.current_floor -= 1

            # 濡쒓렇 湲곕줉
            if event_logger:
                event_logger.log_event(
                    time=current_time,
                    event_type='elevator_moved',
                    floor=est.current_floor + 1,
                    elevator_id=self.name,
                    details=f"Direction: {est.direction}, From: F{old_floor+1}, Passengers: {len(est.passengers)}"
                )

            # 2. �섏감
            staying_passengers = []
            alighted_count = 0
            for p in est.passengers:
                if p[0] == est.current_floor:
                    alighted_count += 1
                else:
                    staying_passengers.append(p)
            est.passengers = staying_passengers

            if alighted_count > 0:
                print(f"[{self.name} @ F{est.current_floor}] Alighted {alighted_count}. Load: {len(est.passengers)}")

                if event_logger:
                    for _ in range(alighted_count):
                        event_logger.log_event(
                            time=current_time,
                            event_type='passenger_alighted',
                            floor=est.current_floor + 1,
                            elevator_id=self.name,
                            details=f"Remaining load: {len(est.passengers)}"
                        )

            # 3. UNLOAD �곹깭濡� �꾪솚
            self.state = "UNLOAD"

        return self.state

    def extTransition(self, inputs):
        global elevator_states, totalbuffer_state, event_logger

        if self.state == "WAIT_CMD" and self.ctrl_in in inputs:
            msg = inputs[self.ctrl_in]
            print(f"[{self.name}] Received: {msg}")
            
            if msg:
                my_cmd = msg[int(self.name.split("_")[1]) - 1]
                est = elevator_states[self.name]

                # 1. 諛⑺뼢 �ㅼ젙
                est.direction = my_cmd

                # 2. �묒듅 (�꾩옱 痢�, �대떦 諛⑺뼢 �밴컼)
                available = est.available_space()
                if available > 0 and est.direction != "IDLE":
                    new_passengers = totalbuffer_state.pop_passengers(
                        est.current_floor, available, est.direction
                    )
                    if new_passengers:
                        est.passengers.extend(new_passengers)
                        print(f"[{self.name} @ F{est.current_floor}] Boarded {len(new_passengers)} ({est.direction}). Load: {len(est.passengers)}")

                        current_time = self.time_last[0]
                        if event_logger:
                            for passenger in new_passengers:
                                event_logger.log_event(
                                    time=current_time,
                                    event_type='passenger_boarded',
                                    floor=est.current_floor + 1,
                                    destination=passenger[0] + 1,
                                    direction=est.direction,
                                    elevator_id=self.name,
                                    details=f"Current load: {len(est.passengers)}"
                                )

                # 3. �대룞 �곹깭濡� �꾪솚
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


if __name__ == "__main__":
    event_logger = EventLogger("simulation_events.csv")
    event_logger.open()
    print("Event logging started: simulation_events_v2.csv")

    try:
        building = Building(floor_max=8, num_elevators=3, hourly_rate=60)
        sim = Simulator(building)
        sim.setTerminationTime(500)
        sim.simulate()

        print("\n=== Simulation Complete ===")
        print(f"Final Elevator States:")
        for name, state in elevator_states.items():
            print(f"  {name}: Floor {state.current_floor}, Direction {state.direction}, Load {len(state.passengers)}")

        print(f"\nRemaining Waiting Passengers:")
        total_remaining = 0
        for floor in range(8):
            up_count = len(totalbuffer_state.state[floor]["up"])
            down_count = len(totalbuffer_state.state[floor]["down"])
            if up_count > 0 or down_count > 0:
                print(f"  Floor {floor}: UP={up_count}, DOWN={down_count}")
                total_remaining += up_count + down_count
        
        if total_remaining == 0:
            print("  None - All passengers served!")
        else:
            print(f"\n  WARNING: {total_remaining} passenger(s) still waiting!")

    finally:
        event_logger.close()
        print("\nEvent logging completed: simulation_events_v2.csv")