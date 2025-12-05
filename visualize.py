#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Elevator Simulation Visualizer (Fixed Version)
"""

import csv
import time
import os
import sys
import re
from collections import defaultdict

# Keyboard input handling
try:
    import msvcrt  # Windows
    WINDOWS = True
except ImportError:
    import tty
    import termios
    import select
    WINDOWS = False

# ANSI Escape Codes
CURSOR_HOME = "\033[H"
CLEAR_SCREEN = "\033[2J"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

class ElevatorVisualizer:
    def __init__(self, events_file="simulation_events.csv", floor_max=8, num_elevators=3):
        if os.name == 'nt':
            os.system('')

        self.events_file = events_file
        self.floor_max = floor_max
        self.num_elevators = num_elevators

        # Current state
        self.elevator_positions = {f"Elevator_{i}": 1 for i in range(1, num_elevators + 1)}
        self.elevator_passengers = {f"Elevator_{i}": 0 for i in range(1, num_elevators + 1)}
        self.elevator_directions = {f"Elevator_{i}": "IDLE" for i in range(1, num_elevators + 1)}
        self.waiting_passengers = defaultdict(lambda: {'UP': 0, 'DOWN': 0})

        self.events = []
        self.load_events()
        
        self.frames = []
        self.build_frames()

    def load_events(self):
        try:
            with open(self.events_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['time']:
                        self.events.append(row)
            self.events.sort(key=lambda x: float(x['time']))
            print(f"Loaded {len(self.events)} events")
        except FileNotFoundError:
            print(f"Error: File '{self.events_file}' not found.")
            sys.exit(1)

    def build_frames(self, time_step=1.0):
        self.frames = []
        if not self.events:
            return
        max_time = float(self.events[-1]['time'])
        current_time = 0.0
        event_idx = 0
        
        while current_time <= max_time + time_step:
            # 현재 시간까지의 이벤트만 포함
            frame_events = []
            while event_idx < len(self.events) and float(self.events[event_idx]['time']) <= current_time:
                frame_events.append(self.events[event_idx])
                event_idx += 1
            
            self.frames.append({
                'time': current_time, 
                'events': frame_events,
                'cumulative_event_count': event_idx
            })
            current_time += time_step
        
        print(f"Built {len(self.frames)} frames")

    def parse_passengers_from_details(self, details):
        """details 문자열에서 Passengers 수 추출"""
        # "Direction: UP, From: F1, Passengers: 2" 형태
        # 또는 "Direction: IDLE, From: F1, Passengers: 0" 형태
        match = re.search(r'Passengers:\s*(\d+)', details)
        if match:
            return int(match.group(1))
        return 0

    def parse_direction_from_details(self, details):
        """details 문자열에서 Direction 추출"""
        match = re.search(r'Direction:\s*(\w+)', details)
        if match:
            return match.group(1)
        return "IDLE"

    def draw_building(self, current_time, recent_events, paused=False, frame_num=0, total_frames=0):
        lines = []
        lines.append(CURSOR_HOME)

        # Header
        lines.append("=" * 80)
        header = f"  DEVS ELEVATOR SIMULATION - Time: {current_time:.2f}s"
        if paused:
            header += " [PAUSED]"
        header += f" (Frame {frame_num}/{total_frames})"
        lines.append(f"{header:<80}")
        lines.append("=" * 80)
        
        # 엘리베이터 상태 헤더
        header_elev = " " * 12
        for i in range(1, self.num_elevators + 1):
            header_elev += f"{f'E{i}':^10}"
        lines.append(f"{header_elev:<80}")
        lines.append("-" * 80)

        # Draw each floor
        for floor in range(self.floor_max, 0, -1):
            floor_str = f"Floor {floor}: "
            
            elevator_part = ""
            for i in range(1, self.num_elevators + 1):
                elev_name = f"Elevator_{i}"
                cell_width = 10
                
                if self.elevator_positions[elev_name] == floor:
                    passengers = self.elevator_passengers[elev_name]
                    direction = self.elevator_directions[elev_name]
                    
                    # 방향 표시 화살표
                    if direction == "UP":
                        arrow = "↑"
                    elif direction == "DOWN":
                        arrow = "↓"
                    else:
                        arrow = "·"
                    
                    content = f"[{arrow}{passengers:02d}]"
                    elevator_part += f"{content:^{cell_width}}"
                else:
                    elevator_part += f"{'│':^{cell_width}}"

            # 대기 승객 표시
            up_count = self.waiting_passengers[floor]['UP']
            down_count = self.waiting_passengers[floor]['DOWN']

            if up_count > 0 or down_count > 0:
                waiting_part = f"  ↑{up_count:2d} ↓{down_count:2d}"
            else:
                waiting_part = ""

            line_content = f"{floor_str:<12}{elevator_part}{waiting_part}"
            lines.append(f"{line_content:<80}")

        lines.append("-" * 80)

        # 총 대기 승객 수
        total_waiting = sum(
            self.waiting_passengers[f]['UP'] + self.waiting_passengers[f]['DOWN']
            for f in range(1, self.floor_max + 1)
        )
        total_in_elevator = sum(self.elevator_passengers.values())
        lines.append(f"{'Total Waiting: ' + str(total_waiting) + '  |  In Elevators: ' + str(total_in_elevator):<80}")
        lines.append("-" * 80)

        # Recent events
        lines.append(f"{'Recent Events:':<80}")
        max_events_display = 5
        display_events = recent_events[-max_events_display:]
        
        for i in range(max_events_display):
            if i < len(display_events):
                lines.append(f"  {display_events[i]:<78}")
            else:
                lines.append(" " * 80)

        lines.append("-" * 80)
        lines.append(f"{'Controls: [SPACE] Play/Pause  [←] Prev  [→] Next  [Q] Quit':<80}")
        lines.append(f"{'Legend: [↑02] = Going UP with 2 passengers  |  ↑3 ↓2 = 3 UP, 2 DOWN waiting':<80}")
        lines.append(" " * 80)

        return "\n".join(lines)

    def process_event(self, event):
        event_type = event['event_type']
        event_msg = ""
        
        try:
            if event_type == 'passenger_generated':
                floor = int(event['floor']) if event['floor'] else None
                dest = int(event['destination']) if event['destination'] else None
                direction = event['direction']
                if floor and direction:
                    self.waiting_passengers[floor][direction] += 1
                    event_msg = f"[{float(event['time']):>6.1f}] NEW: F{floor}→F{dest} ({direction})"

            elif event_type == 'elevator_moved':
                elevator_id = event['elevator_id']
                floor = int(event['floor']) if event['floor'] else None
                details = event.get('details', '')
                
                if floor and elevator_id:
                    self.elevator_positions[elevator_id] = floor
                    self.elevator_passengers[elevator_id] = self.parse_passengers_from_details(details)
                    self.elevator_directions[elevator_id] = self.parse_direction_from_details(details)
                    
                    direction = self.elevator_directions[elevator_id]
                    passengers = self.elevator_passengers[elevator_id]
                    
                    # IDLE이 아닐 때만 이벤트 메시지 표시
                    if direction != "IDLE":
                        elev_num = elevator_id.split('_')[1]
                        event_msg = f"[{float(event['time']):>6.1f}] E{elev_num}: F{floor} ({direction}, {passengers}p)"

            elif event_type == 'passenger_boarded':
                elevator_id = event['elevator_id']
                floor = int(event['floor']) if event['floor'] else None
                direction = event.get('direction', '')

                if floor and elevator_id and direction:
                    self.waiting_passengers[floor][direction] = max(0, self.waiting_passengers[floor][direction] - 1)
                    # elevator_passengers는 elevator_moved에서 이미 업데이트됨
                    elev_num = elevator_id.split('_')[1]
                    event_msg = f"[{float(event['time']):>6.1f}] BOARD: E{elev_num} at F{floor} ({direction})"

            elif event_type == 'passenger_alighted':
                elevator_id = event['elevator_id']
                floor = int(event['floor']) if event['floor'] else None
                if floor and elevator_id:
                    # elevator_passengers는 elevator_moved에서 이미 업데이트됨
                    elev_num = elevator_id.split('_')[1]
                    event_msg = f"[{float(event['time']):>6.1f}] EXIT: E{elev_num} at F{floor}"

        except (ValueError, KeyError) as e:
            pass
        
        return event_msg

    def reset_state(self):
        self.elevator_positions = {f"Elevator_{i}": 1 for i in range(1, self.num_elevators + 1)}
        self.elevator_passengers = {f"Elevator_{i}": 0 for i in range(1, self.num_elevators + 1)}
        self.elevator_directions = {f"Elevator_{i}": "IDLE" for i in range(1, self.num_elevators + 1)}
        self.waiting_passengers = defaultdict(lambda: {'UP': 0, 'DOWN': 0})

    def apply_events_up_to_frame(self, frame_idx):
        """프레임까지의 모든 이벤트 적용"""
        self.reset_state()
        recent_events = []
        
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return recent_events
        
        # 이전 프레임까지의 이벤트 개수
        prev_event_count = self.frames[frame_idx - 1]['cumulative_event_count'] if frame_idx > 0 else 0
        
        # 현재 프레임까지의 모든 이벤트 처리
        for i in range(self.frames[frame_idx]['cumulative_event_count']):
            event = self.events[i]
            msg = self.process_event(event)
            
            # 현재 프레임의 이벤트만 recent_events에 추가
            if i >= prev_event_count and msg:
                recent_events.append(msg)
        
        return recent_events

    def get_keyboard_input(self):
        if WINDOWS:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b' ':
                    return 'pause'
                elif key in (b'q', b'Q'):
                    return 'quit'
                elif key == b'\xe0':
                    key2 = msvcrt.getch()
                    if key2 == b'K':
                        return 'prev'
                    elif key2 == b'M':
                        return 'next'
        else:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == ' ':
                    return 'pause'
                elif key.lower() == 'q':
                    return 'quit'
                elif key == '\x1b':
                    extra = sys.stdin.read(2)
                    if extra == '[D':
                        return 'prev'
                    elif extra == '[C':
                        return 'next'
        return None

    def animate_interactive(self, speed=0.1, time_step=1.0):
        self.build_frames(time_step)

        sys.stdout.write(CLEAR_SCREEN + HIDE_CURSOR)
        sys.stdout.flush()
        print("Starting interactive visualization...")
        time.sleep(1)

        # Non-blocking input setup for Unix
        if not WINDOWS:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

        frame_idx = 0
        paused = False

        try:
            while frame_idx < len(self.frames):
                current_frame = self.frames[frame_idx]
                
                # 현재 프레임까지의 이벤트 적용
                recent_events = self.apply_events_up_to_frame(frame_idx)

                display = self.draw_building(
                    current_frame['time'],
                    recent_events,
                    paused,
                    frame_idx + 1,
                    len(self.frames)
                )
                
                sys.stdout.write(display)
                sys.stdout.flush()

                key = self.get_keyboard_input()

                if key == 'quit':
                    break
                elif key == 'pause':
                    paused = not paused
                    time.sleep(0.1)
                elif key == 'prev':
                    frame_idx = max(0, frame_idx - 1)
                    time.sleep(0.05)
                elif key == 'next':
                    frame_idx = min(len(self.frames) - 1, frame_idx + 1)
                    time.sleep(0.05)
                elif not paused:
                    frame_idx += 1
                    time.sleep(speed)
                else:
                    time.sleep(0.05)

            sys.stdout.write(SHOW_CURSOR + "\n")
            print("SIMULATION COMPLETED")

        except KeyboardInterrupt:
            sys.stdout.write(SHOW_CURSOR + "\n")
            print("\nVisualization stopped")
        finally:
            if not WINDOWS:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def print_summary(self):
        event_counts = defaultdict(int)
        for event in self.events:
            event_counts[event['event_type']] += 1
        
        print("\n" + "=" * 80)
        print("  EVENT SUMMARY")
        print("=" * 80)
        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type:<25}: {count:>5}")
        print("=" * 80)
        
        # 승객 처리 확인
        generated = event_counts.get('passenger_generated', 0)
        alighted = event_counts.get('passenger_alighted', 0)
        
        if generated == alighted:
            print(f"  ✓ All {generated} passengers successfully served!")
        else:
            print(f"  ⚠ Generated: {generated}, Alighted: {alighted}, Remaining: {generated - alighted}")
        print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='simulation_events.csv')
    parser.add_argument('--speed', type=float, default=0.1)
    parser.add_argument('--step', type=float, default=1.0)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()

    viz = ElevatorVisualizer(events_file=args.file)
    if args.summary:
        viz.print_summary()
    else:
        viz.animate_interactive(speed=args.speed, time_step=args.step)
        viz.print_summary()


if __name__ == "__main__":
    main()