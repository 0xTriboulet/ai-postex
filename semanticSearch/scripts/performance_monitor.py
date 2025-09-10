"""
Process Monitor Script for CPU and Memory Usage Logging

This script continuously monitors a target process by name (e.g., "rundll32.exe") and logs its 
resource usage statistics to a CSV file at fixed intervals (100ms). It captures CPU usage, 
Working Set (RSS), Private Memory, and Virtual Memory metrics.

Features:
- Detects and attaches to a running process by name.
- Records metrics with timestamps into a CSV file.
- Creates the output directory if it doesn't exist.
- Provides live console output of recorded stats.
- Automatically handles process disappearance or access errors.

Configuration:
- `TARGET_PROCESS_NAME`: Name of the process to monitor.
- `OUTPUT_DIRECTORY`: Path to the directory where the CSV file will be saved.
- `CSV_FILENAME`: Name of the output CSV file.

CSV Output Columns:
- Timestamp (in YYYY-MM-DD HH:MM:SS.mmm format)
- CPU Usage (%)
- Working Set (Bytes)
- Private Memory (Bytes)
- Virtual Memory (Bytes)

Usage:
    python monitor_process.py

Requirements:
- psutil
- Python 3.x

Note:
- CPU usage is recorded as a percentage of a single core.
- The script flushes to disk after each write to avoid data loss.

"""

import psutil
import time
import csv
import os
from datetime import datetime

# Configuration
TARGET_PROCESS_NAME = "rundll32.exe"  # Change this to your target process
OUTPUT_DIRECTORY = "../..//nextgen_postex/process_stats/"  # Change this to your desired output path
CSV_FILENAME = "rundll32_4_semantic_search_process.csv"
CSV_PATH = os.path.join(OUTPUT_DIRECTORY, CSV_FILENAME)

# Ensure output directory exists
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# CSV Headers
CSV_HEADERS = ["Timestamp", "CPU_Usage(%)", "WorkingSet(Bytes)", "PM(Bytes)", "VM(Bytes)"]


def get_process_metrics(proc):
    """Fetches CPU and Memory metrics for a given process object."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Human-readable timestamp
        cpu_usage = proc.cpu_percent(interval=None)  # CPU usage
        mem_info = proc.memory_info()

        return [
            timestamp,
            cpu_usage, # CPU usge as a fraction of available logical cores
            mem_info.rss,  # Working Set (Bytes)
            mem_info.private,  # Private Memory (Bytes)
            mem_info.vms  # Virtual Memory (Bytes)
        ]
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None

def monitor_process(target_name):
    """Monitors the target process and writes data to a CSV file."""
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADERS)  # Write headers

        print(f"Monitoring process: {target_name} | Output: {CSV_PATH}")
        
        while True:
            found = False
            for proc in psutil.process_iter(attrs=['pid', 'name']):
                if proc.info['name'].lower() == target_name.lower():
                    found = True
                    metrics = get_process_metrics(proc)
                    if metrics:
                        writer.writerow(metrics)
                        file.flush()  # Ensure data is written immediately
                        print(metrics)  # Print to console
                    break  # Process found, no need to check further
            
            if not found:
                print(f"Process {target_name} not found.")
            
            time.sleep(0.1)  # 100ms sampling interval

if __name__ == "__main__":
    monitor_process(TARGET_PROCESS_NAME)
