#!/usr/bin/env python3
"""
Real-time Power Monitoring for LLM Energy Lab
Measures actual power consumption on Linux systems using RAPL (Running Average Power Limit)
"""

import time
import os
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class PowerReading:
    """Single power measurement reading"""
    timestamp: float
    package_watts: float  # CPU package power
    cores_watts: float    # CPU cores power
    uncore_watts: float   # Uncore (memory controller, etc)
    dram_watts: float     # DRAM power
    total_watts: float    # Sum of all components


class RAPLMonitor:
    """Monitor power consumption using RAPL (Intel/AMD CPUs)"""
    
    def __init__(self):
        self.rapl_base = Path("/sys/class/powercap")
        self.rapl_zones = self._find_rapl_zones()
        self.available = len(self.rapl_zones) > 0
        
    def _find_rapl_zones(self) -> Dict[str, Path]:
        """Find available RAPL power zones"""
        zones = {}
        
        if not self.rapl_base.exists():
            return zones
            
        # Look for intel-rapl or amd-rapl subdirectories
        for rapl_dir in self.rapl_base.glob("*-rapl*"):
            # Find package zones
            for zone_dir in rapl_dir.glob("*-rapl:*"):
                name_file = zone_dir / "name"
                if name_file.exists():
                    zone_name = name_file.read_text().strip()
                    zones[zone_name] = zone_dir
                    
        return zones
    
    def _read_energy_uj(self, zone_path: Path) -> Optional[int]:
        """Read energy counter in microjoules"""
        energy_file = zone_path / "energy_uj"
        if energy_file.exists():
            try:
                return int(energy_file.read_text().strip())
            except (ValueError, IOError):
                return None
        return None
    
    def read_power(self, duration_seconds: float = 1.0) -> Optional[PowerReading]:
        """
        Measure power consumption over a short duration
        
        Args:
            duration_seconds: How long to measure (default 1 second)
            
        Returns:
            PowerReading with watts for each component
        """
        if not self.available:
            return None
            
        # Take first measurement
        start_time = time.time()
        start_readings = {}
        for name, zone in self.rapl_zones.items():
            energy = self._read_energy_uj(zone)
            if energy is not None:
                start_readings[name] = energy
        
        # Wait for measurement duration
        time.sleep(duration_seconds)
        
        # Take second measurement
        end_time = time.time()
        end_readings = {}
        for name, zone in self.rapl_zones.items():
            energy = self._read_energy_uj(zone)
            if energy is not None:
                end_readings[name] = energy
        
        # Calculate power (energy difference / time)
        actual_duration = end_time - start_time
        power_readings = {}
        
        for name in start_readings:
            if name in end_readings:
                # Energy in microjoules, convert to watts
                energy_diff_uj = end_readings[name] - start_readings[name]
                power_watts = (energy_diff_uj / 1_000_000) / actual_duration
                power_readings[name] = power_watts
        
        # Extract specific components (names vary by system)
        package_power = power_readings.get('package-0', 0) or power_readings.get('psys', 0)
        cores_power = power_readings.get('core', 0)
        uncore_power = power_readings.get('uncore', 0)
        dram_power = power_readings.get('dram', 0)
        
        total_power = sum(power_readings.values())
        
        return PowerReading(
            timestamp=end_time,
            package_watts=package_power,
            cores_watts=cores_power,
            uncore_watts=uncore_power,
            dram_watts=dram_power,
            total_watts=total_power
        )
    
    def get_info(self) -> Dict[str, any]:
        """Get information about available RAPL zones"""
        return {
            "available": self.available,
            "zones": list(self.rapl_zones.keys()),
            "rapl_base": str(self.rapl_base)
        }


class ProcessPowerMonitor:
    """Monitor power consumption of specific processes (requires root/sudo)"""
    
    def __init__(self):
        self.rapl = RAPLMonitor()
        
    def measure_process_power(self, pid: int, duration_seconds: float = 1.0) -> Optional[float]:
        """
        Estimate power consumption of a specific process
        This is approximate - measures total system power and scales by CPU usage
        
        Returns: Estimated watts for the process
        """
        if not self.rapl.available:
            return None
            
        # Get CPU usage before measurement
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                stat_before = f.read().split()
                utime_before = int(stat_before[13])
                stime_before = int(stat_before[14])
        except (FileNotFoundError, IndexError):
            return None
        
        # Measure total system power
        power_reading = self.rapl.read_power(duration_seconds)
        if not power_reading:
            return None
        
        # Get CPU usage after measurement
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                stat_after = f.read().split()
                utime_after = int(stat_after[13])
                stime_after = int(stat_after[14])
        except (FileNotFoundError, IndexError):
            return None
        
        # Calculate CPU time used by process (in clock ticks)
        process_cpu_ticks = (utime_after - utime_before) + (stime_after - stime_before)
        
        # Get system clock ticks per second
        clock_ticks_per_sec = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
        
        # Calculate process CPU percentage (approximation)
        # This is a rough estimate - actual process power would need more sophisticated tracking
        cpu_cores = os.cpu_count() or 1
        total_possible_ticks = duration_seconds * clock_ticks_per_sec * cpu_cores
        
        if total_possible_ticks > 0:
            process_cpu_fraction = process_cpu_ticks / total_possible_ticks
            estimated_power = power_reading.total_watts * process_cpu_fraction
            return estimated_power
        
        return None


# Global monitor instance
power_monitor = RAPLMonitor()
process_monitor = ProcessPowerMonitor()


if __name__ == "__main__":
    # Test the monitor
    print("=== Real-time Power Monitor Test ===\n")
    
    info = power_monitor.get_info()
    print(f"RAPL Available: {info['available']}")
    print(f"RAPL Zones: {info['zones']}\n")
    
    if info['available']:
        print("Measuring power consumption for 2 seconds...")
        reading = power_monitor.read_power(duration_seconds=2.0)
        
        if reading:
            print(f"\nPower Reading:")
            print(f"  Package:  {reading.package_watts:.2f} W")
            print(f"  Cores:    {reading.cores_watts:.2f} W")
            print(f"  Uncore:   {reading.uncore_watts:.2f} W")
            print(f"  DRAM:     {reading.dram_watts:.2f} W")
            print(f"  TOTAL:    {reading.total_watts:.2f} W")
            
            # Calculate Wh per 1000 tokens (assume 100 tokens/sec inference)
            tokens_per_second = 100
            total_tokens = tokens_per_second * 2.0
            wh_consumed = (reading.total_watts * 2.0) / 3600  # Convert to Wh
            wh_per_1000_tokens = (wh_consumed / total_tokens) * 1000
            
            print(f"\n  Estimated at 100 tokens/sec:")
            print(f"  {wh_per_1000_tokens:.4f} Wh per 1000 tokens")
    else:
        print("RAPL not available on this system.")
        print("You may need to:")
        print("  1. Run with sudo/root permissions")
        print("  2. Enable RAPL in BIOS/kernel")
        print("  3. Check if your CPU supports RAPL")
