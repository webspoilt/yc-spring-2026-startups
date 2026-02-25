"""
GPU Manager — nvidia-smi Parser and GPU Fleet Monitoring.

Parses nvidia-smi XML/CSV output to track GPU utilization, memory,
temperature, power, and process allocation across a cluster.
Provides real-time fleet status, over-utilization alerts, and
optimal GPU selection for job scheduling.
"""

import subprocess
import xml.etree.ElementTree as ET
import re
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class GPUHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNREACHABLE = "unreachable"


class GPUProcess(BaseModel):
    """Process running on a GPU."""
    pid: int
    name: str = ""
    used_memory_mib: int = 0
    gpu_instance_id: Optional[str] = None


class GPUDevice(BaseModel):
    """Parsed state of a single GPU device."""
    index: int
    name: str = Field(default="Unknown GPU")
    uuid: str = ""
    temperature_celsius: int = 0
    gpu_utilization_pct: float = 0.0
    memory_used_mib: int = 0
    memory_total_mib: int = 0
    memory_free_mib: int = 0
    memory_utilization_pct: float = 0.0
    power_draw_watts: float = 0.0
    power_limit_watts: float = 0.0
    fan_speed_pct: int = 0
    driver_version: str = ""
    cuda_version: str = ""
    compute_mode: str = "Default"
    pci_bus_id: str = ""
    processes: List[GPUProcess] = Field(default_factory=list)
    health: GPUHealthStatus = GPUHealthStatus.HEALTHY
    alerts: List[str] = Field(default_factory=list)

    @property
    def is_available(self) -> bool:
        """GPU has >20% memory free and utilization <90%."""
        return (self.memory_utilization_pct < 80 and
                self.gpu_utilization_pct < 90 and
                self.health != GPUHealthStatus.CRITICAL)


class GPUFleetStatus(BaseModel):
    """Aggregate status of all GPUs in the fleet."""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    total_gpus: int = 0
    healthy_gpus: int = 0
    available_gpus: int = 0
    total_memory_mib: int = 0
    used_memory_mib: int = 0
    avg_utilization_pct: float = 0.0
    avg_temperature_celsius: float = 0.0
    total_power_watts: float = 0.0
    devices: List[GPUDevice] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)


class GPUManager:
    """
    Monitors and manages GPU fleet via nvidia-smi parsing.

    Features:
    - Parse nvidia-smi XML output for detailed GPU telemetry
    - Parse CSV output for lightweight monitoring
    - Health classification with configurable thresholds
    - Optimal GPU selection for job scheduling
    - Fleet-level aggregation and alerting
    """

    # Alerting thresholds
    TEMP_WARNING = 80      # Celsius
    TEMP_CRITICAL = 90
    MEM_WARNING = 85       # Percent
    MEM_CRITICAL = 95
    UTIL_WARNING = 95      # Percent
    POWER_WARNING = 0.9    # Fraction of limit

    def __init__(self, nvidia_smi_path: str = "nvidia-smi"):
        self.nvidia_smi_path = nvidia_smi_path

    def _run_nvidia_smi(self, args: List[str]) -> str:
        """Execute nvidia-smi with given arguments."""
        try:
            cmd = [self.nvidia_smi_path] + args
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
            return result.stdout
        except FileNotFoundError:
            raise RuntimeError(
                f"nvidia-smi not found at '{self.nvidia_smi_path}'. "
                "Ensure NVIDIA drivers are installed."
            )

    def parse_xml(self, xml_str: Optional[str] = None) -> List[GPUDevice]:
        """
        Parse nvidia-smi XML output into structured GPUDevice objects.

        Args:
            xml_str: Raw XML string. If None, queries nvidia-smi live.

        Returns:
            List of GPUDevice with full telemetry.
        """
        if xml_str is None:
            xml_str = self._run_nvidia_smi(["-q", "-x"])

        root = ET.fromstring(xml_str)
        devices = []

        driver_version = self._text(root, "driver_version", "")
        cuda_version = self._text(root, "cuda_version", "")

        for idx, gpu in enumerate(root.findall("gpu")):
            # Basic info
            name = self._text(gpu, "product_name", f"GPU {idx}")
            uuid = self._text(gpu, "uuid", "")
            pci_bus = self._text(gpu.find("pci"), "pci_bus_id", "") if gpu.find("pci") is not None else ""

            # Temperature
            temp_node = gpu.find("temperature")
            temp = self._parse_int(self._text(temp_node, "gpu_temp", "0"))

            # Utilization
            util_node = gpu.find("utilization")
            gpu_util = self._parse_float(self._text(util_node, "gpu_util", "0"))
            mem_util = self._parse_float(self._text(util_node, "memory_util", "0"))

            # Memory
            mem_node = gpu.find("fb_memory_usage")
            mem_total = self._parse_int(self._text(mem_node, "total", "0"))
            mem_used = self._parse_int(self._text(mem_node, "used", "0"))
            mem_free = self._parse_int(self._text(mem_node, "free", "0"))

            # Power
            power_node = gpu.find("gpu_power_readings") or gpu.find("power_readings")
            power_draw = self._parse_float(
                self._text(power_node, "power_draw", "0") if power_node is not None else "0"
            )
            power_limit = self._parse_float(
                self._text(power_node, "power_limit", "0") if power_node is not None else "0"
            )

            # Fan
            fan_speed = self._parse_int(self._text(gpu, "fan_speed", "0"))

            # Compute mode
            compute_mode = self._text(gpu, "compute_mode", "Default")

            # Processes
            processes = []
            proc_info = gpu.find("processes")
            if proc_info is not None:
                for proc in proc_info.findall("process_info"):
                    pid = self._parse_int(self._text(proc, "pid", "0"))
                    proc_name = self._text(proc, "process_name", "")
                    proc_mem = self._parse_int(self._text(proc, "used_memory", "0"))
                    processes.append(GPUProcess(
                        pid=pid, name=proc_name, used_memory_mib=proc_mem
                    ))

            # Memory utilization percentage
            mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0

            # Health assessment
            health, alerts = self._assess_health(
                temp, gpu_util, mem_pct, power_draw, power_limit
            )

            device = GPUDevice(
                index=idx,
                name=name,
                uuid=uuid,
                temperature_celsius=temp,
                gpu_utilization_pct=gpu_util,
                memory_used_mib=mem_used,
                memory_total_mib=mem_total,
                memory_free_mib=mem_free,
                memory_utilization_pct=round(mem_pct, 1),
                power_draw_watts=power_draw,
                power_limit_watts=power_limit,
                fan_speed_pct=fan_speed,
                driver_version=driver_version,
                cuda_version=cuda_version,
                compute_mode=compute_mode,
                pci_bus_id=pci_bus,
                processes=processes,
                health=health,
                alerts=alerts,
            )
            devices.append(device)

        return devices

    def parse_csv(self, csv_str: Optional[str] = None) -> List[GPUDevice]:
        """
        Parse nvidia-smi CSV output (lightweight monitoring).

        Args:
            csv_str: Raw CSV string. If None, queries nvidia-smi live.
        """
        if csv_str is None:
            csv_str = self._run_nvidia_smi([
                "--query-gpu=index,name,uuid,temperature.gpu,utilization.gpu,"
                "utilization.memory,memory.used,memory.total,memory.free,"
                "power.draw,power.limit,fan.speed",
                "--format=csv,noheader,nounits"
            ])

        devices = []
        for line in csv_str.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 12:
                continue

            idx = int(parts[0])
            mem_total = int(parts[7]) if parts[7] != "[N/A]" else 0
            mem_used = int(parts[6]) if parts[6] != "[N/A]" else 0
            mem_free = int(parts[8]) if parts[8] != "[N/A]" else 0
            temp = int(parts[3]) if parts[3] != "[N/A]" else 0
            gpu_util = float(parts[4]) if parts[4] != "[N/A]" else 0
            power_draw = float(parts[9]) if parts[9] != "[N/A]" else 0
            power_limit = float(parts[10]) if parts[10] != "[N/A]" else 0
            mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0

            health, alerts = self._assess_health(
                temp, gpu_util, mem_pct, power_draw, power_limit
            )

            devices.append(GPUDevice(
                index=idx,
                name=parts[1],
                uuid=parts[2],
                temperature_celsius=temp,
                gpu_utilization_pct=gpu_util,
                memory_used_mib=mem_used,
                memory_total_mib=mem_total,
                memory_free_mib=mem_free,
                memory_utilization_pct=round(mem_pct, 1),
                power_draw_watts=power_draw,
                power_limit_watts=power_limit,
                fan_speed_pct=int(parts[11]) if parts[11] != "[N/A]" else 0,
                health=health,
                alerts=alerts,
            ))

        return devices

    def get_fleet_status(self, xml_str: Optional[str] = None) -> GPUFleetStatus:
        """
        Get aggregate fleet status from all GPUs.

        Args:
            xml_str: Optional pre-fetched XML. Queries live if None.
        """
        try:
            devices = self.parse_xml(xml_str)
        except Exception:
            try:
                devices = self.parse_csv()
            except Exception:
                return GPUFleetStatus(
                    alerts=["Unable to query nvidia-smi. No GPUs detected."]
                )

        total = len(devices)
        healthy = sum(1 for d in devices if d.health == GPUHealthStatus.HEALTHY)
        available = sum(1 for d in devices if d.is_available)
        total_mem = sum(d.memory_total_mib for d in devices)
        used_mem = sum(d.memory_used_mib for d in devices)
        avg_util = sum(d.gpu_utilization_pct for d in devices) / max(total, 1)
        avg_temp = sum(d.temperature_celsius for d in devices) / max(total, 1)
        total_power = sum(d.power_draw_watts for d in devices)

        fleet_alerts = []
        for d in devices:
            fleet_alerts.extend(
                f"GPU {d.index} ({d.name}): {a}" for a in d.alerts
            )

        if available == 0 and total > 0:
            fleet_alerts.insert(0, "CRITICAL: No GPUs available for scheduling")

        return GPUFleetStatus(
            total_gpus=total,
            healthy_gpus=healthy,
            available_gpus=available,
            total_memory_mib=total_mem,
            used_memory_mib=used_mem,
            avg_utilization_pct=round(avg_util, 1),
            avg_temperature_celsius=round(avg_temp, 1),
            total_power_watts=round(total_power, 1),
            devices=devices,
            alerts=fleet_alerts,
        )

    def select_best_gpu(self, required_memory_mib: int = 0,
                        xml_str: Optional[str] = None) -> Optional[GPUDevice]:
        """
        Select the best available GPU for a new job.

        Strategy: prefer GPU with most free memory that meets requirements,
        among healthy/available devices.
        """
        try:
            devices = self.parse_xml(xml_str)
        except Exception:
            devices = self.parse_csv()

        candidates = [
            d for d in devices
            if d.is_available and d.memory_free_mib >= required_memory_mib
        ]

        if not candidates:
            return None

        # Sort by free memory descending, then utilization ascending
        candidates.sort(key=lambda d: (-d.memory_free_mib, d.gpu_utilization_pct))
        return candidates[0]

    def _assess_health(self, temp: int, util: float, mem_pct: float,
                       power: float, power_limit: float) -> Tuple[GPUHealthStatus, List[str]]:
        """Classify GPU health and generate alerts."""
        alerts = []
        health = GPUHealthStatus.HEALTHY

        if temp >= self.TEMP_CRITICAL:
            alerts.append(f"CRITICAL: Temperature {temp}°C exceeds {self.TEMP_CRITICAL}°C")
            health = GPUHealthStatus.CRITICAL
        elif temp >= self.TEMP_WARNING:
            alerts.append(f"WARNING: Temperature {temp}°C exceeds {self.TEMP_WARNING}°C")
            health = GPUHealthStatus.WARNING

        if mem_pct >= self.MEM_CRITICAL:
            alerts.append(f"CRITICAL: Memory {mem_pct:.0f}% exceeds {self.MEM_CRITICAL}%")
            health = GPUHealthStatus.CRITICAL
        elif mem_pct >= self.MEM_WARNING:
            alerts.append(f"WARNING: Memory {mem_pct:.0f}% exceeds {self.MEM_WARNING}%")
            if health != GPUHealthStatus.CRITICAL:
                health = GPUHealthStatus.WARNING

        if util >= self.UTIL_WARNING:
            alerts.append(f"WARNING: Utilization {util:.0f}% at max capacity")
            if health == GPUHealthStatus.HEALTHY:
                health = GPUHealthStatus.WARNING

        if power_limit > 0 and power / power_limit >= self.POWER_WARNING:
            alerts.append(f"WARNING: Power {power:.0f}W near limit {power_limit:.0f}W")
            if health == GPUHealthStatus.HEALTHY:
                health = GPUHealthStatus.WARNING

        return health, alerts

    @staticmethod
    def _text(parent, tag, default=""):
        if parent is None:
            return default
        el = parent.find(tag) if isinstance(tag, str) else tag
        if el is not None and el.text:
            return el.text.strip()
        return default

    @staticmethod
    def _parse_int(val: str) -> int:
        nums = re.findall(r"[\d]+", val)
        return int(nums[0]) if nums else 0

    @staticmethod
    def _parse_float(val: str) -> float:
        nums = re.findall(r"[\d.]+", val)
        return float(nums[0]) if nums else 0.0

    def get_mock_fleet(self) -> GPUFleetStatus:
        """Generate a mock fleet status for testing without GPUs."""
        devices = [
            GPUDevice(
                index=0, name="NVIDIA A100-SXM4-80GB", uuid="GPU-mock-0001",
                temperature_celsius=45, gpu_utilization_pct=23.0,
                memory_used_mib=12400, memory_total_mib=81920, memory_free_mib=69520,
                memory_utilization_pct=15.1, power_draw_watts=67.0, power_limit_watts=400.0,
                fan_speed_pct=0, driver_version="535.129.03", cuda_version="12.2",
                processes=[GPUProcess(pid=12345, name="python", used_memory_mib=12400)],
                health=GPUHealthStatus.HEALTHY, alerts=[],
            ),
            GPUDevice(
                index=1, name="NVIDIA A100-SXM4-80GB", uuid="GPU-mock-0002",
                temperature_celsius=72, gpu_utilization_pct=89.0,
                memory_used_mib=71000, memory_total_mib=81920, memory_free_mib=10920,
                memory_utilization_pct=86.7, power_draw_watts=295.0, power_limit_watts=400.0,
                fan_speed_pct=0, driver_version="535.129.03", cuda_version="12.2",
                processes=[
                    GPUProcess(pid=23456, name="python", used_memory_mib=35000),
                    GPUProcess(pid=23457, name="python", used_memory_mib=36000),
                ],
                health=GPUHealthStatus.WARNING,
                alerts=["WARNING: Memory 86.7% exceeds 85%"],
            ),
        ]
        return GPUFleetStatus(
            total_gpus=2, healthy_gpus=1, available_gpus=1,
            total_memory_mib=163840, used_memory_mib=83400,
            avg_utilization_pct=56.0, avg_temperature_celsius=58.5,
            total_power_watts=362.0, devices=devices, alerts=devices[1].alerts,
        )
