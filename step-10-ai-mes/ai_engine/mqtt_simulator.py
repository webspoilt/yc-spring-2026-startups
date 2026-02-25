"""
MQTT IoT Sensor Simulator for Manufacturing Floor.

Simulates realistic IoT sensor data from a manufacturing environment:
- Temperature sensors (ambient + machine-level)
- Vibration sensors on rotating equipment
- Pressure gauges on pneumatic systems
- Power meters on production machines
- Humidity sensors for quality-sensitive areas

Uses MQTT protocol (paho-mqtt) to publish to a broker, with
configurable noise, drift, and anomaly injection for testing.
"""

import json
import math
import random
import time
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum


class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    POWER = "power"
    HUMIDITY = "humidity"
    SPEED_RPM = "speed_rpm"
    FLOW_RATE = "flow_rate"


class SensorReading(BaseModel):
    """A single IoT sensor reading."""
    sensor_id: str
    sensor_type: SensorType
    machine_id: str
    value: float
    unit: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    is_anomaly: bool = False
    quality_flag: str = Field(default="normal",
                              description="normal | warning | critical")


class SensorConfig(BaseModel):
    """Configuration for a simulated sensor."""
    sensor_id: str
    sensor_type: SensorType
    machine_id: str
    unit: str
    base_value: float
    noise_std: float = Field(default=0.5, description="Gaussian noise σ")
    drift_rate: float = Field(default=0.001, description="Linear drift per reading")
    anomaly_probability: float = Field(default=0.02, description="Probability of anomaly")
    anomaly_magnitude: float = Field(default=5.0, description="Multiplier for anomaly noise")
    min_value: float = Field(default=0.0)
    max_value: float = Field(default=1000.0)
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None


class MQTTIoTSimulator:
    """
    Simulates a fleet of IoT sensors publishing to an MQTT broker.

    Features:
    - Configurable sensor fleet with per-sensor noise/drift/anomaly profiles
    - Sinusoidal base patterns simulating real machine behavior
    - Anomaly injection for testing alerting systems
    - MQTT publishing to configurable topics
    - Thread-safe for background operation

    Usage:
        sim = MQTTIoTSimulator(broker_host="localhost")
        sim.add_default_sensors()
        sim.start(interval_seconds=1.0)
        # ... later
        sim.stop()
    """

    DEFAULT_SENSORS = [
        SensorConfig(
            sensor_id="TEMP-CNC-01", sensor_type=SensorType.TEMPERATURE,
            machine_id="cnc-01", unit="°C", base_value=45.0, noise_std=1.5,
            warning_high=60.0, critical_high=75.0, warning_low=10.0,
        ),
        SensorConfig(
            sensor_id="TEMP-CNC-02", sensor_type=SensorType.TEMPERATURE,
            machine_id="cnc-02", unit="°C", base_value=42.0, noise_std=1.2,
            warning_high=60.0, critical_high=75.0,
        ),
        SensorConfig(
            sensor_id="VIB-CNC-01", sensor_type=SensorType.VIBRATION,
            machine_id="cnc-01", unit="mm/s", base_value=2.5, noise_std=0.3,
            warning_high=4.5, critical_high=7.0,
        ),
        SensorConfig(
            sensor_id="VIB-WELD-01", sensor_type=SensorType.VIBRATION,
            machine_id="welding-01", unit="mm/s", base_value=1.8, noise_std=0.2,
            warning_high=3.5, critical_high=5.5,
        ),
        SensorConfig(
            sensor_id="PRESS-PNEUM-01", sensor_type=SensorType.PRESSURE,
            machine_id="assembly-01", unit="bar", base_value=6.0, noise_std=0.15,
            warning_low=4.5, critical_low=3.0, warning_high=8.0, critical_high=9.5,
        ),
        SensorConfig(
            sensor_id="PWR-CNC-01", sensor_type=SensorType.POWER,
            machine_id="cnc-01", unit="kW", base_value=15.0, noise_std=2.0,
            warning_high=22.0, critical_high=28.0,
        ),
        SensorConfig(
            sensor_id="HUMID-PAINT-01", sensor_type=SensorType.HUMIDITY,
            machine_id="painting-01", unit="%RH", base_value=45.0, noise_std=3.0,
            warning_low=30.0, critical_low=20.0, warning_high=70.0, critical_high=80.0,
        ),
        SensorConfig(
            sensor_id="RPM-CNC-01", sensor_type=SensorType.SPEED_RPM,
            machine_id="cnc-01", unit="RPM", base_value=3000.0, noise_std=50.0,
            warning_low=2000.0, critical_low=1000.0, warning_high=4500.0, critical_high=5000.0,
        ),
    ]

    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
                 topic_prefix: str = "factory/sensors",
                 client_id: str = "mes-simulator"):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_prefix = topic_prefix
        self.client_id = client_id
        self._mqtt_client = None
        self._sensors: List[SensorConfig] = []
        self._reading_count = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._readings_buffer: List[SensorReading] = []
        self._callbacks: List[Callable[[SensorReading], None]] = []
        self._max_buffer = 1000

    def add_sensor(self, config: SensorConfig):
        """Add a sensor to the simulation."""
        self._sensors.append(config)

    def add_default_sensors(self):
        """Add the default set of manufacturing sensors."""
        for sensor in self.DEFAULT_SENSORS:
            self.add_sensor(sensor)

    def on_reading(self, callback: Callable[[SensorReading], None]):
        """Register a callback for new sensor readings."""
        self._callbacks.append(callback)

    def generate_reading(self, sensor: SensorConfig) -> SensorReading:
        """
        Generate a single sensor reading with realistic patterns.

        Combines:
        - Base value + linear drift
        - Sinusoidal pattern (simulates machine cycles)
        - Gaussian noise
        - Random anomaly injection
        """
        self._reading_count += 1
        t = self._reading_count

        # Sinusoidal pattern (machine cycle)
        cycle = math.sin(2 * math.pi * t / 120) * sensor.noise_std * 0.5

        # Linear drift
        drift = sensor.drift_rate * t

        # Gaussian noise
        noise = random.gauss(0, sensor.noise_std)

        # Anomaly injection
        is_anomaly = random.random() < sensor.anomaly_probability
        if is_anomaly:
            anomaly_noise = random.gauss(0, sensor.noise_std * sensor.anomaly_magnitude)
            noise += anomaly_noise

        value = sensor.base_value + cycle + drift + noise
        value = max(sensor.min_value, min(sensor.max_value, value))

        # Quality flag assessment
        quality = "normal"
        if sensor.critical_high and value >= sensor.critical_high:
            quality = "critical"
        elif sensor.critical_low and value <= sensor.critical_low:
            quality = "critical"
        elif sensor.warning_high and value >= sensor.warning_high:
            quality = "warning"
        elif sensor.warning_low and value <= sensor.warning_low:
            quality = "warning"

        return SensorReading(
            sensor_id=sensor.sensor_id,
            sensor_type=sensor.sensor_type,
            machine_id=sensor.machine_id,
            value=round(value, 3),
            unit=sensor.unit,
            is_anomaly=is_anomaly,
            quality_flag=quality,
        )

    def generate_batch(self) -> List[SensorReading]:
        """Generate one reading from every configured sensor."""
        readings = [self.generate_reading(s) for s in self._sensors]

        # Buffer readings
        self._readings_buffer.extend(readings)
        if len(self._readings_buffer) > self._max_buffer:
            self._readings_buffer = self._readings_buffer[-self._max_buffer:]

        # Fire callbacks
        for reading in readings:
            for cb in self._callbacks:
                try:
                    cb(reading)
                except Exception:
                    pass

        return readings

    def publish_batch(self, readings: List[SensorReading]):
        """Publish readings to MQTT broker."""
        if self._mqtt_client is None:
            self._connect_mqtt()

        if self._mqtt_client is None:
            return  # MQTT not available

        for reading in readings:
            topic = f"{self.topic_prefix}/{reading.machine_id}/{reading.sensor_type.value}"
            payload = reading.model_dump_json()
            try:
                self._mqtt_client.publish(topic, payload, qos=1)
            except Exception:
                pass

    def start(self, interval_seconds: float = 1.0):
        """Start the simulator in a background thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(interval_seconds,),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the simulator."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._mqtt_client:
            try:
                self._mqtt_client.disconnect()
            except Exception:
                pass

    def get_recent_readings(self, limit: int = 100) -> List[SensorReading]:
        """Get recent readings from the buffer."""
        return self._readings_buffer[-limit:]

    def get_sensor_stats(self) -> Dict[str, Any]:
        """Get per-sensor statistics from buffered readings."""
        stats: Dict[str, Dict] = {}
        for r in self._readings_buffer:
            if r.sensor_id not in stats:
                stats[r.sensor_id] = {
                    "sensor_id": r.sensor_id,
                    "type": r.sensor_type.value,
                    "machine_id": r.machine_id,
                    "unit": r.unit,
                    "values": [],
                    "anomaly_count": 0,
                    "warning_count": 0,
                    "critical_count": 0,
                }
            stats[r.sensor_id]["values"].append(r.value)
            if r.is_anomaly:
                stats[r.sensor_id]["anomaly_count"] += 1
            if r.quality_flag == "warning":
                stats[r.sensor_id]["warning_count"] += 1
            elif r.quality_flag == "critical":
                stats[r.sensor_id]["critical_count"] += 1

        result = {}
        for sid, s in stats.items():
            vals = s["values"]
            result[sid] = {
                "sensor_id": s["sensor_id"],
                "type": s["type"],
                "machine_id": s["machine_id"],
                "unit": s["unit"],
                "reading_count": len(vals),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "mean": round(sum(vals) / len(vals), 3),
                "latest": round(vals[-1], 3),
                "anomaly_count": s["anomaly_count"],
                "warning_count": s["warning_count"],
                "critical_count": s["critical_count"],
            }
        return result

    def _run_loop(self, interval: float):
        """Background loop for continuous simulation."""
        while self._running:
            readings = self.generate_batch()
            try:
                self.publish_batch(readings)
            except Exception:
                pass
            time.sleep(interval)

    def _connect_mqtt(self):
        """Connect to MQTT broker."""
        try:
            import paho.mqtt.client as mqtt
            client = mqtt.Client(client_id=self.client_id)
            client.connect(self.broker_host, self.broker_port, 60)
            client.loop_start()
            self._mqtt_client = client
        except Exception:
            self._mqtt_client = None
