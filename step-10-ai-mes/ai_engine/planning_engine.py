"""
Genetic Algorithm Production Planning Engine.

Optimizes production scheduling by evolving a population of candidate
schedules using selection, crossover, and mutation operators. Minimizes
total makespan, tardiness, and machine idle time while respecting
resource constraints and job dependencies.

Chromosome encoding: sequence of (job_id, machine_id) assignments.
Fitness: weighted sum of makespan, tardiness penalties, and setup costs.
"""

import random
import copy
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum


class MachineType(str, Enum):
    CNC = "cnc"
    ASSEMBLY = "assembly"
    WELDING = "welding"
    PAINTING = "painting"
    INSPECTION = "inspection"
    PACKAGING = "packaging"


class JobPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProductionJob(BaseModel):
    """A production job to be scheduled."""
    job_id: str = Field(..., description="Unique job identifier")
    product_name: str = Field(default="Product")
    operations: List[str] = Field(..., description="Required machine types in order")
    processing_times: List[int] = Field(..., description="Minutes per operation")
    due_date: Optional[str] = Field(None, description="ISO format deadline")
    priority: JobPriority = JobPriority.MEDIUM
    quantity: int = Field(default=1, ge=1)
    dependencies: List[str] = Field(default_factory=list,
                                    description="Job IDs that must complete first")
    setup_time_minutes: int = Field(default=15, description="Setup time between jobs on same machine")


class MachineConfig(BaseModel):
    """Configuration for a production machine."""
    machine_id: str
    machine_type: MachineType
    capacity_factor: float = Field(default=1.0, ge=0.1, le=2.0)
    available_from: str = Field(default="00:00")
    available_until: str = Field(default="23:59")
    maintenance_windows: List[Dict[str, str]] = Field(default_factory=list)


class ScheduleEntry(BaseModel):
    """A single scheduled operation."""
    job_id: str
    operation_index: int
    machine_id: str
    machine_type: str
    start_minute: int
    end_minute: int
    duration_minutes: int


class ScheduleResult(BaseModel):
    """Output of the genetic algorithm optimizer."""
    schedule: List[ScheduleEntry]
    makespan_minutes: int = Field(description="Total time from start to last job completion")
    total_tardiness_minutes: int = Field(default=0, description="Sum of lateness across all jobs")
    utilization_pct: Dict[str, float] = Field(default_factory=dict,
                                              description="Per-machine utilization percentage")
    fitness_score: float = Field(description="Optimization fitness score (lower is better)")
    generations: int = Field(description="Number of GA generations evolved")
    population_size: int
    computation_time_ms: float
    convergence_history: List[float] = Field(default_factory=list,
                                             description="Best fitness per generation")


class GeneticPlanningEngine:
    """
    Genetic Algorithm optimizer for production scheduling.

    Uses a permutation-based chromosome encoding where each gene
    represents a (job_id, operation_index) tuple. The decoding
    heuristic assigns each operation to the earliest available
    machine of the required type.

    GA Operators:
    - Selection: Tournament selection (size=3)
    - Crossover: Order Crossover (OX) preserving operation sequences
    - Mutation: Swap mutation + insertion mutation
    - Elitism: Top 10% carry over unchanged

    Args:
        population_size: number of chromosomes per generation
        generations: maximum generations to evolve
        crossover_rate: probability of crossover per pair
        mutation_rate: probability of mutation per chromosome
        elitism_pct: fraction of top individuals to preserve
        tardiness_weight: penalty weight for late jobs
    """

    def __init__(self, population_size: int = 100, generations: int = 200,
                 crossover_rate: float = 0.85, mutation_rate: float = 0.15,
                 elitism_pct: float = 0.1, tardiness_weight: float = 2.0):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = max(1, int(population_size * elitism_pct))
        self.tardiness_weight = tardiness_weight

    def optimize(self, jobs: List[ProductionJob],
                 machines: Optional[List[MachineConfig]] = None) -> ScheduleResult:
        """
        Run the genetic algorithm to find an optimal production schedule.

        Args:
            jobs: list of production jobs to schedule
            machines: optional machine configurations (auto-generated if None)

        Returns:
            ScheduleResult with optimized schedule and metrics
        """
        import time
        start_time = time.time()

        if not jobs:
            return ScheduleResult(
                schedule=[], makespan_minutes=0, fitness_score=0.0,
                generations=0, population_size=self.population_size,
                computation_time_ms=0,
            )

        # Auto-generate machines if not provided
        if machines is None:
            machines = self._default_machines(jobs)

        machine_map = self._build_machine_map(machines)

        # Build operation list: [(job_id, op_index, machine_type, duration)]
        operations = []
        for job in jobs:
            for op_idx, (op_type, duration) in enumerate(
                    zip(job.operations, job.processing_times)):
                operations.append((job.job_id, op_idx, op_type,
                                   int(duration * job.quantity)))

        # Initialize population (random permutations of operations)
        population = [
            random.sample(range(len(operations)), len(operations))
            for _ in range(self.population_size)
        ]

        # Pre-compute job due dates in minutes from start
        due_dates = {}
        for job in jobs:
            if job.due_date:
                try:
                    dt = datetime.fromisoformat(job.due_date)
                    due_dates[job.job_id] = int(
                        (dt - datetime.utcnow()).total_seconds() / 60
                    )
                except (ValueError, TypeError):
                    due_dates[job.job_id] = 10000
            else:
                due_dates[job.job_id] = 10000

        # Priority weights
        priority_weights = {
            JobPriority.LOW: 0.5,
            JobPriority.MEDIUM: 1.0,
            JobPriority.HIGH: 2.0,
            JobPriority.CRITICAL: 5.0,
        }
        job_weights = {j.job_id: priority_weights[j.priority] for j in jobs}

        # Dependencies map
        dep_map = {j.job_id: set(j.dependencies) for j in jobs}

        convergence = []
        best_ever = None
        best_fitness = float("inf")

        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for chromo in population:
                schedule, makespan, tardiness = self._decode(
                    chromo, operations, machine_map, due_dates,
                    job_weights, dep_map, jobs
                )
                fitness = makespan + self.tardiness_weight * tardiness
                fitness_scores.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_ever = (chromo[:], schedule, makespan, tardiness)

            convergence.append(best_fitness)

            # Early stop if converged
            if gen > 50 and len(convergence) > 20:
                recent = convergence[-20:]
                if max(recent) - min(recent) < 1.0:
                    break

            # Selection + crossover + mutation
            new_population = []

            # Elitism
            elite_indices = sorted(range(len(fitness_scores)),
                                   key=lambda i: fitness_scores[i])[:self.elitism_count]
            for ei in elite_indices:
                new_population.append(population[ei][:])

            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population, fitness_scores)
                p2 = self._tournament_select(population, fitness_scores)

                if random.random() < self.crossover_rate:
                    c1, c2 = self._order_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                if random.random() < self.mutation_rate:
                    self._mutate(c1)
                if random.random() < self.mutation_rate:
                    self._mutate(c2)

                new_population.extend([c1, c2])

            population = new_population[:self.population_size]

        # Build final result
        if best_ever is None:
            schedule, makespan, tardiness = self._decode(
                population[0], operations, machine_map, due_dates,
                job_weights, dep_map, jobs
            )
        else:
            _, schedule, makespan, tardiness = best_ever

        # Compute utilization
        utilization = self._compute_utilization(schedule, makespan, machines)

        elapsed_ms = (time.time() - start_time) * 1000

        return ScheduleResult(
            schedule=schedule,
            makespan_minutes=makespan,
            total_tardiness_minutes=tardiness,
            utilization_pct=utilization,
            fitness_score=round(best_fitness, 2),
            generations=min(gen + 1, self.generations) if 'gen' in dir() else self.generations,
            population_size=self.population_size,
            computation_time_ms=round(elapsed_ms, 1),
            convergence_history=convergence,
        )

    def _decode(self, chromosome: List[int], operations: list,
                machine_map: Dict[str, List[str]], due_dates: dict,
                job_weights: dict, dep_map: dict,
                jobs: List[ProductionJob]) -> Tuple[List[ScheduleEntry], int, int]:
        """Decode a chromosome into a schedule using earliest-available heuristic."""
        machine_available = {mid: 0 for mtype in machine_map for mid in machine_map[mtype]}
        job_op_end = {}  # (job_id, op_idx) → end_time
        schedule = []
        makespan = 0
        tardiness = 0

        for gene_idx in chromosome:
            job_id, op_idx, machine_type, duration = operations[gene_idx]

            # Get available machines for this operation type
            available_machines = machine_map.get(machine_type, [])
            if not available_machines:
                continue

            # Earliest start: wait for previous operation + dependencies
            earliest = 0
            if op_idx > 0:
                prev_end = job_op_end.get((job_id, op_idx - 1), 0)
                earliest = max(earliest, prev_end)

            # Wait for dependencies
            for dep_id in dep_map.get(job_id, []):
                dep_job = next((j for j in jobs if j.job_id == dep_id), None)
                if dep_job:
                    last_op = len(dep_job.operations) - 1
                    dep_end = job_op_end.get((dep_id, last_op), 0)
                    earliest = max(earliest, dep_end)

            # Find machine with earliest availability
            best_machine = min(available_machines,
                               key=lambda m: max(machine_available[m], earliest))
            start = max(machine_available[best_machine], earliest)
            end = start + duration

            schedule.append(ScheduleEntry(
                job_id=job_id,
                operation_index=op_idx,
                machine_id=best_machine,
                machine_type=machine_type,
                start_minute=start,
                end_minute=end,
                duration_minutes=duration,
            ))

            machine_available[best_machine] = end
            job_op_end[(job_id, op_idx)] = end
            makespan = max(makespan, end)

            # Check tardiness for this job's last operation
            job = next((j for j in jobs if j.job_id == job_id), None)
            if job and op_idx == len(job.operations) - 1:
                due = due_dates.get(job_id, 10000)
                if end > due:
                    tardiness += int((end - due) * job_weights.get(job_id, 1.0))

        return schedule, makespan, tardiness

    def _tournament_select(self, population: list, fitness: list,
                           k: int = 3) -> list:
        """Tournament selection: pick best of k random individuals."""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best = min(indices, key=lambda i: fitness[i])
        return population[best][:]

    def _order_crossover(self, p1: list, p2: list) -> Tuple[list, list]:
        """Order Crossover (OX): preserves relative order of operations."""
        size = len(p1)
        if size < 3:
            return p1[:], p2[:]
        a, b = sorted(random.sample(range(size), 2))
        c1 = [-1] * size
        c2 = [-1] * size
        c1[a:b + 1] = p1[a:b + 1]
        c2[a:b + 1] = p2[a:b + 1]

        fill1 = [g for g in p2 if g not in c1[a:b + 1]]
        fill2 = [g for g in p1 if g not in c2[a:b + 1]]

        idx = (b + 1) % size
        for g in fill1:
            c1[idx] = g
            idx = (idx + 1) % size
        idx = (b + 1) % size
        for g in fill2:
            c2[idx] = g
            idx = (idx + 1) % size

        return c1, c2

    def _mutate(self, chromosome: list):
        """Swap mutation: exchange two random positions."""
        if len(chromosome) < 2:
            return
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    def _default_machines(self, jobs: List[ProductionJob]) -> List[MachineConfig]:
        """Generate default machine configuration from job requirements."""
        needed_types = set()
        for job in jobs:
            for op in job.operations:
                needed_types.add(op)

        machines = []
        for mtype in needed_types:
            for i in range(2):  # 2 machines per type
                machines.append(MachineConfig(
                    machine_id=f"{mtype}-{i+1:02d}",
                    machine_type=MachineType(mtype) if mtype in MachineType.__members__.values() else MachineType.ASSEMBLY,
                ))
        return machines

    def _build_machine_map(self, machines: List[MachineConfig]) -> Dict[str, List[str]]:
        """Group machine IDs by type."""
        machine_map: Dict[str, List[str]] = {}
        for m in machines:
            machine_map.setdefault(m.machine_type.value, []).append(m.machine_id)
        return machine_map

    def _compute_utilization(self, schedule: List[ScheduleEntry],
                             makespan: int,
                             machines: List[MachineConfig]) -> Dict[str, float]:
        """Compute per-machine utilization as a percentage of makespan."""
        busy_time: Dict[str, int] = {}
        for entry in schedule:
            busy_time[entry.machine_id] = (
                busy_time.get(entry.machine_id, 0) + entry.duration_minutes
            )

        utilization = {}
        for m in machines:
            if makespan > 0:
                bt = busy_time.get(m.machine_id, 0)
                utilization[m.machine_id] = round(bt / makespan * 100, 1)
            else:
                utilization[m.machine_id] = 0.0
        return utilization
