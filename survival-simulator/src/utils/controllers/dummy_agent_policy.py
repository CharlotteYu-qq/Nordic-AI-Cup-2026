import math
import random

from src.utils.DTOs import ActionRequest


_AGENT_MEMORY = {}


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _get_memory(agent_id: int, rng: random.Random):
    memory = _AGENT_MEMORY.get(agent_id)
    if memory is None:
        memory = {
            "wander_bias": rng.uniform(-math.pi / 6, math.pi / 6),
            "last_spawn_step": -9999,
            "food_heading": None,
            "food_ttl": 0,
            "threat_heading": None,
            "threat_ttl": 0,
        }
        _AGENT_MEMORY[agent_id] = memory
    return memory

def action_decision(observation_response: dict, rng: random.Random):
    """
    Dummy action selection for the agent with group awareness
    
    Args:
        observation_response (dict): Observation response from the environment
        rng (random.Random): Random number generator

    Returns:
        ActionRequest: Action decision
    """
    agent_id = observation_response["agent_id"]
    observations = observation_response.get("observations", [])
    energy = float(observation_response.get("energy", 0.0))
    age = float(observation_response.get("age", 0.0))
    speed = float(observation_response.get("speed", 0.0))
    sprint_speed = float(observation_response.get("sprint_speed", speed))
    max_energy = float(observation_response.get("max_energy", 0.0))

    memory = _get_memory(agent_id, rng)
    memory["food_ttl"] = max(0, memory["food_ttl"] - 1)
    memory["threat_ttl"] = max(0, memory["threat_ttl"] - 1)

    predators = []
    fruits = []
    edges = []
    agents = []

    for obs in observations:
        obs_type = obs.get("type")
        if obs_type == "Predator":
            predators.append(obs)
        elif obs_type == "Fruit":
            fruits.append(obs)
        elif obs_type == "Edge":
            edges.append(obs)
        elif obs_type == "Agent":
            agents.append(obs)

    def nearest(items):
        if not items:
            return None
        return min(items, key=lambda item: item.get("distance", float("inf")))

    nearest_predator = nearest(predators)
    nearest_fruit = nearest(fruits)
    nearest_edge = nearest(edges)
    nearest_agent = nearest(agents)

    if nearest_fruit is not None:
        memory["food_heading"] = float(nearest_fruit.get("angle", 0.0))
        memory["food_ttl"] = 20

    if nearest_predator is not None:
        predator_angle = float(nearest_predator.get("angle", 0.0))
        memory["threat_heading"] = _wrap_angle(predator_angle + math.pi)
        memory["threat_ttl"] = 15

    safe_energy = max_energy * 0.82 if max_energy > 0 else 140.0
    can_sprint = energy > max_energy * 0.35 if max_energy > 0 else energy > 40.0
    low_energy = energy < (max_energy * 0.28 if max_energy > 0 else 35.0)

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    # Group awareness: check if there's danger nearby with other agents
    group_threat_factor = 1.0
    if nearest_predator is not None and nearest_agent is not None:
        predator_distance = float(nearest_predator.get("distance", 9999.0))
        agent_distance = float(nearest_agent.get("distance", 9999.0))
        # If both predator and ally are close, boost threat response (group defense)
        if predator_distance < 200 and agent_distance < 150:
            group_threat_factor = 1.3

    # 1) Immediate danger: run away from the closest predator.
    if nearest_predator is not None:
        predator_distance = float(nearest_predator.get("distance", 9999.0))
        predator_angle = float(nearest_predator.get("angle", 0.0))
        escape_direction = _wrap_angle(predator_angle + math.pi)

        turn_angle = _wrap_angle(escape_direction * 0.75)
        move_direction = escape_direction
        move_distance = sprint_speed if can_sprint and predator_distance < (150 * group_threat_factor) else speed

    # 2) Food nearby: move toward the closest fruit.
    elif nearest_fruit is not None:
        fruit_distance = float(nearest_fruit.get("distance", 9999.0))
        fruit_angle = float(nearest_fruit.get("angle", 0.0))

        turn_angle = _wrap_angle(fruit_angle * 0.55)
        move_direction = _wrap_angle(fruit_angle + memory["wander_bias"] * 0.08)

        if fruit_distance < 25:
            move_distance = min(speed, fruit_distance)
        elif fruit_distance < 110 and can_sprint and not low_energy:
            move_distance = min(sprint_speed, fruit_distance)
        else:
            move_distance = min(speed * 0.9, sprint_speed)

    # 3) If we recently saw food, keep committing to that heading.
    elif memory["food_ttl"] > 0 and memory["food_heading"] is not None:
        food_heading = float(memory["food_heading"])
        turn_angle = _wrap_angle(food_heading * 0.35)
        move_direction = _wrap_angle(food_heading + memory["wander_bias"] * 0.05)
        move_distance = sprint_speed * 0.55 if can_sprint and not low_energy else speed * 0.8

    # 4) Group cohesion: if nearby agent exists and no immediate threats, move toward ally
    elif nearest_agent is not None and nearest_predator is None:
        agent_distance = float(nearest_agent.get("distance", 9999.0))
        agent_angle = float(nearest_agent.get("angle", 0.0))
        
        # Move closer to ally if far enough, but maintain some distance
        if agent_distance > 80:
            turn_angle = _wrap_angle(agent_angle * 0.25)
            move_direction = _wrap_angle(agent_angle + memory["wander_bias"] * 0.05)
            move_distance = speed * 0.7
        else:
            # If too close, move away slightly to avoid collision
            move_direction = _wrap_angle(agent_angle + math.pi)
            move_distance = speed * 0.3

    # 5) Edge avoidance if no obvious target.
    elif nearest_edge is not None:
        edge_coords = nearest_edge.get("coords")
        if edge_coords and len(edge_coords) == 2:
            start, end = edge_coords
            edge_mid_x = (float(start[0]) + float(end[0])) / 2.0
            edge_mid_y = (float(start[1]) + float(end[1])) / 2.0
            edge_angle = math.atan2(-edge_mid_y, -edge_mid_x)
            turn_angle = _wrap_angle(edge_angle * 0.3)
            move_direction = _wrap_angle(edge_angle)
        move_distance = speed * 0.65

    # 6) Exploration: low-cost wandering.
    else:
        if memory["threat_ttl"] > 0 and memory["threat_heading"] is not None:
            threat_heading = float(memory["threat_heading"])
            turn_angle = _wrap_angle(threat_heading * 0.15)
            move_direction = _wrap_angle(threat_heading)
            move_distance = speed * 0.85
        else:
            turn_angle = memory["wander_bias"] * 0.12 + rng.uniform(-0.05, 0.05)
            move_direction = turn_angle
            move_distance = speed * 0.55

    # Keep movement within useful bounds.
    move_distance = max(0.0, min(move_distance, sprint_speed))

    # Spawn only when very safe, older, and not in a food-chasing phase.
    spawn_cooldown = 250
    if memory["last_spawn_step"] < 0:
        memory["last_spawn_step"] = -9999
    if (
        energy > safe_energy
        and age > 45
        and nearest_predator is None
        and nearest_fruit is None
        and memory["food_ttl"] == 0
        and (age * 10 - memory["last_spawn_step"]) > spawn_cooldown
    ):
        # Boost spawn chance slightly if there are nearby allies (group expansion)
        spawn_chance = 0.08
        if nearest_agent is not None and float(nearest_agent.get("distance", 9999.0)) < 120:
            spawn_chance = 0.12
        
        spawn_agent = rng.random() < spawn_chance
        if spawn_agent:
            memory["last_spawn_step"] = int(age * 10)

    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )
