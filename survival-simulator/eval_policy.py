"""Headless evaluation: python eval_policy.py [--k=v ...] seed1 seed2 ...

Uses src.utils.controllers.my_policy.Policy (decide_all). Extra --key=value args override
Policy.DEFAULTS (floats). Prints final score / survival time per seed.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys, time, random
from src.core import SimulationCore
from src.utils.controllers.my_policy import Policy


def run(seed, params, max_time=3000, wall_limit=600, progress=0):
    sim = SimulationCore(seed=seed)
    policy = Policy(**params)
    rng = random.Random(seed)
    actions = []
    t0 = time.time()
    peak = 0
    next_print = progress
    while True:
        state = sim.step(actions)
        peak = max(peak, state["num_agents"])
        acts = policy.decide_all(state["observations"], rng)
        actions = [(a.agent_id, a) for a in acts]
        if progress and sim.env.time >= next_print:
            print(f'   t={sim.env.time:7.1f} score={state["score"]:8.1f} agents={state["num_agents"]:3d} '
                  f'trees={len(sim.env.trees):3d} fruits={len(sim.env.fruits):3d} preds={len(sim.env.predators):3d}', flush=True)
            next_print += progress
        if state["num_agents"] == 0 or sim.env.time > max_time or time.time() - t0 > wall_limit:
            break
    return dict(seed=seed, score=state["score"], time=sim.env.time, agents=state["num_agents"],
                peak=peak, wall=time.time() - t0)


if __name__ == "__main__":
    params, seeds, progress, wall, tmax = {}, [], 0, 600, 3000
    for a in sys.argv[1:]:
        if a.startswith("--progress="):
            progress = float(a.split("=")[1])
        elif a.startswith("--tmax="):
            tmax = float(a.split("=")[1])
        elif a.startswith("--wall="):
            wall = float(a.split("=")[1])
        elif a.startswith("--"):
            k, v = a[2:].split("=")
            params[k] = float(v)
        else:
            seeds.append(int(a))
    seeds = seeds or [1, 2, 3]
    scores = []
    for s in seeds:
        r = run(s, params, max_time=tmax, progress=progress, wall_limit=wall)
        scores.append(r["score"])
        print(f"seed={r['seed']} score={r['score']:.1f} time={r['time']:.1f} end_agents={r['agents']} peak={r['peak']} wall={r['wall']:.1f}s", flush=True)
    print(f"MEAN score = {sum(scores)/len(scores):.1f}  params={params}")