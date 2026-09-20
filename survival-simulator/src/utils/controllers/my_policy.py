"""Optimized hivemind policy for Nordic AI Cup 2026 survival simulator.

Key Improvements:
- Energy conservation: Zero-turn camping (utilizing 360-degree hearing for alert).
- Population management: Removed cohesion clustering; added dispersal to avoid starvation & predator chain-kills.
- Smart reproduction: Opportunistic spawning near food sources, no artificial probabilistic delays.
- Obstacle-safe fleeing: Ensures forward vision when escaping to prevent backing into blind walls.
"""

import math
import random
from typing import Dict, List
from src.utils.DTOs import ActionRequest

PI = math.pi


def _wrap(a: float) -> float:
    return (a + PI) % (2 * PI) - PI


class Policy:
    DEFAULTS = dict(
        flee_range=140.0,
        sprint_range=85.0,
        juke_range=60.0,
        juke_away=0.6,
        juke_sprint_range=32.0,
        sprint_min_energy=30.0,
        camp_radius=22.0,
        wander_sigma=0.08,
        wall_stop=38.0,
        wall_turn=0.9,
        agent_repel=50.0,  # 增大排斥距离，防止扎堆抢食
        spawn_safe_energy=135.0,  # 能量安全线：扣除100后至少保留35
        old_age=50.0,  # 临界老龄，尽快繁衍基因
        max_pop=18,  # 适度调高以拉高基础存活时长贡献
        stuck_ticks=12,
        stuck_progress=3.0,
        ignore_ticks=35,
    )

    def __init__(self, **overrides):
        p = dict(self.DEFAULTS)
        p.update(overrides)
        self.p = p
        self._reset()

    def _reset(self):
        self.mem: Dict[int, dict] = {}
        self.tick = 0

    @staticmethod
    def _edge_info(edges):
        front_d, front_side = 1e9, 0
        px = py = 0.0
        for e in edges:
            (x1, y1), (x2, y2) = e
            dx, dy = x2 - x1, y2 - y1
            l2 = dx * dx + dy * dy
            if l2 < 1e-9:
                pts = [(x1, y1)]
            else:
                t = -(x1 * dx + y1 * dy) / l2
                t = 0.0 if t < 0 else 1.0 if t > 1 else t
                pts = [(x1 + t * dx, y1 + t * dy), (x1, y1), (x2, y2)]
            for cx, cy in pts:
                d = math.hypot(cx, cy)
                if d < 1e-6:
                    continue
                ang = math.atan2(cy, cx)
                if abs(ang) < 0.96 and d < front_d:
                    front_d, front_side = d, (1 if cy > 0 else -1)
                if d < 32.0:
                    w = (32.0 - d) / 32.0
                    px -= cx / d * w
                    py -= cy / d * w
        return front_d, front_side, px, py

    def decide_all(self, states: List[dict], rng: random.Random) -> List[ActionRequest]:
        if states and len(states) <= 5 and all(s["age"] < 0.35 for s in states):
            self._reset()
        self.tick += 1
        self._n = len(states)
        alive = {s["agent_id"] for s in states}
        for k in list(self.mem.keys()):
            if k not in alive:
                del self.mem[k]
        return [self._decide(s, self._n, rng) for s in states]

    def _decide(self, st: dict, n_alive: int, rng: random.Random) -> ActionRequest:
        p = self.p
        aid = st["agent_id"]
        E, maxE, age = st["energy"], st["max_energy"], st["age"]
        speed, sprint = st["speed"], st["sprint_speed"]
        mem = self.mem.setdefault(
            aid, {"hist": [], "ignore_until": -1, "side": 0, "side_until": -1}
        )
        tick = self.tick

        fruits, trees, preds, agents, edges = [], [], [], [], []
        for o in st["observations"]:
            t = o.get("type")
            if t == "Fruit":
                fruits.append((o["distance"], o["angle"]))
            elif t == "Tree":
                trees.append((o["distance"], o["angle"]))
            elif t == "Predator":
                preds.append((o["distance"], o["angle"], o.get("rel_dir", 0.0)))
            elif t == "Agent":
                agents.append((o["distance"], o["angle"]))
            elif t == "Edge":
                edges.append(o["coords"])

        front_d, front_side, sx, sy = self._edge_info(edges)
        wall_ahead = front_d < p["wall_stop"]

        # 果子卡墙/死角检测
        fruits_use = fruits if tick >= mem["ignore_until"] else []
        if fruits_use:
            dnear = min(fruits_use)[0]
            h = mem["hist"]
            h.append(dnear)
            if len(h) > p["stuck_ticks"]:
                h.pop(0)
                if h[0] - h[-1] < p["stuck_progress"] and dnear > 10:
                    mem["ignore_until"] = tick + p["ignore_ticks"]
                    mem["hist"] = []
                    fruits_use = []
        else:
            mem["hist"] = []

        dist_move = 0.0
        heading = 0.0
        turn = 0.0
        near_pred = min(preds) if preds else None

        # ---------------- 状态机判断 ----------------
        if near_pred is not None and near_pred[0] < p["flee_range"]:
            # ---- 1. 逃避捕食者 (FLEE) ----
            d, a, r = near_pred
            can_sprint = E > p["sprint_min_energy"]

            if d < p["juke_range"]:
                # 捕食者贴脸：横向机动切角
                psi = a + PI - r
                ax, ay = -math.cos(a), -math.sin(a)
                best = None
                for sgn in (1.0, -1.0):
                    sx_, sy_ = math.cos(psi + sgn * PI / 2), math.sin(
                        psi + sgn * PI / 2
                    )
                    score = sx_ * ax + sy_ * ay + 0.3 * (
                        1.0 if sgn == mem.get("juke_side", 1.0) else 0.0
                    )
                    if best is None or score > best[0]:
                        best = (score, sgn, sx_, sy_)
                _, sgn, sx_, sy_ = best
                mem["juke_side"] = sgn
                vx = sx_ + p["juke_away"] * ax + 1.8 * sx
                vy = sy_ + p["juke_away"] * ay + 1.8 * sy
                heading = math.atan2(vy, vx)
                dist_move = (
                    sprint if (d < p["juke_sprint_range"] and can_sprint) else speed
                )
                turn = heading
            else:
                # 常规逃逸：面向逃逸路线，防止看不到身后墙体
                vx, vy = -math.cos(a) + 1.6 * sx, -math.sin(a) + 1.6 * sy
                heading = math.atan2(vy, vx)
                if wall_ahead and abs(heading) < 0.9:
                    heading = 1.3 * (front_side * -1 if front_side else 1)
                dist_move = sprint if (d < p["sprint_range"] and can_sprint) else speed
                turn = heading

        elif fruits_use:
            # ---- 2. 觅食 (EAT) ----
            d, a = min(fruits_use)
            vx, vy = math.cos(a) + 0.8 * sx, math.sin(a) + 0.8 * sy
            heading = math.atan2(vy, vx)
            dist_move = min(speed, d)
            turn = heading

        elif trees:
            # ---- 3. 果树下驻守 (CAMP / GOTO_TREE) ----
            d, a = min(trees)
            if d > p["camp_radius"]:
                vx, vy = math.cos(a) + 0.8 * sx, math.sin(a) + 0.8 * sy
                heading = math.atan2(vy, vx)
                dist_move = min(speed, d - p["camp_radius"] * 0.4)
                turn = heading
            else:
                # 已经就位：完全静止不耗体力，靠听觉被动警戒
                dist_move = 0.0
                turn = 0.0

        else:
            # ---- 4. 巡游探索 (EXPLORE) ----
            if wall_ahead:
                if tick > mem["side_until"]:
                    mem["side"] = -front_side if front_side else rng.choice((-1, 1))
                    mem["side_until"] = tick + 22
                heading = 0.0
                dist_move = 0.0 if front_d < 22 else speed * 0.4
                turn = mem["side"] * p["wall_turn"]
            else:
                vx, vy = 1.0 + 1.6 * sx, 1.6 * sy
                heading = math.atan2(vy, vx)
                dist_move = speed
                turn = heading + rng.gauss(0.0, p["wander_sigma"])

        # 同伴排斥（无论是在巡游还是露营，保持距离避免争抢同一片果源）
        if near_pred is None and agents:
            d, a = min(agents)
            if d < p["agent_repel"]:
                repel_h = _wrap(a + PI)
                heading = repel_h
                dist_move = max(dist_move, speed * 0.5)
                turn = repel_h * 0.7

        # ---------------- 5. 繁殖决策 (REPRODUCTION) ----------------
        spawn = False
        if self._n < p["max_pop"]:
            # 老年且能量大于 105，抓紧遗传；或能量充足且靠近食物/树
            can_afford = (E - 100.0) >= 30.0
            is_old = age >= p["old_age"] and E > 105.0
            is_thriving = can_afford and (
                E > p["spawn_safe_energy"] and (fruits or trees)
            )

            if is_old or is_thriving:
                spawn = True
                self._n += 1

        return ActionRequest(
            agent_id=aid,
            move_distance=float(max(0.0, dist_move)),
            move_direction=float(_wrap(heading)),
            turn_angle=float(max(-PI, min(PI, turn))),
            spawn_agent=bool(spawn),
        )


_default = Policy()


def action_decision(observation_response: dict, rng: random.Random):
    return _default._decide(observation_response, 0, rng)