import os
import random
import threading

from fastapi import FastAPI, Body

from src.utils.DTOs import StepResponse, ActionRequest
from src.utils.controllers.my_policy import Policy

HOST = "0.0.0.0"
PORT = 8000

app = FastAPI(title="Survival Simulator Agent Endpoint")

_policy = Policy()
_rng = random.Random(int(os.environ.get("POLICY_SEED", "12345")))
_lock = threading.Lock()  # one game at a time; keeps policy memory consistent


@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    states = [a.model_dump() for a in step.agent_status]
    with _lock:
        actions = _policy.decide_all(states, _rng)
    # Must return {"actions": [...]} format
    return {"actions": [a.model_dump() for a in actions]}


@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")