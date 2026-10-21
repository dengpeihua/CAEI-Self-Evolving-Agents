"""
Static-partition trajectory collector for ALFWorld.

Goal: walk through the 3553 train games in order, in fixed-size batches, with
NO replacement across the whole run. Each game is rolled out exactly once
with a no-memory policy and saved as a trajectory.

Why not reuse gpt4o_alfworld.py?
  - Its train path samples with replacement (each env picks a random game
    from the full pool at every reset).
  - Its test_times is hard-coded to 1000 when split=="train", ignoring
    the env var.
  - Static partitioning needs start_idx to be honored, which the train
    branch of envs.py silently discards.

This script forces is_train=False (so envs.py honors start_idx) but tells
the underlying AlfredTWEnv to load the train split (env_kwargs=
{'eval_dataset': 'train'}). That gives us a deterministic, non-repeating
walk through the 3553 train games.

Usage:
    OPENAI_BASE_IP_ADDR=127.0.0.1:8001 \
    EMB_VLLM_SERVER=127.0.0.1 \
    EMBED_MODEL_NAME=./Qwen3-Embedding-4B \
    python -m examples.prompt_agent.collect_trajectories
"""

import asyncio
import json
import logging
import math
import os
import sys
import time
import uuid
from datetime import datetime

# Pull in the agent + env plumbing from the existing script.
# gpt4o_alfworld.py runs `os.environ["SPLIT"]` at import time, so make
# sure the variable is set before the import.
os.environ.setdefault("SPLIT", "train")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt4o_alfworld import (  # noqa: E402
    Agent,
    AlfWorldEnvironmentManager,
    extract_action_from_response,
    ALFWORLD_TEMPLATE_NO_HIS,
)
# alfworld_projection + build_alfworld_envs are imported locally inside
# gpt4o_alfworld.build_env, so pull them from the original module path.
from agent_system.environments.env_package.alfworld import (  # noqa: E402
    alfworld_projection,
    build_alfworld_envs,
)


# ----------------------------- configuration -----------------------------

# Where the trajectory JSON lives. We write one file at the end with the
# trajectory list (mirrors the layout produced by gpt4o_alfworld.py so the
# build_index_*.py scripts can find it via a symlink).
LOG_DIR = os.environ.get(
    "LOG_DIR",
    f"logs/alfworld/{os.environ.get('MODEL_NAME', 'Qwen/Qwen3-4B-Instruct-2507').replace('/', '_')}",
)
os.makedirs(LOG_DIR, exist_ok=True)

# Use the existing path convention so build_index.py can find the file
# (it reads logs/{dataset}_old/{model_safe}/traj_train.json, but we still
# write the timestamped variant and a soft-link to the canonical name).
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_UUID = str(uuid.uuid4())[:8]
TRAJ_FILE = os.path.join(
    LOG_DIR, f"traj_train_{TIMESTAMP}_{RUN_UUID}_no-memory_static.json"
)
STATS_FILE = TRAJ_FILE.replace("traj_", "stats_").replace(".json", ".txt")


# ----------------------------- logging ----------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(STATS_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logging.info(f"Saving trajectories to {TRAJ_FILE}")


# ----------------------------- helpers ----------------------------------


def build_env_manager(batch_size: int, start_idx: int):
    """Build an AlfWorldEnvironmentManager bound to games
    [start_idx, start_idx+batch_size) of the train split.

    is_train=False forces envs.py to honor start_idx; eval_dataset='train'
    makes AlfredTWEnv load the 3553 train game files (envs.py:107).
    """
    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        "../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    env_kwargs = {"eval_dataset": "train"}
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
    envs = build_alfworld_envs(
        alf_config_path,
        seed=1,
        env_num=batch_size,
        group_n=1,
        is_train=False,            # <-- so envs.py uses start_idx
        env_kwargs=env_kwargs,     # <-- tells base_env to load train split
        resources_per_worker=resources_per_worker,
        start_idx=start_idx,
    )
    return AlfWorldEnvironmentManager(
        envs, alfworld_projection, "alfworld/AlfredThorEnv", mem_type=None, topk=1
    )


def build_initial_prompt(text_obs: str, admissible_actions: list) -> str:
    """Build the no-memory initial prompt for one env."""
    reformatted = "\n ".join(f"'{s}'" for s in admissible_actions if s != "help")
    return ALFWORLD_TEMPLATE_NO_HIS.format(
        current_observation=text_obs,
        admissible_actions=reformatted,
    )


# ----------------------------- rollout loop -----------------------------


async def run_batch(agent: Agent, env_manager, start_idx: int, batch_size: int, max_steps: int):
    """Roll out one batch of `batch_size` envs to completion (or max_steps).

    Returns a list of trajectory dicts in env order, each with:
        { "steps": [...], "won": bool, "reward": float, "start_idx": int }
    """
    obs, infos = env_manager.reset(kwargs={})
    admissible = env_manager.envs.get_admissible_commands  # populated by reset()

    env_dones = [False] * batch_size
    env_won = [False] * batch_size
    env_steps: list = [[] for _ in range(batch_size)]
    env_step_count = [0] * batch_size

    for step_idx in range(max_steps):
        active = [i for i in range(batch_size) if not env_dones[i]]
        if not active:
            break

        prompts = []
        for i in active:
            prompts.append(build_initial_prompt(obs["text"][i], admissible[i]))

        # Fan out LLM calls (vLLM client handles concurrent requests).
        results = await asyncio.gather(
            *(agent.get_action_from_gpt(p) for p in prompts)
        )

        # Fill actions for every env. Already-done envs get a no-op
        # ("look") so the projection layer never sees None; envs.py will
        # still report done=True for them and we'll skip the response.
        actions = ["look"] * batch_size
        step_timings = [{} for _ in range(batch_size)]
        for active_idx, (text, _t) in enumerate(results):
            i = active[active_idx]
            actions[i] = extract_action_from_response(text) or "look"
            step_timings[i] = {"action_time": _t}

        # Build the next-step text obs using the env manager (handles
        # action projection + admissible actions).
        next_obs, rewards, dones, new_infos = env_manager.step(actions)

        # Record steps for active envs.
        for i in active:
            env_steps[i].append({
                "step_idx": step_idx,
                "curr_prompt": obs["text"][i],
                "curr_action": actions[i],
            })
            if dones[i]:
                env_dones[i] = True
                env_won[i] = bool(new_infos[i].get("won", False))
                env_step_count[i] = step_idx + 1
        obs = next_obs
        admissible = env_manager.envs.get_admissible_commands

        logging.info(
            f"  step {step_idx}: {sum(env_dones)}/{batch_size} done, "
            f"SR={sum(env_won) / max(1, sum(env_dones)):.3f}"
        )

    # Force-close envs that never finished (ran out of max_steps).
    for i in range(batch_size):
        if not env_dones[i]:
            env_dones[i] = True
            env_step_count[i] = max_steps
            env_won[i] = False

    trajs = []
    for i in range(batch_size):
        trajs.append({
            "steps": env_steps[i],
            "won": env_won[i],
            "reward": 1.0 if env_won[i] else 0.0,
            "start_idx": start_idx + i,
            "num_steps": env_step_count[i],
        })
    return trajs


# ----------------------------- main ------------------------------------


async def main():
    # ---- run-time configuration (env vars override defaults) ----
    num_games = int(os.environ.get("NUM_GAMES", 3553))
    batch_size = int(os.environ.get("BATCH_SIZE", 100))
    max_steps = int(os.environ.get("MAX_STEPS", 50))
    temperature = float(os.environ.get("TEMPERATURE", 0.7))

    num_batches = math.ceil(num_games / batch_size)
    logging.info(
        f"Static-partition collect: {num_games} games, "
        f"batch_size={batch_size}, num_batches={num_batches}, max_steps={max_steps}"
    )

    agent = Agent()
    all_trajs = []
    total_won = 0

    overall_start = time.time()
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_games)
        bs = end - start
        logging.info(
            f"\n=== Batch {batch_idx + 1}/{num_batches}: games [{start}:{end}] ==="
        )
        t0 = time.time()
        env_manager = build_env_manager(bs, start)
        try:
            batch_trajs = await run_batch(agent, env_manager, start, bs, max_steps)
        finally:
            env_manager.close()
        elapsed = time.time() - t0

        batch_won = sum(1 for t in batch_trajs if t["won"])
        total_won += batch_won
        all_trajs.extend(batch_trajs)
        logging.info(
            f"  batch done in {elapsed:.1f}s | "
            f"batch SR={batch_won}/{bs}={batch_won / max(bs, 1):.3f} | "
            f"running SR={total_won}/{len(all_trajs)}="
            f"{total_won / max(len(all_trajs), 1):.3f}"
        )

        # Persist after every batch so a kill mid-run still leaves a
        # usable partial file.
        with open(TRAJ_FILE, "w") as f:
            json.dump(all_trajs, f, indent=2)

    logging.info(
        f"\nDone. Collected {len(all_trajs)} trajectories "
        f"({total_won} won, SR={total_won / max(len(all_trajs), 1):.3f}) "
        f"in {time.time() - overall_start:.1f}s"
    )
    logging.info(f"Trajectories: {TRAJ_FILE}")


if __name__ == "__main__":
    os.environ['ENV_NAME'] = 'alfworld'
    os.environ['OPENAI_BASE_IP_ADDR'] = '127.0.0.1:8001'
    os.environ['EMB_VLLM_SERVER'] = '127.0.0.1'
    os.environ['EMBED_MODEL_NAME'] = './Qwen3-Embedding-4B'
    os.environ['ENV_NAME'] = 'alfworld'
    os.environ['MEMORY_MODEL_NAME'] = 'Qwen/Qwen3-4B-Instruct-2507'
    os.environ['MODEL_NAME'] = 'Qwen/Qwen3-4B-Instruct-2507'
    os.environ['OPENAI_API_KEY'] = 'EMPTY'
    os.environ['OPENAI_BASE_URL'] = 'http://127.0.0.1:8001/v1'
    asyncio.run(main())



    