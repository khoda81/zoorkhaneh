from typing import Optional

from pathlib import Path

import gymnasium
import lovely_tensors as lt
import wandb

from general_q.agents import DQN, Agent
from general_q.utils import evaluate, load_pretrained, save_pretrained

lt.monkey_patch()
lt.set_config(precision=4, sci_mode=False)

SAVE_PATH = Path("tmp/pretrained")


def train(wandb_project="general_q") -> None:
    env = create_env()
    agent = load_agent(SAVE_PATH, env) or create_agent(env)
    # agent = create_agent(env)

    print("Training:")
    print(f"\tAgent: {agent}")
    try:
        print(f"\tEnv:   {env.unwrapped.spec.id}")
    except AttributeError:
        print(f"\tEnv:   {env.unwrapped!s}")

    def save_agent(step, *args, **kwargs):
        if (step + 1) % 2000 == 0:
            save_pretrained(agent, SAVE_PATH)

    def log_to_wandb(step, length, loss, reward, *args, **kwargs):
        wandb.log({
            "step"  : step,
            "loss"  : loss,
            "reward": reward,
            "length": length,
        })

    wandb.init(
        project=wandb_project,
        dir=SAVE_PATH.parent,
        name=str(agent),
    )

    with env, agent:
        evaluate(
            env,
            agent,
            steps=1_000_000,
            train=True,
            step_callback=save_agent,
            episode_callback=log_to_wandb,
        )

    save_pretrained(agent, SAVE_PATH)


def create_env():
    env = gymnasium.make(
        # id="Acrobot-v1",
        # id="BipedalWalker-v3",
        # id="Blackjack-v1",
        # id="CarRacing-v2",
        id="CartPole-v1",
        # id="CliffWalking-v0",
        # id="LunarLander-v2",
        # id="MountainCarContinuous-v01",
        # id="Pendulum-v1",

        render_mode="human",

        # continuous=False,
    )

    return env


def load_agent(path, env: gymnasium.Env) -> Optional[Agent]:
    path = Path(path)

    if not path.exists():
        return None

    best_time, best_agent = -float("inf"), None
    for path in path.iterdir():
        agent = load_pretrained(path, raise_error=True)
        if agent is None:
            continue

        # agent: Agent
        if (
            agent.action_space != env.action_space
        ) or (
            agent.observation_space != env.observation_space
        ):
            continue

        modify_time = path.stat().st_mtime
        if modify_time > best_time:
            best_time, best_agent = modify_time, agent

    return best_agent


def create_agent(env: gymnasium.Env) -> Agent:
    return DQN(
        action_space=env.action_space,
        observation_space=env.observation_space,
        memory_capacity=1024,
    )


if __name__ == "__main__":
    train()
