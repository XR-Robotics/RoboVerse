from __future__ import annotations

import argparse
import os
import sys
import time

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import numpy as np
import torch
import xrobotoolkit_sdk as xrt
from dex_retargeting.constants import HandType, RetargetingType, RobotName
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.dex_hand_utils import DexHandTracker, pico_hand_state_to_mediapipe
from metasim.utils.setup_util import get_robot, get_sim_env_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="allegro_hand")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument(
        "--sim",
        type=str,
        default="isaaclab",
        choices=["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"],
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_envs: int = args.num_envs
    device = "cuda:0"
    robot = get_robot(args.robot)
    camera = PinholeCameraCfg(pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioCfg(robots=[robot], cameras=[camera], num_envs=num_envs, sim=args.sim)

    tic = time.time()
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    xrt.init()

    dextracker = DexHandTracker(
        robot_name=RobotName.shadow,
        urdf_path=os.path.join(
            os.path.dirname(__file__), "../../roboverse_data/robots/allegro_hand/urdf/allegro_hand_right.urdf"
        ),
        retargeting_type=RetargetingType.vector,
        hand_type=HandType.right,
        config_path=os.path.join(
            os.path.dirname(__file__), "../../roboverse_data/robots/allegro_hand/allegro_dex_retarget.yml"
        ),
    )

    init_states = [
        {
            "robots": {
                "allegro_hand": {"pos": torch.tensor([0.0, 0.0, 0.0]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
            },
            "objects": {},
        }
    ] * scenario.num_envs

    env.reset(states=init_states)

    joint_names = [
        "ffj0",
        "ffj1",
        "ffj2",
        "ffj3",
        "thj0",
        "thj1",
        "thj2",
        "thj3",
        "mfj0",
        "mfj1",
        "mfj2",
        "mfj3",
        "rfj0",
        "rfj1",
        "rfj2",
        "rfj3",
    ]

    step = 0
    while True:
        right_hand_state = np.array(xrt.get_right_hand_tracking_state())
        if np.all(right_hand_state == 0):
            log.warning("No right hand tracking data available. Skipping step.")
            continue
        mediapipe_hand_state = pico_hand_state_to_mediapipe(right_hand_state)
        pin_q = dextracker.retarget(mediapipe_hand_state)
        tic = time.time()
        obs, reward, success, timeout, extra = env.step([
            {
                robot.name: {
                    "dof_pos_target": {joint_name: joint_value for joint_name, joint_value in zip(joint_names, pin_q)}
                }
            }
        ])
        toc = time.time()
        log.trace(f"Step {step} took {toc - tic:.2f}s")
        step += 1

        if step % 10 == 0:
            log.info(f"Step: {step}, Obs: {obs}, Reward: {reward}, Success: {success}, Timeout: {timeout}")


if __name__ == "__main__":
    main()
