from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class AllegroHandCfg(BaseRobotCfg):
    name: str = "allegro_hand"
    num_joints: int = 16
    fix_base_link: bool = True
    usd_path: str = "roboverse_data/robots/allegro_hand/usd/allegro_hand.usd"
    mjcf_path: str = "roboverse_data/robots/allegro_hand/mjcf/allegro_hand.xml"
    urdf_path: str = "roboverse_data/robots/allegro_hand/urdf/allegro_touch_sensor.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "ffj0": BaseActuatorCfg(velocity_limit=30.0),
        "ffj1": BaseActuatorCfg(velocity_limit=30.0),
        "ffj2": BaseActuatorCfg(velocity_limit=30.0),
        "ffj3": BaseActuatorCfg(velocity_limit=30.0),
        "mfj0": BaseActuatorCfg(velocity_limit=30.0),
        "mfj1": BaseActuatorCfg(velocity_limit=30.0),
        "mfj2": BaseActuatorCfg(velocity_limit=30.0),
        "mfj3": BaseActuatorCfg(velocity_limit=30.0),
        "rfj0": BaseActuatorCfg(velocity_limit=30.0),
        "rfj1": BaseActuatorCfg(velocity_limit=30.0),
        "rfj2": BaseActuatorCfg(velocity_limit=30.0),
        "rfj3": BaseActuatorCfg(velocity_limit=30.0),
        "thj0": BaseActuatorCfg(velocity_limit=30.0),
        "thj1": BaseActuatorCfg(velocity_limit=30.0),
        "thj2": BaseActuatorCfg(velocity_limit=30.0),
        "thj3": BaseActuatorCfg(velocity_limit=30.0),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "ffj0": (-0.47, 0.47),
        "ffj1": (-0.196, 1.61),
        "ffj2": (-0.174, 1.709),
        "ffj3": (-0.227, 1.618),
        "mfj0": (-0.47, 0.47),
        "mfj1": (-0.196, 1.61),
        "mfj2": (-0.174, 1.709),
        "mfj3": (-0.227, 1.618),
        "rfj0": (-0.47, 0.47),
        "rfj1": (-0.196, 1.61),
        "rfj2": (-0.174, 1.709),
        "rfj3": (-0.227, 1.618),
        "thj0": (0.263, 1.396),
        "thj1": (-0.105, 1.163),
        "thj2": (-0.189, 1.644),
        "thj3": (-0.162, 1.719),
    }

    default_joint_positions: dict[str, float] = {
        "ffj0": 0.0,
        "ffj1": 0.0,
        "ffj2": 0.0,
        "ffj3": 0.0,
        "mfj0": 0.0,
        "mfj1": 0.0,
        "mfj2": 0.0,
        "mfj3": 0.0,
        "rfj0": 0.0,
        "rfj1": 0.0,
        "rfj2": 0.0,
        "rfj3": 0.0,
        "thj0": 0.0,
        "thj1": 0.0,
        "thj2": 0.0,
        "thj3": 0.0,
    }

    default_position: tuple[float, float, float] = (0.0, 0.0, 0.5)

    default_orientation: tuple[float, float, float, float] = (
        0.2575507164001465,
        0.28304457664489746,
        0.6833299994468689,
        -0.6217824220657349,
    )  # w, x, y, z
