from pathlib import Path
from typing import Optional

import numpy as np
from dex_retargeting.constants import OPERATOR2MANO, HandType, RetargetingType, RobotName, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig

MEDIAPIPE2PICO = {
    0: 1,  # WRIST -> Wrist
    1: 2,  # THUMB_CMC -> Thumb_metacarpal
    2: 3,  # THUMB_MCP -> Thumb_proximal
    3: 4,  # THUMB_IP  -> Thumb_distal
    4: 5,  # THUMB_TIP -> Thumb_tip
    5: 7,  # INDEX_FINGER_MCP -> Index_proximal
    6: 8,  # INDEX_FINGER_PIP -> Index_intermediate
    7: 9,  # INDEX_FINGER_DIP -> Index_distal
    8: 10,  # INDEX_FINGER_TIP -> Index_tip
    9: 12,  # MIDDLE_FINGER_MCP -> Middle_proximal
    10: 13,  # MIDDLE_FINGER_PIP -> Middle_intermediate
    11: 14,  # MIDDLE_FINGER_DIP -> Middle_distal
    12: 15,  # MIDDLE_FINGER_TIP -> Middle_tip
    13: 17,  # RING_FINGER_MCP -> Ring_proximal
    14: 18,  # RING_FINGER_PIP -> Ring_intermediate
    15: 19,  # RING_FINGER_DIP -> Ring_distal
    16: 20,  # RING_FINGER_TIP -> Ring_tip
    17: 22,  # PINKY_MCP -> Little_proximal
    18: 23,  # PINKY_PIP -> Little_intermediate
    19: 24,  # PINKY_DIP -> Little_distal
    20: 25,  # PINKY_TIP -> Little_tip
}

PICO2MEDIAPIPE = {
    1: 0,  # Wrist
    2: 1,  # Thumb_metacarpal -> THUMB_CMC
    3: 2,  # Thumb_proximal   -> THUMB_MCP
    4: 3,  # Thumb_distal     -> THUMB_IP
    5: 4,  # Thumb_tip        -> THUMB_TIP
    7: 5,  # Index_proximal   -> INDEX_FINGER_MCP
    8: 6,  # Index_intermediate -> INDEX_FINGER_PIP
    9: 7,  # Index_distal     -> INDEX_FINGER_DIP
    10: 8,  # Index_tip        -> INDEX_FINGER_TIP
    12: 9,  # Middle_proximal  -> MIDDLE_FINGER_MCP
    13: 10,  # Middle_intermediate -> MIDDLE_FINGER_PIP
    14: 11,  # Middle_distal    -> MIDDLE_FINGER_DIP
    15: 12,  # Middle_tip       -> MIDDLE_FINGER_TIP
    17: 13,  # Ring_proximal    -> RING_FINGER_MCP
    18: 14,  # Ring_intermediate -> RING_FINGER_PIP
    19: 15,  # Ring_distal      -> RING_FINGER_DIP
    20: 16,  # Ring_tip         -> RING_FINGER_TIP
    22: 17,  # Little_proximal  -> PINKY_MCP
    23: 18,  # Little_intermediate -> PINKY_PIP
    24: 19,  # Little_distal    -> PINKY_DIP
    25: 20,  # Little_tip       -> PINKY_TIP
}


def pico_hand_state_to_mediapipe(hand_state: np.ndarray) -> np.ndarray:
    mediapipe_state = np.zeros((21, 3), dtype=float)
    for pico_idx, mediapipe_idx in PICO2MEDIAPIPE.items():
        mediapipe_state[mediapipe_idx] = hand_state[pico_idx, :3]
    return mediapipe_state - mediapipe_state[0:1, :]  # Center at wrist


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / (np.linalg.norm(x) + 1e-6)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame


class DexHandTracker:
    def __init__(
        self,
        robot_name: RobotName,
        urdf_path: str,
        retargeting_type: RetargetingType,
        hand_type: HandType,
        config_path: Optional[str] = None,
    ):
        self.robot_name = robot_name
        self.retargeting_type = retargeting_type
        self.hand_type = hand_type
        self.urdf_path = urdf_path

        if config_path is not None:
            self.config_path = config_path
        else:
            self.config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
        self.OPERATOR2MANO = OPERATOR2MANO[hand_type]

        # Set the default URDF directory for the retargeting library
        robot_dir = Path(urdf_path).parent
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        # Build the retargeting module from the specified config
        self.retargeting = RetargetingConfig.load_from_file(self.config_path).build()

    def retarget(self, hand_pos: np.ndarray, wrist_rot: np.ndarray = None) -> Optional[np.ndarray]:
        if hand_pos is None or hand_pos.shape != (21, 3):
            return None

        # # Estimate the wrist orientation frame from the hand points
        if wrist_rot is None:
            # If wrist rotation is not provided, estimate it from the hand points
            wrist_rot = estimate_frame_from_hand_points(hand_pos)
        # mediapipe_wrist_rot = self.detector.estimate_frame_from_hand_points(hand_pos)

        # Transform the hand points into the MANO frame, which is the reference for retargeting
        transformed_pos = hand_pos @ wrist_rot @ self.OPERATOR2MANO

        # Prepare the reference values for the optimizer based on retargeting type
        indices = self.retargeting.optimizer.target_link_human_indices
        if self.retargeting_type == RetargetingType.position:
            ref_value = transformed_pos[indices, :]
        elif self.retargeting_type == RetargetingType.vector:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = transformed_pos[task_indices, :] - transformed_pos[origin_indices, :]
        else:
            raise NotImplementedError(f"Retargeting type {self.retargeting_type} is not supported.")

        try:
            # Perform the retargeting to get the robot's joint positions
            qpos = self.retargeting.retarget(ref_value)
            return qpos
        except (RuntimeWarning, RuntimeError) as e:
            # Catch potential numerical issues in the optimizer and return None
            print(f"Warning: Retargeting failed with an error: {e}")
            return None
