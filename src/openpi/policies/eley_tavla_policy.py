"""ELEY TA-VLA policy (unilateral data collection).

Adapts upstream TavlaInputs/Outputs (hardcoded to ALOHA's 14 dims) to ELEY's 16:
7 arm joints + 1 gripper per arm. Torque ("effort") stays a SEPARATE field from
state (TA-VLA token injection), not concatenated into state.

Raw dataset dims (from rebake config/robot_model/eley_tavla_unilateral.yaml):
- observation.state  : [18]   /bilateral_servo_state /position, GoMa order
- observation.effort : [history, 18]   /bilateral_servo_state /rtob, GoMa order
- action.left_arm/right_arm : [8] each   trajectory-controller commands (scapula first)
- action.left_gripper/right_gripper : [1] each

GoMa-18 order: [51..58 (L arm, 51=scapula), 61 (L grip), 71..78 (R arm, 71=scapula), 81 (R grip)].

Two 18->16 variants, selected by `include_scapula`:
- include_scapula=False (default, original): drop the scapula axes (idx 0, 9) ->
  [L_arm7, L_grip, R_arm7, R_grip], delta mask make_bool_mask(7,-1,7,-1).
- include_scapula=True: keep scapula, drop the grippers (idx 8, 17) ->
  [L_arm8, R_arm8] (scapula first per arm, matching the trajectory-controller
  order), delta mask make_bool_mask(16). Grippers are never commanded; the
  deploy side holds them at their init-pose value.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms

# Indices kept when converting a GoMa-18 vector to ELEY-16 (drop scapula at 0 and 9).
# Result order: [L_arm 1..7, L_grip 8, R_arm 10..16, R_grip 17].
_GOMA18_TO16 = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17)
# Scapula variant: keep scapula (idx 0, 9), drop the grippers (idx 8, 17).
# Result order: [L_arm 0..7, R_arm 9..16] (scapula first per arm).
_GOMA18_TO16_SCAP = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def eley_tavla_make_example() -> dict:
    return {
        "images": {
            "cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "state": np.ones((18,)),
        "effort": np.ones((1, 18)),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class ELEYTavlaInputs(transforms.DataTransformFn):
    """ELEY (16-dim) variant of TavlaInputs.

    Expects images dict {cam_high, cam_left_wrist, cam_right_wrist}, state[18],
    effort[history,18], and the four trajectory-controller action parts
    (action.left_arm[8], action.right_arm[8], action.left_gripper[1], action.right_gripper[1]).
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    # False: [L_arm7, L_grip, R_arm7, R_grip] (scapula dropped). True: [L_arm8, R_arm8]
    # (scapula included, grippers dropped). Must match the TrainConfig the checkpoint
    # was trained with.
    include_scapula: bool = False

    def __call__(self, data: dict) -> dict:
        idx = list(_GOMA18_TO16_SCAP if self.include_scapula else _GOMA18_TO16)

        state = np.asarray(data["state"])[..., idx]  # [18] -> [16]
        state = transforms.pad_to_dim(state, self.action_dim)

        in_images = data["images"]
        images = {
            "base_0_rgb": _parse_image(in_images["cam_high"]),
            "left_wrist_0_rgb": _parse_image(in_images["cam_left_wrist"]),
            "right_wrist_0_rgb": _parse_image(in_images["cam_right_wrist"]),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Effort: [history, 18] -> [history, 16]. Must equal model.effort_dim (NOT padded).
        if "effort" in data:
            inputs["effort"] = np.asarray(data["effort"])[..., idx]

        # Actions only present during training. Concatenate the controller parts into
        # [..., 16]: scapula variant keeps the full 8-dim arm commands and skips the
        # grippers ([L8, R8]); the default drops the leading scapula axis from each arm
        # and appends the grippers ([L7, Lg, R7, Rg]).
        if self.include_scapula:
            action_keys = ("action.left_arm", "action.right_arm")
            if all(k in data for k in action_keys):
                left_arm = np.asarray(data["action.left_arm"])[..., 0:8]
                right_arm = np.asarray(data["action.right_arm"])[..., 0:8]
                actions = np.concatenate([left_arm, right_arm], axis=-1)
                inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        else:
            action_keys = ("action.left_arm", "action.right_arm", "action.left_gripper", "action.right_gripper")
            if all(k in data for k in action_keys):
                left_arm = np.asarray(data["action.left_arm"])[..., 1:8]
                right_arm = np.asarray(data["action.right_arm"])[..., 1:8]
                left_grip = np.asarray(data["action.left_gripper"])
                right_grip = np.asarray(data["action.right_gripper"])
                actions = np.concatenate([left_arm, left_grip, right_arm, right_grip], axis=-1)
                inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ELEYTavlaOutputs(transforms.DataTransformFn):
    """Return only the first 16 (ELEY) action dims."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}
