import argparse, os, time, numpy as np, inspect
from pathlib import Path
from imageio.v2 import imwrite
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig
import rlbench.tasks as T

CANDIDATES = ["PickAndLift", "PickUpCup", "LiftNumberedBlock"]  # grasp/lift only

AVAILABLE = {}
for name, obj in inspect.getmembers(T):
  if inspect.isclass(obj) and name in CANDIDATES:
    snake = "".join([("_"+c.lower() if c.isupper() else c) for c in name]).lstrip("_")
    AVAILABLE[name] = obj
    AVAILABLE[snake] = obj  # allow snake_case too

def make_obs_config(w, h):
  oc = ObservationConfig(); oc.set_all(False)
  cam = CameraConfig()
  # turn on RGB only
  if hasattr(cam, "rgb"): cam.rgb = True
  if hasattr(cam, "depth"): cam.depth = False
  # Some RLBench versions use cam.segmentation, others cam.masks; both off
  if hasattr(cam, "segmentation"): cam.segmentation = False
  if hasattr(cam, "masks"): cam.masks = False
  if hasattr(cam, "image_size"): cam.image_size = (w, h)
  oc.front_camera = cam
  oc.wrist_camera = cam
  oc.left_shoulder_camera = cam
  return oc

def save(img, path):
  try: imwrite(path, img)
  except Exception as e: print(f"[WARN] couldn't save {path}: {e}")

def main():
  if not AVAILABLE:
    raise SystemExit("No candidate grasp tasks available in this RLBench install.")
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True, choices=sorted(AVAILABLE.keys()))
  ap.add_argument("--out", required=True)
  ap.add_argument("--width", type=int, default=640)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--steps", type=int, default=1)
  ap.add_argument("--variation", type=int, default=None)
  args = ap.parse_args()

  out = Path(os.path.expandvars(args.out)); out.mkdir(parents=True, exist_ok=True)
  env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()),
                    obs_config=make_obs_config(args.width, args.height),
                    headless=True)
  env.launch()

  task = env.get_task(AVAILABLE[args.task])
  if args.variation is not None:
    task.sample_variation(args.variation)
  descs, obs = task.reset()

  tag = f"{args.task}_{('var'+str(args.variation)) if args.variation is not None else 'random'}"
  ts = int(time.time()*1e3)
  for name in ("front_rgb","wrist_rgb","left_shoulder_rgb"):
    img = getattr(obs, name, None)
    if img is not None: save(img, str(out / f"{tag}_{ts}_{name.replace('_rgb','')}.png"))

  for _ in range(args.steps):
    a = np.random.normal(size=env.action_shape)
    obs, reward, done = task.step(a)

  ts2 = int(time.time()*1e3)
  for name in ("front_rgb","wrist_rgb","left_shoulder_rgb"):
    img = getattr(obs, name, None)
    if img is not None: save(img, str(out / f"{tag}_{ts2}_{name.replace('_rgb','')}_step.png"))

  env.shutdown()
  print("Saved snapshots to:", out)
  print("Available grasp tasks in this install:", ", ".join(sorted(set(AVAILABLE.keys()))))

if __name__ == "__main__":
  main()
