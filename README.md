# deploy_mujoco

MuJoCo deployment for G1 policies. Use the unified entry point:

```bash
python deploy_mujoco.py --config <path-to-deploy.yaml> --xml-path <path-to-mjcf.xml>
```

Policy configs and exported models are stored under:

```text
config/g1_29dof/config/policy/
```

## Install

Create a Python virtual environment and install dependencies from `pyproject.toml`:

```bash
cd deploy_mujoco
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Keep the environment active when running deploy commands:

```bash
source .venv/bin/activate
```

## Launch Commands

Velocity:

```bash
python deploy_mujoco.py \
  --config config/g1_29dof/config/policy/velocity/v0/params/deploy.yaml \
  --xml-path g1_description/g1_29dof_rev_1_0.xml
```

AMP:

```bash
python deploy_mujoco.py \
  --config config/g1_29dof/config/policy/velocity/g1_amp/params/deploy.yaml \
  --xml-path g1_description/g1_29dof_rev_1_0.xml
```

DWAQ:

```bash
python deploy_mujoco.py \
  --config config/g1_29dof/config/policy/velocity/dwaq/params/deploy.yaml \
  --xml-path g1_description/g1_29dof_stairs.xml
```

TS-Depth:

```bash
python deploy_mujoco.py \
  --config config/g1_29dof/config/policy/velocity/ts_depth/params/deploy.yaml \
  --xml-path g1_description/g1_29dof_ts_depth_stairs.xml
```

Parkour:

```bash
python deploy_mujoco.py \
  --config config/g1_29dof/config/policy/velocity/parkour/params/deploy.yaml \
  --xml-path g1_description/g1_29dof_torsobase_popsicle_with_shoe_terrain_park.xml
```

Mimic:

```bash
python deploy_mujoco.py \
  --config config/g1_29dof/config/policy/mimic/dance_102/params/deploy.yaml \
  --xml-path g1_description/g1_29dof_rev_1_0.xml
```

```bash
python deploy_mujoco.py \
  --config config/g1_29dof/config/policy/mimic/gangnam_style/params/deploy.yaml \
  --xml-path g1_description/g1_29dof_rev_1_0.xml
```

```bash
python deploy_mujoco.py \
  --config config/g1_29dof/config/policy/mimic/kungfu_1307_stage_i/params/deploy.yaml \
  --xml-path g1_description/g1_29dof_rev_1_0.xml
```

HITTER:

```bash
python deploy_mujoco_g1_hitter.py --config config/g1_hitter.yaml
```

## Controls

For velocity-command policies, use the numpad:

```text
6 / 4: increase / decrease vx
8 / 2: increase / decrease vy
9 / 3: increase / decrease wz
5: reset command to zero
```

