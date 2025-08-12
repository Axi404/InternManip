# Challenge Participation Guidelines (WIP)

This document walks you through the end-to-end workflow: from pulling the base image, developing your model, to submitting and seeing your results on the leaderboard.


## 🧩 Environment Setup (Linux)  
### Pull the latest code
```bash
git clone --depth 1 --recurse-submodules https://github.com/InternRobotics/InternManip.git
```

### Prepare train & validation dataset
```bash
dataset_save_path=/custom/dataset/save/path # Please replace it with your local path. Note that at least 50G of storage space is required.
mkdir InternManip/data
ln -s ${dataset_save_path} InternManip/data/dataset

sudo apt-get install git git-lfs
git lfs install

git lfs clone https://huggingface.co/datasets/InternRobotics/IROS-2025-Challenge-Manip ${dataset_save_path}/IROS-2025-Challenge-Manip
```

### Pull base Docker image
```bash
docker pull crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internmanip:v1.0
```
  
### Run the container
```bash
xhost +local:root # Allow the container to access the display

cd InternManip

docker run --name internmanip -it --rm --gpus all --network host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e "DISPLAY=${DISPLAY}" \
  --entrypoint /bin/bash \
  -w /root/InternManip \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v ${PWD}:/root/InternManip \
  -v ${HOME}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ${HOME}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ${HOME}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ${HOME}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ${HOME}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ${HOME}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ${HOME}/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ${HOME}/docker/isaac-sim/documents:/root/Documents:rw \
  crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internmanip:v1.0
```  
  


## 🛠️ Local Development & Testing
  
### Implement your policy  
- Implement your policy under `internmanip/agent` and `internmanip/model`.
  
---  
  
### Test  
#### 1. Configuration File Preparation  
Create your custom configuration file by replicating the structure of: 
`challenge/run_configs/eval/gr00t_n1_5_on_genmanip.py`


#### 2. Evaluation Methods (Choose One)  
**Method 1: Dual-Terminal Approach**  
Terminal 1 (Agent Server):  
```bash
conda activate your_agent_env_name
python -m scripts.eval.start_agent_server --host localhost --port 5000
```

Terminal 2 (Evaluator):  
```bash
conda activate genmanip
python -m scripts.eval.start_evaluator \
  --config challenge/run_configs/eval/gr00t_n1_5_on_genmanip.py \
  --server
```

**Method 2: Single-Command Approach**  
```bash
./challenge/bash_scripts/eval.sh \
  --server_conda_name your_agent_env_name \
  --config challenge/run_configs/eval/gr00t_n1_5_on_genmanip.py \
  --server \
  --dataset_path ./data/dataset \
  --res_save_path ./results
```
▶ **Output logs in Method 2** : Check `./results/server.log` and `./results/eval.log`

### How to Integrate Your Custom Model for Inference in Internmanip

To run your own model in **Internmanip**, you mainly need to:

* Receive the inputs from the environment in the expected format.
* Run your model inference with those inputs.
* Return outputs in the required format so the server can simulate and send back the next observation.

#### 1. Input to the `step` Function

The core integration point is the `step` function located in `internmanip/agent/genmanip_agent.py`. It receives a **list with one element**, which is a Python dictionary containing robot state and sensor data. The complete field structure is described in the [Aloha Split Input Format](https://internrobotics.github.io/user_guide/internmanip/tutorials/environment.html#when-robote-type-is-aloha-split).

Here is an example of how you might access inputs inside this function:

```python
def step(self, inputs):
  # inputs is a list with one dict
  input_data = inputs[0]
  # Get the joint positions (28 DOF including the base)
  joint_positions = input_data["robot"]["joints_state"]["positions"]
  # joint indices follow the official docs
  left_arm_qpos = joint_positions[[12, 14, 16, 18, 20, 22]]
  right_arm_qpos = joint_positions[[13, 15, 17, 19, 21, 23]]
  left_gripper_qpos = joint_positions[[24, 25]]
  right_gripper_qpos = joint_positions[[26, 27]]
  # Get RGB images from the three cameras
  left_rgb = input_data["robot"]["sensors"]["left_camera"]["rgb"]
  right_rgb = input_data["robot"]["sensors"]["right_camera"]["rgb"]
  top_rgb = input_data["robot"]["sensors"]["top_camera"]["rgb"]
  # Get the task description instruction
  instruction = input_data["annotation"]["human"]["action"]["task_description"]
  # Prepare your model input here (e.g., packaging images, qpos, and instruction)
  model_input = {
      "left_rgb": left_rgb,
      "right_rgb": right_rgb,
      "top_rgb": top_rgb,
      "left_arm_qpos": left_arm_qpos,
      "right_arm_qpos": right_arm_qpos,
      "left_gripper_qpos": left_gripper_qpos,
      "right_gripper_qpos": right_gripper_qpos,
      "instruction": instruction,
  }
```

#### 2. Running Model Inference

Use your model to run inference on the processed inputs, for example:

```python
pred_actions = self.policy_model.inference(model_input)['action_pred'][0].cpu().numpy()
```

#### 3. Return Output from `step`

Finally, package your output into the required format: a list containing one dictionary, for example:

```python
def step(self, inputs):
  ...
  output = [{
      "action": {
        'left_arm_action': pred_actions[:6], # (6,)
        # Scale from [0,1] to [-1,1], where 1=open and -1=close
        'left_gripper_action': [pred_actions[12] * 2 - 1] * 2, # (2,) || -1 or 1 -> open or close
        'right_arm_action': pred_actions[6:12], # (6,)
        'right_gripper_action': [pred_actions[13] * 2 - 1] * 2, # (2,) || -1 or 1 -> open or close
    },
  }]
  return output
```

#### Notes:

* All joint positions and actions are expressed as absolute joint positions (absolute qpos) or absolute end-effector pose (absolute eepose) with respect to each arm base frame. If you want to control relative poses, you need to convert it to absolute control yourself.
* The `step` function and input/output formats are described in detail here:
  [Aloha Split Input/Output Format](https://internrobotics.github.io/user_guide/internmanip/tutorials/environment.html#when-robote-type-is-aloha-split)

Currently, the codebase includes an example step implementation that supports smoothing and outputs relative joint positions. You can refer to it or reuse it directly in your development.

### Critical Notes
1. **Environment Parameters**  
   Refer to : [Evaluation Environment Configuration](https://internrobotics.github.io/user_guide/internmanip/tutorials/environment.html#evaluation-environment-configuration-parameters)  
   **Mandatory**: `robot_type` must be set to `aloha_split` for this challenge.

2. **I/O Specifications**  
   For GenManip environment data formats and action schemas: [I/O Documentation](https://internrobotics.github.io/user_guide/internmanip/tutorials/environment.html#when-robote-type-is-aloha-split)

## 📦 Packaging & Submission

### ✅ Create your image registry  
You can follow the following [document](https://help.aliyun.com/zh/acr/user-guide/create-a-repository-and-build-images?spm=a2c4g.11186623.help-menu-60716.d_2_15_4.75c362cbMywaYx&scm=20140722.H_60997._.OR_help-T_cn~zh-V_1) to create a free personal image registry. After uploading the image to be submitted, please set it to public access.

### ✅ Build your submission image

You can build a new image by customizing the `Dockerfile`, or use command `docker commit`:

```bash
$ docker commit internmanip your-custom-image:v1
```

Push to your public registry
```bash
$ docker tag your-custom-image:v1 your-registry/submit-image:v1
$ docker push your-registry/submit-image:v1
```

> **NOTE !**  
> - **Make sure your Modified Code are correctly packaged in your submitted Docker image at `/root/InternManip`.**  
> - **Your trained model weights are best placed in your submitted Docker image at `/root/InternManip/data/model`.**  
> - **Please delete cache and other operations to reduce the image size.**  

### ✅ Submit your image URL on Eval.AI

#### Submission Format

Create a JSON file with your Docker image URL and team information. The submission must follow this exact structure:

```json
{
    "url": "your-registry/submit-image:v1",
    "args": {
        "agent_conda_name": "gr00t",
        "config_path": "/root/InternManip/challenge/run_configs/eval/gr00t_n1_5_on_genmanip.py"
    },
    "team": {
        "name": "your-team-name",
        "members": [
            {
                "name": "John Doe",
                "affiliation": "University of Example",
                "email": "john.doe@example.com",
                "leader": true
            },
            {
                "name": "Jane Smith",
                "affiliation": "Example Research Lab",
                "email": "jane.smith@example.com",
                "leader": false
            }
        ]
    }
}
```

##### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Complete Docker registry URL for your submission image |
| `agent_conda_name` | string | The name of the agent conda env being implemented |
| `config_path` | string | The path of test config file.<br>e.g. /root/InternManip/challenge/run_configs/eval/gr00t_n1_5_on_genmanip.py |
| `team.name` | string | Official team name for leaderboard display |
| `team.members` | array | List of all team members with their details |
| `members[].name` | string | Full name of team member |
| `members[].affiliation` | string | University or organization affiliation |
| `members[].email` | string | Valid contact email address |
| `members[].leader` | boolean | Team leader designation (exactly one must be `true`) |


For detailed submission guidelines and troubleshooting, refer to the official Eval.AI platform documentation.

## Official Evaluation Flow
### DSW Creation
- We use the AliCloud API to instantiate a DSW from your image link.
- The system mounts our evaluation config + full dataset (val_seen, val_unseen, test).
### Evaluation Execution
- Via SSH + `screen`, we launch `scripts/eval/run_eval.sh`.
- A polling loop watches for `results.json`.
### Results Collection
- Upon completion, metrics for each split are parsed and pushed to Eval.AI leaderboard.

> 😄 Good luck, and may the best vision-based policy win!
