# Alpamayo1.5 → NavSim 适配方案

## 一、核心差异对比

| 维度 | Alpamayo1.5 | NavSim |
|------|-------------|--------|
| **相机配置** | 4相机(cross_left/front_wide/cross_right/front_tele), 4帧/相机@10Hz | 8相机(f0/l0-l2/r0-r2/b0), 4帧@2Hz |
| **ego历史** | 16步@10Hz(1.6s), xyz+3x3旋转矩阵 | 4帧@2Hz(2s), SE2(x,y,heading) |
| **轨迹输出** | 64步@10Hz(6.4s), xyz+3x3旋转 | 8步@0.5Hz(4s), SE2(x,y,heading) |
| **坐标系** | ego车辆局部坐标系 | ego后轴局部坐标系(SE2) |
| **数据源** | PhysicalAI-AV Dataset (HF streaming) | OpenScene (nuScenes扩展), 本地pickle |
| **评估** | minADE | Extended PDM Score (EPDMS) |

## 二、适配架构

```
NavSim AgentInput
  ├── ego_statuses (4帧 SE2)  ──→  插值到16步, heading→rot_matrix ──→  Alpamayo ego_history
  ├── cameras (8相机)         ──→  选3相机(f0,l0,r0), 映射到alpamayo索引 ──→  Alpamayo image_frames
  └─ ...
  
Alpamayo Output
  ├── pred_xyz (64步, xyz)    ──→  降采样到8步, z丢弃 ──→  NavSim poses (x,y)
  ├── pred_rot (64步, 3x3)    ──→  降采样到8步, rot→heading ──→  NavSim poses (heading)
  └─ CoT text                 ──→  (可选, 不参与评分)
```

## 三、关键适配细节

### 3.1 相机映射

NavSim的8相机和Alpamayo的7相机索引(0-6)之间不存在完美对齐。Alpamayo训练用的是NVIDIA专用相机(front_wide_120fov, front_tele_30fov, cross_left_120fov, cross_right_120fov)，NavSim用的是nuScenes相机布局。

映射方案（取最接近语义的）：
- `cam_f0`(front) → Alpamayo index 1 (Front camera / front_wide)
- `cam_l0`(front-left) → Alpamayo index 0 (Front left / cross_left)
- `cam_r0`(front-right) → Alpamayo index 2 (Front right / cross_right)

**不使用cam_l1/l2/r1/r2/b0**：Alpamayo没有对应的rear/side-rear相机训练数据，输入未知相机索引可能导致模型行为异常。

**时间帧问题**：Alpamayo每相机用4帧@10Hz(覆盖0.4s历史)，NavSim只给当前帧。方案：**重复当前帧4次**。这会导致模型看到的"历史"全是同一帧，丢失运动信息，但对静态场景影响有限。更好的方案需要从NavSim的4个历史帧中取不同时间步的图像。

### 3.2 ego历史插值

NavSim给4帧@2Hz(间隔0.5s)，Alpamayo要16步@10Hz(间隔0.1s)。

方案：线性插值x,y,对heading用unwrap避免角度跳变。

注意：NavSim的ego_pose是后轴坐标系下的(x,y,heading)，Alpamayo用的是车辆中心的3D坐标。二者在平面上近似等价（后轴偏移很小），但heading的含义相同。

### 3.3 输出转换

Alpamayo输出64步@10Hz的3D坐标+旋转矩阵，NavSim要8步@0.5Hz的SE2。

降采样：对Alpamayo的第i步，NavSim pose j对应时间 (j+1)*0.5s，
对应Alpamayo步 = round((j+1)*0.5 / 0.1) - 1 = 5*j+4

即取步[4, 9, 14, 19, 24, 29, 34, 39]

heading从3x3旋转矩阵提取：atan2(R[1,0], R[0,0])

### 3.4 评估流程

NavSim的EPDMS评估包含两阶段pseudo-closed-loop：
1. 第一阶段：原始场景评测
2. 第二阶段：reactive交通agent的合成场景评测

agent需对每个token输出一个Trajectory，包含8个SE2 poses。

## 四、部署步骤（在远程GPU服务器上）

### 4.1 安装环境

```bash
# 先安装navsim环境
cd navsim_0616/navsim
pip install -e .

# 再安装alpamayo环境（可能需要单独venv或合并到navsim的venv）
cd navsim_0616/alpamayo1.5
# 用uv或pip安装alpamayo依赖
uv venv --python 3.12
source .venv/bin/activate
uv sync --active
# 或手动pip install需要的包

# 关键：alpamayo需要 flash-attn, transformers>=4.57.1, torch>=2.8
# navsim需要 nuplan, pytorch_lightning, hydra-core
# 需要确保两边的torch版本兼容
```

### 4.2 下载NavSim数据

```bash
# 需要下载OpenScene数据集
# 参考 navsim/docs/install.md
# 设置环境变量：
export OPENSCENE_DATA_ROOT=/path/to/openscene
export NUPLAN_MAPS_ROOT=/path/to/nuplan_maps

# NavSim数据包含：
# - navsim_log_path: 场景日志 pickle文件
# - sensor_blobs_path: 相机图像和lidar点云
# - metric_cache_path: 预计算的PDM评分缓存
```

### 4.3 HuggingFace认证

Alpamayo模型需要HF访问权限：
```bash
huggingface-cli login
# 需要先在 https://huggingface.co/nvidia/Alpamayo-1.5-10B 申请访问
# 以及 https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
```

### 4.4 运行评测

方式A：使用NavSim的Hydra评测系统（推荐）
```bash
# 先缓存metric数据
python navsim/planning/script/run_metric_caching.py \
    train_test_split=warmup_two_stage

# 然后跑评测
python navsim/planning/script/run_pdm_score.py \
    agent=alpamayo_agent \
    train_test_split=warmup_two_stage
```

方式B：使用standalone脚本（调试用）
```bash
python run_alpamayo_on_navsim.py \
    --navsim_log_path /path/to/navsim_logs \
    --sensor_blobs_path /path/to/sensor_blobs \
    --model_path nvidia/Alpamayo-1.5-10B \
    --token <some_token>
```

### 4.5 创建提交文件

```bash
python navsim/planning/script/run_create_submission_pickle.py \
    agent=alpamayo_agent \
    train_test_split=warmup_two_stage \
    team_name=YourTeam \
    authors=YourName \
    email=your@email \
    institution=YourUni \
    country=China
```

## 五、已知限制和改进方向

1. **相机不对齐**：NavSim的前三相机视角和Alpamayo的训练相机视角(FoV、畸变、位置)不一致，模型可能产生偏差。如果Alpamayo支持"灵活相机数量"特性，可以尝试只用cam_f0单相机推理。

2. **时间帧缺失**：当前方案重复4次同一帧，丢失运动信息。改进方案：利用NavSim的4个历史帧，每相机取4帧@2Hz的图像，虽然频率低于Alpamayo的10Hz但至少有运动信息。

3. **坐标系差异**：NavSim用后轴SE2，Alpamayo用车辆中心3D坐标。平面上的差异很小（后轴偏移~1.3m在纵向），但heading定义一致。如果需要精确对齐，需知道具体车辆参数做偏移校正。

4. **输出分辨率**：Alpamayo的10Hz高分辨率轨迹被降到0.5Hz，丢失了很多细节。NavSim的PDM评分在0.5Hz上进行，但中间步的轨迹形状也会影响LQR跟踪模拟。可以考虑保持10Hz输出然后让NavSim的InterpolatedTrajectory做插值。

5. **VRAM需求**：Alpamayo单sample推理需要~24GB，多sample需要40-60GB。NavSim评测可能有上百个场景，需要确保GPU内存够用或做batch=1逐场景推理。

6. **多轨迹采样**：Alpamayo支持num_traj_samples>1产生多条轨迹，NavSim只取一条。可以在多条中选minADE最小的提交，但NavSim的PDM评分只看单条轨迹。
