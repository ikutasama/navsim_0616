# 服务器部署指南：在navsim中测试alpamayo1.5

在远程GPU服务器上操作。容器名 alpa15_rl_0612，venv /opt/venv/a1_5_venv。

## Phase 0: 拉取最新代码

```bash
cd /data/mnt_m181/z59900495/workspace
git clone https://github.com/ikutasama/navsim_0616.git
cd navsim_0616
```

## Phase 1: 环境准备

已经装好navsim(2.0.0)的a1_5_venv里，需要补装alpamayo1.5。

### 1.1 检查当前环境

```bash
source /opt/venv/a1_5_venv/bin/activate
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python3 -c "import transformers; print('transformers:', transformers.__version__)"
python3 -c "import flash_attn; print('flash-attn:', flash_attn.__version__)"
python3 -c "import navsim; print('navsim installed')"
python3 -c "import hydra; print('hydra:', hydra.__version__)"
python3 -c "import nuplan; print('nuplan installed')"
```

### 1.2 安装alpamayo1.5

```bash
cd /data/mnt_m181/z59900495/workspace/navsim_0616/alpamayo1.5
pip install -e .
# 如果physical-ai-av下载有问题（公司代理），可以先跳过：
# pip install -e . --no-deps && pip install accelerate einops hydra-core scipy av pandas pillow matplotlib seaborn torch==2.8.0 torchvision transformers==4.57.1
```

**关键：hydra-core版本冲突**
- navsim要求 ==1.2.0
- alpamayo要求 >=1.3.2
- 先装alpamayo需要的>=1.3.2，测试navsim是否兼容。如果navsim报错，需要手动处理。

```bash
# 检查hydra版本
pip show hydra-core
# 如果是1.2.0（navsim装的），升级：
pip install hydra-core>=1.3.2
# 再测试navsim是否还能用：
python3 -c "from navsim.common.dataloader import SceneLoader; print('OK')"
```

### 1.3 安装NavSim需要的额外包

```bash
# nuplan-devkit可能已在navsim pip install时装好了，确认：
pip show nuplan-devkit
# 如果没有：
pip install git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2

# pytorch-lightning
pip install pytorch-lightning==2.2.1
```

### 1.4 HF认证

```bash
huggingface-cli login
# 用你的HF token登录
# 确保已申请访问：
#   https://huggingface.co/nvidia/Alpamayo-1.5-10B
#   https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
```

## Phase 2: 数据准备

### 2.1 下载OpenScene数据

需要NavSim的OpenScene数据集（基于nuScenes）。

```bash
# 设置环境变量
export OPENSCENE_DATA_ROOT=/data/mnt_m181/z59900495/workspace/navsim_data/openscene
export NUPLAN_MAPS_ROOT=/data/mnt_m181/z59900495/workspace/navsim_data/nuplan_maps

# 参考navsim/docs/install.md下载方式
# NavSim v2的mini split很小，先下载mini用于测试
```

### 2.2 缓存metrics（可选，但需要跑PDM评分）

```bash
python navsim/navsim/planning/script/run_metric_caching.py \
    train_test_split=navmini \
    navsim_log_path=$OPENSCENE_DATA_ROOT/navsim_logs \
    sensor_blobs_path=$OPENSCENE_DATA_ROOT/sensor_blobs \
    metric_cache_path=$OPENSCENE_DATA_ROOT/metric_cache
```

## Phase 3: 分步测试

用test_alpamayo_navsim.py脚本分步验证：

```bash
source /opt/venv/a1_5_venv/bin/activate
cd /data/mnt_m181/z59900495/workspace/navsim_0616

# Step 1: 验证所有import
python test_alpamayo_navsim.py \
    --navsim_log_path $OPENSCENE_DATA_ROOT/navsim_logs \
    --sensor_blobs_path $OPENSCENE_DATA_ROOT/sensor_blobs \
    --split mini

# 如果Step 1失败，修好依赖再继续。
# 如果Step 1-2通过但Step 3失败，修好agent代码。
# 如果Step 1-3通过，继续Step 4（需要GPU）。

# Step 4: 实际推理（跳过前面已通过的步骤）
python test_alpamayo_navsim.py \
    --navsim_log_path $OPENSCENE_DATA_ROOT/navsim_logs \
    --sensor_blobs_path $OPENSCENE_DATA_ROOT/sensor_blobs \
    --metric_cache_path $OPENSCENE_DATA_ROOT/metric_cache \
    --split mini \
    --skip-to-step 4
```

## Phase 4: 完整评测（Hydra框架）

所有步骤通过后，用NavSim的Hydra评测框架跑完整评测：

```bash
# Warmup评测
python navsim/navsim/planning/script/run_pdm_score.py \
    agent=alpamayo_agent \
    train_test_split=warmup_two_stage \
    navsim_log_path=$OPENSCENE_DATA_ROOT/navsim_logs \
    sensor_blobs_path=$OPENSCENE_DATA_ROOT/sensor_blobs \
    synthetic_sensor_path=$OPENSCENE_DATA_ROOT/synthetic_sensor_blobs \
    metric_cache_path=$OPENSCENE_DATA_ROOT/metric_cache
```

## 常见问题

### Q: hydra-core版本冲突怎么办？
A: 先装>=1.3.2（alpamayo需要），再测navsim。navsim主要用hydra.utils.instantiate和@hydra.main装饰器，
1.3.x通常兼容1.2.x的API。如果navsim脚本报错，看具体错误——如果是hydra._internal的私有API变了，
可能需要小改navsim代码。

### Q: physical-ai-av安装失败（公司代理/SSL）？
A: 在navsim模式下不需要physical-ai-av！我们用navsim的SceneLoader加载数据，
不走alpamayo的PhysicalAIAVDatasetInterface。如果pip install失败，跳过它。
但alpamayo模型代码本身import了physical_ai_av，需要做一下monkey-patch或
在agent.initialize()之前设置一个mock。

### Q: 模型下载慢？
A: Alpamayo-1.5-10B有22GB权重。可以先在本地下载到磁盘，然后from_pretrained指向本地路径。

### Q: flash-attn编译失败？
A: 需要CUDA 12.x和nvcc。如果编译不了，用PyTorch的SDPA fallback：
设环境变量 ALPAMAYO_USE_SDPA=1 或在模型config中设 attn_implementation="sdpa"。
