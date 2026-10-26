# Smart HPC / AutoDL Task Guard

新服务器 `git clone` 后即可使用的独立工具：**一键准备深度学习环境 + 后台跑任务 + 微信通知 + 可选自动关机**。守护程序只用 Python 标准库，不需要 AutoDL 开发者 Token，也不是 Codex 插件。

```bash
git clone https://github.com/TTJTR/Smart_HPC.git
cd Smart_HPC
python bootstrap.py --apply
```

仓库若设置为私有，需要先配置自己的 GitHub 读取凭据。不要把 Token 写进克隆 URL。

## 一、自动准备深度学习环境

需要已有 Python 3.10+。建议在 AutoDL 选择含现代 Python 的官方 PyTorch/Miniconda 镜像。

```bash
# 只检测并输出建议，不安装
python bootstrap.py

# 一键检测、创建隔离环境、安装、做真实运算检查
python bootstrap.py --apply

# 同时准备视觉与 LLM 工具链
python bootstrap.py --apply --profile vision --profile llm

# 明确需要 CPU 环境时
python bootstrap.py --apply --device cpu
```

运行逻辑：

1. 识别操作系统、架构、Python、CPU 数、GPU 型号、计算能力、NVIDIA 驱动和驱动支持的 CUDA 上限；记录 nvcc 但不把本地 Toolkit 版本作为唯一标准。
2. 动态读取 PyTorch 官方稳定 wheel 渠道，排除高于驱动 CUDA 上限的渠道，并对 Blackwell / 旧 GPU 做保守架构筛选。不使用 nightly。
3. 使用当前 Python 的 pip 检查可用 wheel，按稳定版 torch 版本排序；配套 torchvision、torchaudio 交给依赖解析器选择。
4. 仅在 `--apply` 时创建新的 venv，不修改系统 Python、驱动、CUDA Toolkit 或 HPC 共享环境。
5. 安装所选用途库，并锁住 torch 三件套版本，防止附加库悄悄换掉已选核心组合。
6. 执行 pip check、所有选定库的导入、GPU 矩阵运算/反向传播、卷积及 torchvision NMS 检查。没有 GPU 时不会悄悄改装 CPU 版。
7. 保存 plan、解析报告、测试日志、requirements.lock.txt；只有检查通过才更新 latest.json。失败保留诊断目录并尝试下一候选，默认最多 2 次。

默认目录 `~/autodl-tmp/task-guard-envs`，可用 `--destination` 指向持久数据盘。完成后使用程序输出的绝对路径激活，例如：

```bash
source /root/autodl-tmp/task-guard-envs/setup_日期/1_cuXXX/bin/activate
```

### 安装哪些库

| 组件 | 库 |
|---|---|
| PyTorch | torch、torchvision、torchaudio |
| 通用计算 | numpy、scipy、pandas、scikit-learn、matplotlib、Pillow、tqdm、PyYAML |
| 工具 | tensorboard、psutil、requests、ipykernel |
| 默认 vision | timm、opencv-python-headless |
| 可选 llm | transformers、accelerate、datasets、peft、safetensors、sentencepiece、huggingface-hub |

“最新”指运行时在当前 Python、系统和保守驱动约束下能解析到的稳定版本；不是承诺任意硬件都能运行全球最新版。旧 Python、旧 GPU、旧 glibc、网络限制或无可用 wheel 都可能导致失败并给出诊断。默认只接受二进制 wheel，不自动启动长时间源码编译。不会自动安装 TensorFlow、FlashAttention、Mamba、xformers 等额外兼容矩阵不同的组件。

重现既有论文实验时应使用已验证的版本锁文件，不应每轮都升级。`requirements.lock.txt` 主要用于相同 OS/架构/Python 平台，不保证跨平台通用。

若识别到 Slurm，`--apply` 必须在已分配的计算作业内执行，不在登录节点批量安装或测试 GPU。AutoDL 新实例本身无需 Slurm。

## 二、配置微信通知

首选 **Server酱 Turbo**，在 <https://sct.ftqq.com/> 登录、按服务方说明绑定微信接收通道，取得 SCT 开头的 SendKey。服务费用和推送额度以服务方当前规则为准。

```bash
python guard.py init
python guard.py doctor
python guard.py test-notify
```

`init` 隐藏输入 SendKey，保存在仓库外 `~/.config/autodl-task-guard/config.json`，权限 600。不要把真实 SendKey、Webhook、SSH 密码提交到 Git。也可通过 `SERVERCHAN_SENDKEY` 环境变量提供。

首版微信通道指 Server酱 Turbo 的 SCT Key，不是 Server酱³或个人微信机器人。保留可选飞书支持：`python guard.py init --provider feishu`（用于新配置文件）。

`doctor` 不发送消息；`test-notify` 会明确发送一条配置测试消息，不关机。真实微信收到消息与实际 AutoDL 关机需要在配置完成后自行或授权实机验证，自动测试不冒充实机验证。

## 三、登记并后台运行训练任务

在训练项目目录执行，使用守护工具的绝对路径：

```bash
python /root/autodl-tmp/autodl-task-guard/guard.py start \
  --name "DINO-CIFAR10分类头" \
  --shutdown \
  --grace 60 \
  --timeout 14400 \
  --success-file results/metrics.json \
  -- python train.py --epochs 50
```

- `start` 脱离 SSH 会话运行，关闭终端后任务继续。
- `--shutdown` 按次开启；成功、非零退出、超时或训练阶段收到终止信号后都会尝试关机。未传时保留实例。
- 先保存日志和状态，再通知微信，然后倒计时，最后调用 AutoDL 文档中的 `/usr/bin/shutdown`。
- 通知最多 3 次尝试、单次网络超时 8 秒。通知失败仍执行已授权的关机，避免无限等待继续计费。
- 消息含任务名、ID、状态、退出码、耗时和结束时间，不自动上传命令参数或训练日志。
- `--success-file` 为可选附加检查：零退出码还必须存在结果文件。请使用本次新建的结果目录，避免误认旧文件；它不验证科研结果质量。
- `--timeout` 单位秒，0 表示不限时。超时会终止被管理的进程组，给程序保存/清理的机会后再强制终止；训练仍应定期保存 checkpoint。
- 关机通知说“准备请求关机”，不声称已确认停止计费；实际以 AutoDL 控制台为准。

多个连续步骤放在同一条前台任务链中：

```bash
python /path/to/guard.py start --name "训练及评估" --shutdown -- \
  bash -lc 'set -e; python train.py; python evaluate.py'
```

被包装的命令必须等待真正任务结束；内部不要用 `nohup`、`&` 或只执行 Slurm 提交。工具不会通过 GPU 利用率或日志关键词猜测完成，也无法识别被代码吞掉的异常。

## 四、状态、日志与撤销关机

```bash
python /path/to/guard.py status
tail -f ~/.local/state/autodl-task-guard/<任务ID>/task.log
python /path/to/guard.py cancel-shutdown <任务ID>
```

可提前或倒计时期间撤销；已调用关机命令后不保证可撤销。状态是磁盘记录，不是云端电源查询。守护进程被 SIGKILL、实例崩溃、断电或被平台回收时，不能保证发消息或关机，也可能留下 running 记录。

首版每个实例默认只管理一个任务，OS 文件锁防止重复启动。请把共同完成的步骤串联起来。未登记的其他任务不会被工具自动识别，自动关机应只用于专用训练实例。不要通过更改 `--state` 绕过锁并并发启用关机。检测到 Slurm/HPC 时拒绝实例关机，防止误关共享节点。

## 五、本地验证

```bash
python -m unittest discover -s tests -v
python guard.py start --name smoke --no-notify -- python -c 'print("hello")'
```

测试模拟通知与关机，不关闭测试电脑。bootstrap 默认只是只读规划；真正安装要显式传 `--apply`。真实模型的准确率和 CKKS 实验仍需另行验证。

## 官方依据

- AutoDL：<https://www.autodl.com/docs/save_money/>
- 微信推送：<https://sct.ftqq.com/docs/getting-started/sendkey/>
- PyTorch：<https://pytorch.org/get-started/locally/>
- NVIDIA 驱动与 CUDA：<https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html>
- 可选飞书：<https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot>

不包含云服务器租用、充值、账户自动开通权限或远程驱动升级。GitHub 地址确认并发布后，可在新实例克隆同一仓库使用。
