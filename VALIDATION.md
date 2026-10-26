# 验证记录（2026-10-26）

## 补充环境依赖冲突修复

- 用户 AutoDL V100S 日志显示：成功复用 torch 2.5.1+cu124 并补装库，但继承的 httpcore 1.0.7 与补充环境 h11 0.16.0 冲突；旧版仅运行 smoke test，错误地输出 ready。
- 已为复用和补充路径添加 pip check，记录 overlay/pip-check.log。依赖冲突阻止 ready，并停止自动回退重新下载框架。
- 32 项测试：Windows 本地 31 项通过，1 项 POSIX 专用测试跳过。新增依赖冲突回归测试。
- 未远程登录用户 AutoDL 实例，尚未替用户实际修复 h11/httpcore；提供的修复命令需要在其补充环境执行并再次检查。

## 预装环境复用更新

- 新增默认优先复用、缺包隔离补充、`--fresh` 强制新建模式。
- 更新后 29 项测试：Windows 本地 28 项通过、1 项 POSIX 专用测试跳过。Windows 沙箱对 detached 子进程启动存在限制，最终在沙箱外运行同一套测试通过；真实通知和关机仍是模拟。
- 新增测试覆盖完整复用时不访问包索引、不安装包；缺包计划只读；补充失败不发布 ready 指针；GPU 运行失败拒绝复用；补充环境只请求缺失依赖并锁住已有 torch 版本。
- 本机真实只读检查识别出已有 torch 2.5.1+cu124、torchvision 0.20.1+cu124、torchaudio 2.5.1+cu124，并通过 torch CUDA 运算。缺少 tensorboard、timm、opencv-python-headless，正确返回 supplement 计划，没有执行安装。
- 此次变更未在真实 AutoDL 实例执行缺包安装；该部分自动测试使用模拟 pip 和环境创建。

## 已完成

- 本机 Python 3.12.4：21 项自动测试，其中 20 项通过，1 项 POSIX 文件锁测试因 Windows 跳过。
- GitHub Actions Ubuntu / Python 3.10 与 3.12 的首次测试矩阵均成功，涵盖 POSIX 文件锁和后台进程测试。实际 AutoDL 平台行为仍另行验证。
- 训练进程成功、失败、超时、缺少产物、后台脱离启动和日志持久化，使用真实短 Python 子进程测试。
- 通知请求构造、失败重试上限、错误中密钥不泄露、签名计算，用模拟 HTTP 测试。
- 自动关机按次开启、通知失败仍按策略关机、取消标记生效、识别 Slurm 时拒绝关机，用模拟关机测试。
- 环境规划测试覆盖 CUDA 驱动上限、Blackwell 最低 CUDA、旧 GPU 排除 CUDA13、无 GPU 不静默安装 CPU、稳定版本数值排序、只读规划不创建环境。
- 实际硬件探测识别 Windows 11、i9 对应 32 个逻辑 CPU、RTX 4060 Laptop、驱动 560.94、驱动 CUDA 上限 12.6、计算能力 8.9。
- 实际访问官方 PyTorch wheel 索引并运行 pip index，生成候选配置。没有在本机执行 --apply，没有安装或升级任何深度学习包。

## 尚待实机验证

- AutoDL Linux 实例上完整 --apply 安装与 GPU 运算测试。
- AutoDL 容器特有的后台运行与信号/进程组行为（通用 Linux 单元测试已在 CI 验证）。
- 使用自己的 SCT SendKey 后，实际微信投递。
- AutoDL /usr/bin/shutdown 的真实关机与控制台状态确认。

代码具备上述流程并不等于已在租赁实例实测通过。尚未购买或连接 AutoDL 实例；没有调用真实关机、没有发送真实通知。

仓库已发布：https://github.com/TTJTR/Smart_HPC 。第一次 CI 运行：https://github.com/TTJTR/Smart_HPC/actions/runs/40295638526 。该记录不代表完整深度学习环境已在 AutoDL 安装成功。
