# 验证记录（2026-10-26）

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
