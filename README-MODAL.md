# MuseTalk 1.5 Modal Starter 单任务

这次只准备代码。没有 Modal 登录、镜像构建、部署、GPU、推理、权重下载，也没有生成 MP4。用量和耗时里的实测格全部是空的。账号认证和 attestation 都还没有就绪。下文的 `submit`、`recover`、`modal app stop` 只在那些条件以后满足时执行，这次没有执行。

女性人物加男性音色只作为技术核对。这段测试片不是完整 EL01，也不是合格作品。

## 这份 ZIP 不是完整仓库

`/opt/cursor/artifacts/musetalk15-modal-starter-code.zip` 是适配层 overlay，不是一份能单独跑起来的 MuseTalk 源码树。压缩包里有 `modal_musetalk/`、`modal_app.py`、`README-MODAL.md`、两份 requirements、`tests/test_modal_adapter.py`，以及改过清理逻辑的 `scripts/inference.py`。里面没有 `musetalk/`、`configs/`、`app.py`、权重或媒体。只解开 ZIP、不检出本 fork 分支，不能跑推理，也不能把 ZIP 当成上游仓库。

CPU 测试和以后的提交都从本分支的 git 检出目录开始。ZIP 只用于把适配层拷回已有检出，或对照文件。

## 本地安装与复跑 CPU 测试

本次 20 项 CPU 测试使用的解释器是 CPython 3.12.3。Modal SDK 1.6.4 声明支持 Python 3.10–3.15。3.10、3.11 和 3.13 没有在这里复跑。云镜像写的是 Python 3.10，与本地 venv 不是同一个环境，而且镜像没有构建。

系统需要 `ffmpeg` 和 `ffprobe`。本次测试看到的是 ffmpeg 6.1.1。别的版本没有复跑。Debian/Ubuntu 上这两个命令来自同一个 `ffmpeg` 包。

本地提交器的第三方依赖只有 `modal==1.6.4`，写在 `requirements-local-client.txt`。代码没有 `import yaml`。`huggingface_hub` 和 `gdown` 只在远程的 `prepare_weights()` 里导入，本地 CLI 不导入它们。20 项 CPU 测试只用标准库；不安装 Modal 也能复跑。安装 `modal==1.6.4` 是为了以后门闩打开时能够 `import modal`，这次没有做认证。

不要为了复跑测试去安装 `requirements.txt` 或 `requirements-modal-inference.txt`。前者是上游训练/完整环境清单，含 tensorflow。后者是尚未构建的 GPU 镜像用的推理包，含 torch 生态，不是本地提交器。

从空目录检出此 fork 分支：

```bash
git clone https://github.com/xiaoge541/MuseTalk.git
cd MuseTalk
git fetch origin cursor/musetalk15-modal-starter-78f8
git switch cursor/musetalk15-modal-starter-78f8
git rev-parse HEAD
```

`git rev-parse HEAD` 应等于本 README 所在提交。然后在这个检出目录里：

```bash
ffmpeg -version
ffprobe -version
# 若上面两个命令不存在（Debian/Ubuntu）：
# sudo apt-get update && sudo apt-get install -y ffmpeg

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local-client.txt
python -m unittest tests.test_modal_adapter -v
```

`python -m unittest` 期望 20 项通过。它不下载权重，不调用 Modal API。

云镜像（未构建）和 `requirements-modal-inference.txt` 的差别：

1. 基础是 `debian_slim`、Python 3.10，再 `apt` 安装 `ffmpeg`、`git`、`libgl1`、`libglib2.0-0`、`wget`、`ca-certificates`。
2. 先从 `https://download.pytorch.org/whl/cu118` 安装 `torch==2.0.1`、`torchvision==0.15.2`、`torchaudio==2.0.2`。这三行不在 requirements 文件里。
3. 然后才 `pip install -r requirements-modal-inference.txt`（diffusers、transformers、face-alignment==1.3.4、modal==1.6.4 等）。
4. 再执行镜像里的 `run_commands`：`openmim==0.3.9`、`mmcv==2.0.1`、`mmdet==3.1.0`、`mmpose==1.1.0`、`mmengine==0.10.5`。这五项也不在那个 requirements 文件里，因为 mmcv 要在 CUDA 版 torch 之后装。
5. 最后把 git 检出目录挂进容器，不是把 ZIP 当成仓库。

## 入口结论

正式入口是 `python -m scripts.inference`，版本参数 `--version v15`。

`musetalk/utils/utils.py` 的 `get_file_type` 把 `.png` 判成 `image`。`scripts/inference.py` 对单张图走 `input_img_list = [video_path]`，帧率用 `--fps`（这里固定 25）。所以肖像 PNG 可以直接作为 `video_path`，不必先包成视频。

`app.py` 的 Gradio `inference()` 不是这条路径。非视频输入被当成图片目录做 `glob`，单张 PNG 不会按单图推理。本适配不调用 `app.py`。

`input_still_container_NOT_DYNAMIC.mp4` 是同一张静图的 25 fps、85 帧、3.4 秒、H264/yuv420p、无声容器，不是动态底片。上游并没有拒绝这种文件。`musetalk/utils/utils.py` 里 `.mp4` 属于 `get_file_type` 的 video 列表；`scripts/inference.py` 的 video 分支会用 ffmpeg 抽出全部帧，再用 `get_video_fps` 取帧率。把它写进上游 yaml 的 `video_path` 会走这条视频路径。

本适配器的 `--portrait` 只收 `.png`。`validate_portrait_png` 看到非 PNG 就拒绝。远程任务也把字节写成 `portrait.png`，yaml 的 `video_path` 指向这张图，从而走 image 分支。这是适配器的选择，不是上游不支持静图视频。

单图清理原先无条件 `shutil.rmtree(save_dir_full)`。这个变量只在视频分支里赋值，异常又被函数里的 `except` 吃掉。fork 上的 `scripts/inference.py` 改成仅在该目录存在时删除。这个改动没有在 GPU 上跑过。

音频方面，`AudioProcessor.get_audio_feature` 使用 `librosa.load(wav_path, sr=16000)`。librosa 默认混合为单声道并重采样到 16 kHz。`scripts/inference.py` 自己不调用 `ensure_wav`。显式 ffmpeg 仍要做，并且写到新文件，原 WAV 不改。推理配置指向这份 16 kHz 单声道文件；返回的 MP4 再用原 WAV 做音轨替换，画面复制，`-shortest`。原文件保持不变。这条替换没有在 MuseTalk 成片上验证。

按音频时长估算的推理帧数是 `floor(3.377708 * 25) = 84`。这是 `audio_processor.py` 里 `floor((样本数 / 16000) * fps)` 的时长替身。重采样后的样本数没有实测，所以 84 不是已确认帧数。静图容器的 85 帧不作为输出长度。

## 输入与不改原文件的 ffmpeg

占位路径：

- 肖像：`PORTRAIT.png`，PNG，RGB，1086x1448
- 音频：`AUDIO.wav`，WAV，PCM16，48 kHz，立体声，3.377708 秒
- 输出：`./out/result.mp4`

核对原文件：

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,pix_fmt -of default=nw=1 PORTRAIT.png
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -show_entries format=duration -of default=nw=1 AUDIO.wav
```

生成 16 kHz 单声道副本：

```bash
ffmpeg -y -i AUDIO.wav -ar 16000 -ac 1 -c:a pcm_s16le AUDIO_16k_mono.wav
```

若只是想从 PNG 重建那个无声静图容器（本任务不把它送进推理）：

```bash
ffmpeg -y -loop 1 -i PORTRAIT.png -t 3.4 -r 25 -c:v libx264 -pix_fmt yuv420p -an STILL_CONTAINER_NOT_DYNAMIC.mp4
```

适配器的提交命令会拒绝把 `.mp4` 放在 `--portrait` 上，也会拒绝把 16 kHz 单声道当作原始音频。直接调用上游 `scripts.inference` 时，`.mp4` 仍然可以写进 `video_path`。

## 花费门闩

门闩默认关闭。代码不读取账户余额来放行，不写 attestation，也不设置 Modal 的 workspace budget。

操作者描述的账户状态（本仓库没有登录 Modal 去核对）：已登录 Starter；可用额度里有 $1，营销赠额 $30 中其余 $29 需要付款方式；本月已用 $0；页面上的 Usage limit 为 $1，由 Modal 设置；没有看到可单独保存再读回的 post-credit Spend limit 字段。定价文档里的默认 $0 不是已保存的现金上限。GPU 是否真能在零新增现金下调度，尚未核实。2026-11-17 01:41 Asia/Shanghai 的窗口已经过期，这次不是新的试验窗口。

这些事实全部不能打开门闩。

认证和 attestation 现在都没有。下面的环境变量和 `submit` 不是这次要执行的命令。以后若经授权的操作者核对过账单页，并且账号持有人决定允许登录和这一次费用，再把核对结果写成仓库外的 JSON：

```bash
export MUSETALK_SPEND_ATTESTATION=/path/to/attestation.json
```

文件必须同时满足：`schema` 为 `musetalk-modal-spend-gate/v1`；`human_verified_at` 晚于 `2026-11-17T01:41:00+08:00`；`explicit_saved_cash_spend_cap_usd` 为数字 `0`；`cash_cap_is_saved_effective` 与 `cash_cap_is_not_docs_default_formula` 都为 true；`gpu_eligibility_verified` 为 true；`payment_requirement` 为非空说明；`single_task_only` 与 `expired_window_is_not_authorization` 为 true；`statement` 与 `modal_musetalk/gate.py` 里的 `REQUIRED_STATEMENT` 逐字相同。缺任何一项，程序在 `import modal` 之前退出。直接执行 `modal run` 或 `modal deploy` 也会在导入时退出。Modal 容器里 SDK 会设置 `MODAL_IS_REMOTE=1`，应用模块在容器内被再次导入时不再重复读 attestation。这不是授权。命令行提交和恢复都不看这个变量。本地如果手工导出它再调用 Modal CLI，等于绕过导入护栏；那不是支持的入口，这次也没有这样跑。容器是否在导入前设置该变量，没有实测。

## 单次提交与恢复

调用顺序是：门闩通过后，若名为 `musetalk15-starter-once` 的已部署函数不存在，才 `deploy` 一次；然后 `Function.spawn` 一次；立刻把 `object_id` fsync 到 `.musetalk_modal_calls/<job-id>.json`；然后才 `FunctionCall.get`。同一个 job id 再提交会被拒绝。

超时抛出 `TimeoutError` 时只把状态写成 `waiting_recovery`，不再次 spawn。恢复命令只调用 `modal.FunctionCall.from_id`。

SDK 1.6.4 里，应用异常在 `retries=0` 时不由我们重试。容器崩溃（例如 OOM、抢占）仍可能被 Modal 重新调度。那不是客户端的第二次 `spawn`，但可能继续消耗 GPU 时间。这一点没有实测。

下面三条都还没有执行。前两条还要求 attestation 已经写好，并且 Modal 认证已经由账号持有人决定做过。现在这两件事都没有。

```bash
python -m modal_musetalk submit \
  --portrait PORTRAIT.png \
  --audio AUDIO.wav \
  --output ./out/result.mp4 \
  --job-id job-001

python -m modal_musetalk recover \
  --call-id CALL_ID \
  --output ./out/result.mp4
```

MP4 已经下载并且用下面的 ffprobe / 全解码核对之后，用 Modal SDK 1.6.4 的 CLI 停止这个已部署应用。该命令会永久停止应用并终止仍在跑的容器。文件还没落地时不要执行。这次没有执行：

```bash
modal app stop musetalk15-starter-once --yes
```

`--yes` 跳过 CLI 的确认提问。不带 `--yes` 时，`modal app stop` 会先询问。应用名与 `modal_musetalk/spec.py` 里的 `APP_NAME` 相同。`scaledown_window=2` 只缩短空闲容器的保留时间；停止应用是上面这条命令。

`--wait-s` 默认 1800。函数超时 3600 秒，容器启动超时 600 秒。这两个数是配置，不是测量。

远程函数把字节写成通用文件名 `portrait.png`、`audio_original.wav`、`audio_16k_mono.wav`，再执行：

```bash
python -m scripts.inference \
  --inference_config <job.yaml> \
  --result_dir <inference_results> \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config models/musetalkV15/musetalk.json \
  --whisper_dir models/whisper \
  --vae_type sd-vae \
  --version v15 \
  --use_float16 \
  --fps 25 \
  --batch_size 8 \
  --extra_margin 10 \
  --parsing_mode jaw \
  --left_cheek_width 90 \
  --right_cheek_width 90 \
  --gpu_id 0 \
  --output_vid_name result.mp4
```

成片路径是 `<result_dir>/v15/result.mp4`，随后 remux 成 `result.mp4` 返回字节。本地写出 `./out/result.mp4` 和 `./out/result.usage.json`。

下载完成之前不要执行 `modal app stop`。下载并核对之后的精确命令写在上一节，这次没有执行。本代码不会自动 stop。

## GPU 释放与权重缓存成本

函数参数：`gpu="T4"`，`scaledown_window=2`，`max_containers=1`，`retries=0`。没有 `min_containers`，没有 web endpoint，没有 Secret，没有 Volume。

Modal 文档写明空闲窗口默认 60 秒且会计费，`scaledown_window` 可设为 2 到 1200 秒。这里取文档下限 2 秒。容器可能更早退出，所以空闲尾部在 `(0, 2]` 秒，精确值未知。

权重放在容器临时盘，不建 cache volume。每次冷容器都会重新下载。下载发生在 GPU 函数内部，因此下载时间按 GPU 计价。这是为了保持一次 spawn，而不是再开一个 CPU 下载调用。Volume 单价仍列在下面，但本代码的 volume 费用是 0，因为没有创建。

已知 Hugging Face 文件大小（2026-11-17 tree API）：4,292,943,922 字节，约 3.998 GiB。这只包括：

| 文件 | 修订 | 字节 |
| --- | --- | --- |
| `TMElyralab/MuseTalk` `musetalkV15/unet.pth` 与 `musetalk.json` | `2bcb936e2fddb4d86db4c62fd45b387d0c061571` | 3,400,074,924 + 748 |
| `stabilityai/sd-vae-ft-mse` `diffusion_pytorch_model.bin` 与 `config.json` | `31f26fdeee1355a5c34592e401dd41e45d25a493` | 334,707,217 + 547 |
| `openai/whisper-tiny` `pytorch_model.bin`、`preprocessor_config.json`、`config.json` | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | 151,095,027 + 184,990 + 1,983 |
| `yzd-v/DWPose` `dw-ll_ucoco_384.pth` | `1a7144101628d69ee7a3768d1ee3a094070dc388` | 406,878,486 |

另外还有未测大小、也没有 git 修订的文件：gdown id `154JgKpzCPW82qINcVieuPH3fZ2e0P812` 的 `79999_iter.pth`，以及 `https://download.pytorch.org/models/resnet18-5c106cde.pth`。推理进程还可能再取 face-alignment 1.3.4 的 `s3fd-619a316812.pth` 与 `2DFAN4-cd938726ad.zip`。mmpose 配置里的 CSPNeXt `init_cfg` 指向一条 OpenMMLab URL；`init_model` 会不会再下载它，没有执行过。

`ByteDance/LatentSync` 修订 `405eda8eab9f65c1a6e0c292a5dee5a08089e2ae` 的 `latentsync_syncnet.pt` 不在推理下载计划里。`train.py` 会加载它，`scripts.inference` 不会。`app.py` 的启动检查会要求它存在。Hub 卡字段是 `openrail++`，这里不把它当成商用授权。

代码基线是 fork `main` 的 `0a89dec45a0192b824e3cf4daf96c239440c5ed8`，再加上本分支对 `scripts/inference.py` 的清理修复。

## 显存证据与价格情景

这些不是本仓库的测量，也不是报价。

官方推理显存只有 README 里的两处说法：

- Windows 上的 NVIDIA GeForce RTX 3050 Ti Laptop，4GB VRAM，fp16，约 8 秒视频大约 5 分钟。分辨率没有写成 1086x1448。
- Tesla V100 上 30 fps 以上。README 没有写这张 V100 是 16GB 还是 32GB。

训练表是 8 张 H20：stage 1 大约每卡 32GB 到 74GB，stage 2 大约 54GB 到 85GB。那是训练，不是这次推理。

1086x1448、fp16、batch 8 在 T4 上会不会超出显存：未知。OOM 后不要用同一个 job id 再提交。

2026-11-17 从 https://modal.com/pricing 读到的列表价（每秒）：

| 定价页名称 | 代码里的 GPU 字符串 | USD / 秒 | 显存 |
| --- | --- | --- | --- |
| Nvidia T4 | `T4` | 0.000164 | NVIDIA 产品规格 16GB；不是这次任务的测量 |
| Nvidia L4 | `L4` | 0.000222 | NVIDIA 产品规格 24GB；不是这次任务的测量 |
| Nvidia A10 | `A10G`（SDK 枚举 `GPU_TYPE_A10G`） | 0.000306 | 定价页写 A10；SDK 枚举是 A10G。是否同一 SKU 以 Modal 当时接受的字符串为准，未部署验证 |

同页还有：CPU `0.0000131` USD / 物理核 / 秒（最少 0.125 核）；内存 `0.00000222` USD / GiB / 秒（最少 128 MiB）；出口 `0.04` USD / GiB；Volume `0.09` USD / GiB / 月。Starter 含每月 1 TiB 出口。区域倍率 1.15 到 1.75 只有在选择区域时出现，本函数不设置区域。非抢占是 3 倍，本函数不开启。

费用公式，其中每一项秒数都未知：

```text
gpu_seconds = cold_start_s + weight_download_s + inference_subprocess_s
              + ffmpeg_audio_s + ffmpeg_remux_s + idle_tail_s
gpu_usd = gpu_seconds * 0.000164          # 仅当实际调度的是 T4
cpu_usd = cpu_core_seconds * 0.0000131
memory_usd = gib_seconds * 0.00000222
egress_usd = output_gib * 0.04
volume_usd = 0                             # 本代码不创建 volume
```

`cold_start_s`、`weight_download_s`、`inference_subprocess_s`、`idle_tail_s` 都不能从这次代码准备填数。函数内部即使在真实 GPU 上运行，也看不到平台在进程启动前的冷启动，也看不到返回之后的空闲尾部。`*.usage.json` 里的 `modal_dashboard_gpu_billed_s` 和 `modal_dashboard_usd` 保持 null，只能由真实运行后的账单手工填写。

## 许可

- 本仓库 `LICENSE` 是 MIT。该文本允许在保留声明的前提下使用、复制、修改和出售这份代码。
- README 的 Disclaimer 第 2 条写着训练模型可用于任何目的，包括商业。这是 README 里的一句话，不是另一份权重许可合同，也不是法律结论。
- Hugging Face 卡字段（2026-11-17 models API）：MuseTalk `mit`，sd-vae-ft-mse `mit`，whisper-tiny `apache-2.0`，DWPose `apache-2.0`，LatentSync `openrail++`。
- MuseTalk 的 `LICENSE` 把 whisper 代码引为 MIT，Hub 卡写的是 `apache-2.0`。两处都记录，不合并成一个结论。
- 训练数据包含 HDTF 和私有数据。MIT 代码许可没有把这些数据权利授出去。
- `79999_iter.pth`、resnet18 权重、S3FD / 2DFAN 权重的许可在本仓库里没有单独文本。未知。
- 不用某个 “NOASSERTION” 字段来主张商用授权。

## 成片核对

真实 MP4 还不存在。命令留在这里，等有文件再跑：

```bash
ffprobe -v error -show_format -show_streams ./out/result.mp4
ffmpeg -v error -i ./out/result.mp4 -f null -
```

CPU 测试只用纯色图和正弦波跑过同样的 ffprobe / 全解码命令。那不是 MuseTalk 成片。

## 已执行与未测

已在 CPU 上执行，`python3 -m unittest tests.test_modal_adapter -v`，20 项通过：

- 门闩默认关闭；过期窗口、文档默认 $0、未核实 GPU、只有额度数字，都不能打开。
- 一份只存在于测试临时目录的完整 attestation 能让判定函数返回打开。仓库里没有这份文件，也没有因此调用 Modal。
- mock 客户端：spawn 一次，先落盘 call id，再 get；超时后不第二次 spawn；恢复只走 `from_id`。
- 合成灰图 1086x1448 与合成 48 kHz 立体声 WAV 上的 ffmpeg 16 kHz 单声道、remux、ffprobe、全解码。原文件哈希不变。推理子进程被替换，没有加载 MuseTalk。
- 静态阅读 `scripts/inference.py`、`app.py`、`audio_processor.py` 和权重计划。
- 关闭门闩时，`python -m modal_musetalk submit` 与 `import modal_app` 在导入 Modal 前退出。

未测清单在 `modal_musetalk/untested.py`。至少包括：Modal 认证、镜像构建、deploy、GPU、权重下载、MuseTalk 推理、模型 MP4、冷启动/下载/推理/空闲秒数、账单金额、T4 对 1086x1448 是否够用、mmcv/mmpose/face-alignment 是否能装上。

## 阻塞与下一步

唯一关键阻塞：花费门闩关闭。还没有一份写明有效 $0 现金支出上限、并且 GPU 资格已经核对过的 attestation。代码不会把 $1 的 Usage limit 或营销赠额当成放行条件，也不会去登录或改账户限额。

账单页上的额度、付款要求和现金上限，由经授权的操作者核查。是否登录、是否改预算，由账号持有人决定。

1. 经授权的操作者核对可用额度、付款方式要求，以及是否存在可回读的、不是文档默认公式的 $0 现金上限，并核对 GPU 能否在零新增现金下调度。
2. 只有这些条件成立，且账号持有人决定允许这一次费用时，才在仓库外写 attestation，并执行上面的 `submit`。那一次会构建镜像并可能产生费用。现在不要执行。
3. 若等待超时，只用保存的 call id 执行 `recover`，不要再次 `submit`。
4. 拿到 MP4 后跑 ffprobe 和全解码，再把 Modal 仪表上的实际秒数和金额填进 usage 文件。然后执行 `modal app stop musetalk15-starter-once --yes`。

这一步之前，不要把任何输出当成 MuseTalk 成片或实际用量。
