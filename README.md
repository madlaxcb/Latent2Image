# Latent to Image Decoder (Latent2Image)

[English](#english) | [简体中文](#简体中文)

---

## English

A standalone Windows GUI application to decode ComfyUI latent files (`.latent`, `.safetensors`, `.pt`) into viewable images using a VAE model. This tool is designed for users who want to quickly preview or recover images from latent files without launching the full ComfyUI environment.

### Key Features
- **Standalone Executable**: No Python environment required for end-users (when using the built EXE).
- **Drag & Drop Support**: 
    - Drag files onto the application icon to launch and process automatically.
    - Drag files directly into the running window for instant decoding.
- **Smart Environment Detection**: Automatically finds and uses the Python environment and VAE models from an existing ComfyUI installation.
- **VAE Caching**: Loads the VAE model once into memory for high-speed batch processing.
- **Hardware Acceleration**: Supports both CUDA (NVIDIA GPU) and CPU decoding.
- **Memory Optimization**: Optional "Low VRAM" mode to offload models when idle.
- **File Management**: Automatically move processed `.latent` files to a designated "Processed" directory.
- **Metadata Support**: Decodes metadata and handles various latent formats from ComfyUI.

### How to Use
1. **Initial Setup**:
   - Set the **ComfyUI Directory**: Point to your local ComfyUI folder to use its internal environment.
   - Select a **VAE Path**: Choose a VAE model (e.g., `vae-ft-mse-840000-ema-pruned.safetensors`).
2. **Decoding**:
   - Click **Select Latent Files & Run** to pick files.
   - **OR** Simply drag `.latent` files into the window.
3. **Settings**:
   - **Processed Dir**: (Optional) Set a directory. After decoding, the source `.latent` files will be moved here.
   - **Low VRAM Mode**: Enable if you experience out-of-memory issues.

### Requirements for Building
- Python 3.10+
- `torch`, `diffusers`, `safetensors`, `tkinterdnd2`
- Run `build_exe.bat` to generate a portable `.exe`.

---

## 简体中文

一个独立的 Windows GUI 应用程序，用于使用 VAE 模型将 ComfyUI 的 latent 文件（`.latent`, `.safetensors`, `.pt`）解码为可查看的图像。该工具专为希望在不启动完整 ComfyUI 环境的情况下快速预览或恢复 latent 图像的用户设计。

### 主要功能
- **独立可执行文件**: 终端用户无需安装 Python 环境（使用打包后的 EXE）。
- **拖放支持**:
    - 将文件拖到应用程序图标上即可自动启动并处理。
    - 将文件直接拖入正在运行的窗口进行即时解码。
- **智能环境检测**: 自动发现并使用现有 ComfyUI 安装中的 Python 环境和 VAE 模型。
- **VAE 缓存**: 将 VAE 模型加载到内存中一次，实现高速批量处理。
- **硬件加速**: 支持 CUDA (NVIDIA GPU) 和 CPU 解码。
- **内存优化**: 可选的“低显存模式”，在闲置时卸载模型。
- **文件管理**: 自动将处理过的 `.latent` 文件移动到指定的“已处理”目录。
- **元数据支持**: 解码元数据并处理来自 ComfyUI 的各种 latent 格式。

### 使用方法
1. **初始设置**:
   - 设置 **ComfyUI 目录**: 指向您的本地 ComfyUI 文件夹以使用其内部环境。
   - 选择 **VAE 路径**: 选择一个 VAE 模型文件（例如 `vae-ft-mse-840000-ema-pruned.safetensors`）。
2. **解码**:
   - 点击 **选择 Latent 文件并运行** 手动选择文件。
   - **或者** 直接将 `.latent` 文件拖入窗口。
3. **可选设置**:
   - **已处理目录**: (可选) 设置一个目录。解码成功后，源 `.latent` 文件将被移动到此处。
   - **低显存模式**: 如果遇到显存不足问题，请启用此项。

### 开发与构建
- 需要 Python 3.10+
- 依赖项：`torch`, `diffusers`, `safetensors`, `tkinterdnd2`
- 运行 `build_exe.bat` 生成便携式 `.exe`。

---

<img width="1052" height="1023" alt="图片" src="https://github.com/user-attachments/assets/16f5c2bb-e833-4e4f-87e2-54fa67a3e176" />


## License
MIT License. Created by [madlaxcb](https://github.com/madlaxcb).
