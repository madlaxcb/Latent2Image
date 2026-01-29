@echo off
echo Building Lightweight Launcher EXE...
rem Exclude heavy libraries that should be provided by ComfyUI environment
pyinstaller --noconfirm --onefile --windowed --name "Latent2Image" ^
    --add-data "latent_decoder.py;." ^
    --add-data "configs;configs" ^
    --collect-all tkinterdnd2 ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module diffusers ^
    --exclude-module safetensors ^
    --exclude-module transformers ^
    --exclude-module accelerate ^
    --exclude-module numpy ^
    --exclude-module PIL ^
    --exclude-module huggingface_hub ^
    app.py

echo Build complete. executable is in dist folder.
pause
