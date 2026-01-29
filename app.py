import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    print("Warning: tkinterdnd2 not found. Drag and drop will not work.")

import sys
import os
import shutil
import json
import threading
import subprocess
import re
from datetime import datetime

# NOTE: No heavy imports here (torch, safetensors, diffusers)
# They are only used in the worker script (latent_decoder.py)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Latent to Image Decoder (Launcher)")
        self.root.geometry("700x650")

        # Determine config paths
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            self.user_data_dir = os.path.dirname(sys.executable)
            self.resource_dir = sys._MEIPASS
        else:
            # Running as script
            self.user_data_dir = os.path.dirname(os.path.abspath(__file__))
            self.resource_dir = self.user_data_dir
            
        self.config_file = os.path.join(self.user_data_dir, "config.json")
        self.load_config()

        self.tooltip_window = None
        self.tooltip_job = None
        self.stop_requested = False
        self.current_process = None

        # Enable Drag and Drop
        if HAS_DND:
            try:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind('<<Drop>>', self.drop_event)
            except Exception as e:
                self.log(f"Drag and Drop init failed: {e}")

        # UI Elements
        self.create_widgets()
        
        # Auto-check environment if config is loaded
        if self.comfy_dir_var.get():
             self.root.after(500, self.check_comfy_env)

        # Check for drag-and-drop args
        if len(sys.argv) > 1:
            files = sys.argv[1:]
            # Filter for likely latent files
            self.dropped_files = [f for f in files if os.path.isfile(f)]
            if self.dropped_files:
                self.log(f"Detected {len(self.dropped_files)} file(s) from launch arguments.")
                # Delay start to allow UI to render
                self.root.after(1000, self.auto_run_dropped)

    def drop_event(self, event):
        files = self.root.tk.splitlist(event.data)
        valid_files = [f for f in files if os.path.isfile(f) and (f.endswith('.latent') or f.endswith('.safetensors') or f.endswith('.pt'))]
        
        if valid_files:
            self.log(f"Dropped {len(valid_files)} file(s).")
            # Run in separate thread or schedule
            self.root.after(100, lambda: self.run_process(valid_files))
        else:
            self.log("No valid latent files dropped.")

    def create_widgets(self):
        # Settings Frame
        settings_frame = tk.LabelFrame(self.root, text="Settings", padx=10, pady=10)
        settings_frame.pack(fill="x", padx=10, pady=5)

        # ComfyUI Directory
        comfy_frame = tk.Frame(settings_frame)
        comfy_frame.pack(fill="x", pady=2)
        tk.Label(comfy_frame, text="ComfyUI Dir:").pack(side="left")
        self.comfy_dir_var = tk.StringVar(value=self.config.get("comfy_dir", ""))
        tk.Entry(comfy_frame, textvariable=self.comfy_dir_var).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(comfy_frame, text="Browse", command=self.browse_comfy).pack(side="right")
        self.create_tooltip(comfy_frame, "Path to your ComfyUI installation folder.\nThe program will use the Python environment from there.")

        # VAE Selection
        vae_frame = tk.Frame(settings_frame)
        vae_frame.pack(fill="x", pady=2)
        tk.Label(vae_frame, text="VAE Path:").pack(side="left")
        self.vae_path_var = tk.StringVar(value=self.config.get("vae_path", ""))
        tk.Entry(vae_frame, textvariable=self.vae_path_var).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(vae_frame, text="Browse", command=self.browse_vae).pack(side="right")
        self.create_tooltip(vae_frame, "Path to the VAE model file (.safetensors, .pt, .ckpt).\nUsed to decode the latent representation into an image.")

        # Model Type (Scale Factor)
        model_frame = tk.Frame(settings_frame)
        model_frame.pack(fill="x", pady=2)
        tk.Label(model_frame, text="Model Type:").pack(side="left")
        self.model_type_var = tk.StringVar(value=self.config.get("model_type", "sd15"))
        self.scale_var = tk.DoubleVar(value=self.config.get("scale_factor", 0.18215))
        
        tk.Radiobutton(model_frame, text="SD 1.5 (Scale: 0.18215)", variable=self.model_type_var, value="sd15", command=self.update_scale_from_type).pack(side="left", padx=5)
        tk.Radiobutton(model_frame, text="SDXL (Scale: 0.13025)", variable=self.model_type_var, value="sdxl", command=self.update_scale_from_type).pack(side="left", padx=5)
        self.create_tooltip(model_frame, "Select the model type used to generate the latents.\nThis sets the default scale factor.")
        
        tk.Label(model_frame, text="Custom Scale:").pack(side="left", padx=5)
        entry_scale = tk.Entry(model_frame, textvariable=self.scale_var, width=10)
        entry_scale.pack(side="left")
        self.create_tooltip(entry_scale, "Manually override the scale factor.\nIf images are too saturated/blown out, try 1.0.\nSD1.5 default: 0.18215\nSDXL default: 0.13025")
        
        # Initial update of scale factor - ONLY if not loaded from config
        # If config has a custom value that differs from default, we should keep it.
        # But simpler: check if self.scale_var has a value. It does from init.
        # We only want to force update if the user explicitly clicks.
        # So we REMOVE the auto-call here.
        # self.update_scale_from_type() 
        
        # Device Selection
        device_frame = tk.Frame(settings_frame)
        device_frame.pack(fill="x", pady=2)
        tk.Label(device_frame, text="Device:").pack(side="left")
        self.device_var = tk.StringVar(value=self.config.get("device", "cuda"))
        tk.Radiobutton(device_frame, text="CUDA (GPU)", variable=self.device_var, value="cuda").pack(side="left")
        tk.Radiobutton(device_frame, text="CPU", variable=self.device_var, value="cpu").pack(side="left")
        self.create_tooltip(device_frame, "Select the computation device.\nCUDA is much faster but requires an NVIDIA GPU.")
        
        # Show CUDA Status (Dynamic check now)
        self.cuda_label = tk.Label(device_frame, text="[Status: Unknown]", fg="gray")
        self.cuda_label.pack(side="left", padx=5)
        self.create_tooltip(self.cuda_label, "Status of CUDA availability in the selected ComfyUI environment.")
        
        # Dependency Status
        self.dep_label = tk.Label(device_frame, text="", fg="gray")
        self.dep_label.pack(side="left", padx=5)
        
        self.btn_install_deps = tk.Button(device_frame, text="Install Libs", command=self.install_dependencies, state="disabled", font=("Arial", 8))
        self.btn_install_deps.pack(side="left", padx=5)
        self.create_tooltip(self.btn_install_deps, "Install missing 'diffusers' library to the selected ComfyUI environment.\nRequired for VAE decoding.")

        # Low VRAM
        self.low_vram_var = tk.BooleanVar(value=self.config.get("low_vram", False))
        chk_low = tk.Checkbutton(settings_frame, text="Low VRAM Mode (Offload to RAM)", variable=self.low_vram_var)
        chk_low.pack(anchor="w")
        self.create_tooltip(chk_low, "If enabled, keeps VAE in RAM and only moves parts to GPU when needed.\nUse this if you get Out Of Memory errors.")

        # Output Directory
        out_frame = tk.Frame(settings_frame)
        out_frame.pack(fill="x", pady=2)
        tk.Label(out_frame, text="Output Directory:").pack(side="left")
        self.output_dir_var = tk.StringVar(value=self.config.get("output_dir", ""))
        tk.Entry(out_frame, textvariable=self.output_dir_var).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(out_frame, text="Browse", command=self.browse_output).pack(side="right")
        self.create_tooltip(out_frame, "Directory where generated images will be saved.\nIf empty, saves next to input file.")

        # Processed Directory
        processed_frame = tk.Frame(settings_frame)
        processed_frame.pack(fill="x", pady=2)
        tk.Label(processed_frame, text="Processed Dir:").pack(side="left")
        self.processed_dir_var = tk.StringVar(value=self.config.get("processed_dir", ""))
        tk.Entry(processed_frame, textvariable=self.processed_dir_var).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(processed_frame, text="Browse", command=self.browse_processed).pack(side="right")
        self.create_tooltip(processed_frame, "Directory to move source latent files after successful processing.\nIf empty, files remain in their original location.")

        # Filename Prefix
        prefix_frame = tk.Frame(settings_frame)
        prefix_frame.pack(fill="x", pady=2)
        tk.Label(prefix_frame, text="Filename Prefix:").pack(side="left")
        self.prefix_var = tk.StringVar(value=self.config.get("filename_prefix", ""))
        tk.Entry(prefix_frame, textvariable=self.prefix_var).pack(side="left", fill="x", expand=True, padx=5)
        self.create_tooltip(prefix_frame, "Prefix for output filenames.\nSupports %date:format% (e.g. %date:yyyy-MM-dd%)\nand %NodeTitle.widget% (e.g. %EmptyLatent.width%).\nFiles will be saved as Prefix_00001.png")

        # Debug Mode
        debug_frame = tk.Frame(settings_frame)
        debug_frame.pack(fill="x", pady=2)
        self.debug_var = tk.BooleanVar(value=self.config.get("debug_mode", False))
        chk_debug = tk.Checkbutton(debug_frame, text="Debug Mode", variable=self.debug_var)
        chk_debug.pack(side="left")
        self.create_tooltip(chk_debug, "Enable verbose logging and diagnostic reports on failure.")
        
        btn_diag = tk.Button(debug_frame, text="Generate Diagnostic Report", command=self.generate_diagnostic_report)
        btn_diag.pack(side="right")
        self.create_tooltip(btn_diag, "Manually generate a system diagnostic report to help troubleshoot issues.")

        # Actions
        action_frame = tk.Frame(self.root, padx=10, pady=10)
        action_frame.pack(fill="x")
        
        self.btn_run = tk.Button(action_frame, text="Select Latent Files & Run", command=self.select_latents, bg="#dddddd", height=2)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.create_tooltip(self.btn_run, "Click to select .latent files manually.\nOr drag and drop files onto this window.")
        
        self.btn_stop = tk.Button(action_frame, text="Stop", command=self.stop_process, bg="#ffcccc", height=2, state="disabled")
        self.btn_stop.pack(side="right", padx=(5, 0))
        self.create_tooltip(self.btn_stop, "Stop the current process and kill the worker subprocess.\nThis will also attempt to free VRAM/RAM.")

        # Log
        log_frame = tk.LabelFrame(self.root, text="Log", padx=5, pady=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, state="disabled", height=10)
        self.log_text.pack(fill="both", expand=True)

    def create_tooltip(self, widget, text):
        def enter(event):
            self.schedule_tooltip(event, text)
        def leave(event):
            self.hide_tooltip()
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
        # Bind motion to cancel if user moves out quickly? 
        # Actually <Leave> is reliable enough if we manage state correctly.

    def schedule_tooltip(self, event, text):
        # Cancel any pending tooltip
        self.hide_tooltip()
        # Schedule new one with 500ms delay
        self.tooltip_job = self.root.after(500, lambda: self.show_tooltip(event.widget, text))

    def show_tooltip(self, widget, text):
        # Double check if we should still show (in case job wasn't cancelled in time, though hide_tooltip should handle it)
        self.hide_tooltip() # Ensure clean state
        
        try:
            # Get widget absolute coordinates
            x_root = widget.winfo_rootx()
            y_root = widget.winfo_rooty()
            height = widget.winfo_height()
            
            # Position tooltip directly below the widget
            x = x_root
            y = y_root + height + 1
            
            self.tooltip_window = tk.Toplevel(self.root)
            self.tooltip_window.wm_overrideredirect(True)
            self.tooltip_window.wm_geometry(f"+{x}+{y}")
            # Ensure it's on top
            self.tooltip_window.attributes("-topmost", True)
            
            label = tk.Label(self.tooltip_window, text=text, justify='left',
                           background="#ffffe0", relief='solid', borderwidth=1,
                           font=("tahoma", "8", "normal"))
            label.pack(ipadx=1)
        except Exception as e:
            print(f"Tooltip error: {e}")

    def hide_tooltip(self):
        # Cancel pending job
        if self.tooltip_job:
            self.root.after_cancel(self.tooltip_job)
            self.tooltip_job = None
            
        # Destroy window
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def browse_comfy(self):
        path = filedialog.askdirectory(title="Select ComfyUI Installation Directory")
        if path:
            self.comfy_dir_var.set(path)
            self.save_config()
            # Try to check python status
            self.check_comfy_env()

    def update_scale_from_type(self):
        m_type = self.model_type_var.get()
        if m_type == "sd15":
            self.scale_var.set(0.18215)
        elif m_type == "sdxl":
            self.scale_var.set(0.13025)

    def load_config(self):
        self.config = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
            except:
                pass

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir_var.set(path)
            self.save_config()

    def browse_processed(self):
        path = filedialog.askdirectory(title="Select Processed Directory")
        if path:
            self.processed_dir_var.set(path)
            self.save_config()

    def save_config(self):
        self.config = {
            "comfy_dir": self.comfy_dir_var.get(),
            "vae_path": self.vae_path_var.get(),
            "model_type": self.model_type_var.get(),
            "scale_factor": self.scale_var.get(),
            "device": self.device_var.get(),
            "low_vram": self.low_vram_var.get(),
            "filename_prefix": self.prefix_var.get(),
            "output_dir": self.output_dir_var.get(),
            "processed_dir": self.processed_dir_var.get(),
            "debug_mode": self.debug_var.get()
        }
        with open(self.config_file, "w") as f:
            json.dump(self.config, f)
            
    def get_output_path(self, latent_path, prefix_pattern, output_dir_override=None):
        # Determine output directory
        if output_dir_override and os.path.isdir(output_dir_override):
            dir_name = output_dir_override
        else:
            dir_name = os.path.dirname(latent_path)
            
        # Parse prefix
        if not prefix_pattern:
            base_prefix = "image"
        else:
            formatted_prefix = prefix_pattern
            
            # 1. Date formatting %date:format%
            def date_replacer(match):
                fmt = match.group(1)
                fmt = fmt.replace("yyyy", "%Y").replace("MM", "%m").replace("dd", "%d")
                fmt = fmt.replace("HH", "%H").replace("mm", "%M").replace("ss", "%S")
                return datetime.now().strftime(fmt)
                
            formatted_prefix = re.sub(r"%date:(.*?)%", date_replacer, formatted_prefix)
            
            # 2. Workflow Metadata Extraction %NodeTitle.widget%
            if "%" in formatted_prefix:
                try:
                    metadata = {}
                    if latent_path.endswith(".safetensors") or latent_path.endswith(".latent"):
                        with safetensors.safe_open(latent_path, framework="pt", device="cpu") as f:
                            meta = f.metadata()
                            if meta and "workflow" in meta:
                                workflow = json.loads(meta["workflow"])
                                node_map = {}
                                if "nodes" in workflow:
                                    for node in workflow["nodes"]:
                                        title = node.get("title", node.get("type", ""))
                                        if title:
                                            node_map[title] = node
                                            
                                def node_replacer(match):
                                    token = match.group(1)
                                    if "." not in token:
                                        return match.group(0)
                                        
                                    node_title, widget_name = token.split(".", 1)
                                    node = node_map.get(node_title)
                                    if not node:
                                        return match.group(0)
                                    
                                    # Find widget value
                                    widget_index = -1
                                    current_idx = 0
                                    found_widget = False
                                    for inp in node.get("inputs", []):
                                        if "widget" in inp:
                                            if inp["widget"]["name"] == widget_name:
                                                widget_index = current_idx
                                                found_widget = True
                                                break
                                            current_idx += 1
                                    
                                    if found_widget and "widgets_values" in node:
                                        vals = node["widgets_values"]
                                        if 0 <= widget_index < len(vals):
                                            return str(vals[widget_index])
                                            
                                    return match.group(0)
    
                                formatted_prefix = re.sub(r"%([^%]+)%", node_replacer, formatted_prefix)
                except Exception as e:
                    print(f"Error parsing metadata: {e}")
                    
            # 3. Clean up invalid characters
            formatted_prefix = re.sub(r'[\\/*?:"<>|]', "", formatted_prefix)
            base_prefix = formatted_prefix

        # 4. Sequential Naming (Prefix_00001.png)
        # Find next available number in dir_name matching base_prefix_XXXXX.png
        # We need to scan directory
        i = 1
        while True:
            candidate_name = f"{base_prefix}{i:05d}.png"
            full_path = os.path.join(dir_name, candidate_name)
            if not os.path.exists(full_path):
                return full_path
            i += 1

    def browse_vae(self):
        path = filedialog.askopenfilename(title="Select VAE Model", filetypes=[("VAE Models", "*.safetensors *.pt *.ckpt *.bin")])
        if path:
            self.vae_path_var.set(path)
            self.save_config()

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update()

    def select_latents(self):
        files = filedialog.askopenfilenames(title="Select Latent Files", filetypes=[("Latent Files", "*.latent *.safetensors *.pt")])
        if files:
            self.run_process(files)

    def stop_process(self):
        if self.current_process:
            self.stop_requested = True
            self.log("Stopping process...")
            try:
                # Force kill to ensure memory release
                self.current_process.kill() 
            except Exception as e:
                self.log(f"Error killing process: {e}")
            self.btn_stop.config(state="disabled")
            self.btn_run.config(state="normal")
        else:
            self.stop_requested = True # Flag just in case loop is running but no process yet

    def get_comfy_python(self):
        comfy_dir = self.comfy_dir_var.get()
        if not comfy_dir or not os.path.exists(comfy_dir):
            return None
            
        # Prioritize python_embeded (Portable)
        candidates = [
            os.path.join(comfy_dir, "python_embeded", "python.exe"),
            os.path.join(comfy_dir, "python", "python.exe"),
            os.path.join(comfy_dir, "venv", "Scripts", "python.exe"),
            # Check if user pointed to python directory itself
            os.path.join(comfy_dir, "python.exe")
        ]
        
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def get_clean_env(self):
        # Create a clean environment for subprocesses
        # This removes PyInstaller's injected environment variables that cause conflicts
        env = os.environ.copy()
        
        # 1. Remove Python/PyInstaller specific variables
        for key in ["PYTHONPATH", "PYTHONHOME", "TCL_LIBRARY", "TK_LIBRARY", "LD_LIBRARY_PATH"]:
            if key in env:
                del env[key]
        
        # 2. Fix PATH: Remove PyInstaller's temp directory (_MEIxxxx)
        # This is critical to prevent the subprocess from loading the wrong DLLs
        if hasattr(sys, '_MEIPASS'):
            current_path = env.get("PATH", "")
            paths = current_path.split(os.pathsep)
            
            # Normalize MEIPASS for comparison
            mei_path = os.path.normcase(os.path.abspath(sys._MEIPASS))
            
            clean_paths = []
            for p in paths:
                # Normalize path for comparison
                norm_p = os.path.normcase(os.path.abspath(p))
                if mei_path not in norm_p:
                    clean_paths.append(p)
                    
            env["PATH"] = os.pathsep.join(clean_paths)
            
        # 3. Ensure essential system variables exist
        if "SystemRoot" not in env:
            env["SystemRoot"] = "C:\\Windows"
        if "windir" not in env:
            env["windir"] = "C:\\Windows"
            
        return env

    def check_comfy_env(self):
        py_path = self.get_comfy_python()
        if not py_path:
            self.cuda_label.config(text="[ComfyUI Python Not Found]", fg="red")
            self.dep_label.config(text="")
            self.btn_install_deps.config(state="disabled")
            return
            
        # Run a quick check for CUDA and Diffusers
        try:
            # Check script
            script = "import torch; import importlib.util; " \
                     "cuda = torch.cuda.is_available(); " \
                     "diffusers = importlib.util.find_spec('diffusers') is not None; " \
                     "print(f'{cuda}|{diffusers}')"
                     
            cmd = [py_path, "-c", script]
            
            # Use clean environment
            env = self.get_clean_env()
            
            # Prepend Python dir to PATH to ensure correct DLLs are found
            py_dir = os.path.dirname(py_path)
            env["PATH"] = py_dir + os.pathsep + env.get("PATH", "")
            
            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW, env=env, cwd=self.user_data_dir)
            
            if result.returncode == 0:
                out = result.stdout.strip()
                if "|" in out:
                    is_cuda_str, is_diffusers_str = out.split("|")
                    is_cuda = is_cuda_str == "True"
                    is_diffusers = is_diffusers_str == "True"
                    
                    # Update CUDA status
                    status = "Available" if is_cuda else "Not Found (Using CPU)"
                    color = "green" if is_cuda else "red"
                    self.cuda_label.config(text=f"[{status}]", fg=color)
                    
                    # Update Deps status
                    if is_diffusers:
                        self.dep_label.config(text="[Libs OK]", fg="green")
                        self.btn_install_deps.config(state="disabled")
                    else:
                        self.dep_label.config(text="[Missing diffusers]", fg="red")
                        self.btn_install_deps.config(state="normal")
                else:
                     self.cuda_label.config(text="[Check Output Invalid]", fg="red")
            else:
                self.cuda_label.config(text="[Environment Check Failed]", fg="red")
        except Exception as e:
            self.cuda_label.config(text=f"[Check Error]", fg="red")

    def install_dependencies(self):
        py_path = self.get_comfy_python()
        if not py_path:
            return
            
        if not messagebox.askyesno("Install Dependencies", "This will install 'diffusers' and 'transformers' library to your ComfyUI Python environment.\n\nDo you want to proceed?"):
            return
            
        def _install():
            self.btn_install_deps.config(state="disabled", text="Installing...")
            self.log("Installing dependencies (diffusers, transformers, accelerate)...")
            try:
                # pip install diffusers transformers accelerate
                cmd = [py_path, "-m", "pip", "install", "diffusers", "transformers", "accelerate", "--no-warn-script-location"]
                
                # Use clean environment
                env = self.get_clean_env()
                
                # Prepend Python dir to PATH
                py_dir = os.path.dirname(py_path)
                env["PATH"] = py_dir + os.pathsep + env.get("PATH", "")
                
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    bufsize=1, 
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    env=env,
                    cwd=self.user_data_dir
                )
                
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        self.log(f"PIP: {line.strip()}")
                
                if process.returncode == 0:
                    self.log("Dependencies installed successfully.")
                    self.root.after(0, self.check_comfy_env)
                    self.root.after(0, lambda: self.btn_install_deps.config(text="Install Libs"))
                else:
                    stderr = process.stderr.read()
                    self.log(f"Installation failed: {stderr}")
                    self.root.after(0, lambda: self.btn_install_deps.config(state="normal", text="Install Libs"))
                    
            except Exception as e:
                self.log(f"Error installing dependencies: {e}")
                self.root.after(0, lambda: self.btn_install_deps.config(state="normal", text="Install Libs"))

        threading.Thread(target=_install, daemon=True).start()

    def auto_run_dropped(self):
        self.run_process(self.dropped_files)

    def run_process(self, files):
        vae_path = self.vae_path_var.get()
        if not vae_path or not os.path.exists(vae_path):
            messagebox.showerror("Error", "Please select a valid VAE model first.")
            return

        self.save_config()
        
        # Run in thread to not freeze UI
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.stop_requested = False
        threading.Thread(target=self._process_thread, args=(files,), daemon=True).start()

    def _process_thread(self, files):
        py_path = self.get_comfy_python()
        if not py_path:
            self.log("Error: ComfyUI Python environment not found. Please set ComfyUI Directory in Settings.")
            return

        worker_script_src = os.path.join(self.resource_dir, "latent_decoder.py")
        if not os.path.exists(worker_script_src):
             # Fallback for dev mode
             worker_script_src = "latent_decoder.py"
             
        # Copy worker script to user_data_dir to avoid running it from PyInstaller's temp dir
        # This is crucial because running from _MEI dir adds it to sys.path, causing DLL conflicts
        worker_script = os.path.join(self.user_data_dir, "latent_decoder_worker.py")
        try:
            with open(worker_script_src, "r", encoding="utf-8") as src, open(worker_script, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        except Exception as e:
            self.log(f"Error copying worker script: {e}")
            return

        try:
            device = self.device_var.get()
            offload = self.low_vram_var.get()
            scale = self.scale_var.get()
            model_type = self.model_type_var.get()
            
            # Select config file based on model type
            config_filename = "sd15_config.json" if model_type == "sd15" else "sdxl_config.json"
            config_path = os.path.join(self.resource_dir, "configs", config_filename)
            
            if not os.path.exists(config_path):
                 self.log(f"Warning: Config file {config_path} not found. Using default.")
                 config_path = None # Will be passed as None string if we don't handle it, better handle in loop

            prefix = self.prefix_var.get()
            out_dir = self.output_dir_var.get()
            
            vae_path = self.vae_path_var.get()
            if not vae_path or not os.path.exists(vae_path):
                self.log("Error: VAE Path invalid.")
                return

            self.log(f"Using Environment: {py_path}")
            
            for file_path in files:
                if self.stop_requested:
                    self.log("Batch processing stopped by user.")
                    break
                    
                self.log(f"Processing: {file_path}")
                try:
                    # Calculate output path with prefix
                    output_path = self.get_output_path(file_path, prefix, output_dir_override=out_dir)
                    
                    # Construct Command
                    cmd = [
                        py_path,
                        worker_script,
                        "--latent", file_path,
                        "--vae", vae_path,
                        "--output", output_path,
                        "--scale", str(scale),
                        "--device", device
                    ]
                    
                    if offload:
                        cmd.append("--offload")
                    
                    if config_path:
                        cmd.extend(["--config", config_path])
                        
                    # Use clean environment
                    env = self.get_clean_env()
                    
                    # Prepend Python dir to PATH to ensure correct DLLs are found
                    py_dir = os.path.dirname(py_path)
                    env["PATH"] = py_dir + os.pathsep + env.get("PATH", "")
                        
                    # Run Worker
                    # Capture stdout/stderr line by line
                    self.current_process = subprocess.Popen(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True, 
                        bufsize=1, 
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        env=env,
                        cwd=self.user_data_dir
                    )
                    
                    process = self.current_process
                    
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            self.log(line.strip())
                            
                    stderr = process.stderr.read()
                    if stderr:
                        self.log(f"STDERR: {stderr}")
                        
                    if process.returncode == 0:
                        self.log(f"  -> Saved: {output_path}")
                        
                        # Move source file if processed_dir is set
                        processed_dir = self.processed_dir_var.get()
                        # Only move if it is a .latent file (per user request)
                        if processed_dir and os.path.isdir(processed_dir) and file_path.lower().endswith('.latent'):
                            try:
                                file_name = os.path.basename(file_path)
                                dest_path = os.path.join(processed_dir, file_name)
                                
                                # Handle duplicates
                                if os.path.exists(dest_path):
                                    base, ext = os.path.splitext(file_name)
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    dest_path = os.path.join(processed_dir, f"{base}_{timestamp}{ext}")
                                
                                shutil.move(file_path, dest_path)
                                self.log(f"  -> Moved source to: {dest_path}")
                            except Exception as move_err:
                                self.log(f"  -> Warning: Failed to move source file: {move_err}")
                    else:
                        if self.stop_requested:
                             self.log("  -> Process Terminated.")
                        else:
                             self.log("  -> Failed.")
                             if self.debug_var.get():
                                 self.generate_diagnostic_report(last_error=stderr)
                    
                    self.current_process = None

                except Exception as e:
                    self.log(f"  ERROR: {e}")
                    if self.debug_var.get():
                        self.generate_diagnostic_report(last_error=str(e))
            
            if not self.stop_requested:
                self.log("All tasks completed.")
            
        except Exception as e:
            self.log(f"Critical Error: {e}")
            if self.debug_var.get():
                self.generate_diagnostic_report(last_error=str(e))
        finally:
            self.root.after(0, lambda: self.btn_run.config(state="normal"))

    def generate_diagnostic_report(self, last_error=None):
        report = []
        report.append(f"Diagnostic Report - {datetime.now()}")
        report.append("-" * 30)
        report.append(f"App Config: {json.dumps(self.config, indent=2)}")
        
        py_path = self.get_comfy_python()
        report.append(f"ComfyUI Python Path: {py_path}")
        
        if py_path:
            try:
                cmd = [py_path, "-c", "import sys; import torch; print(f'Python: {sys.version}'); print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"]
                
                env = self.get_clean_env()
                py_dir = os.path.dirname(py_path)
                env["PATH"] = py_dir + os.pathsep + env.get("PATH", "")
                
                res = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW, env=env, cwd=self.user_data_dir)
                report.append(f"Environment Check:\n{res.stdout}\n{res.stderr}")
            except Exception as e:
                report.append(f"Environment Check Failed: {e}")
        else:
            report.append("Environment Check Skipped (Python not found)")
            
        if last_error:
            report.append(f"Last Error:\n{last_error}")
            
        filename = f"diagnostic_report_{int(datetime.now().timestamp())}.txt"
        path = os.path.join(self.user_data_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(report))
            self.log(f"Diagnostic report saved to: {path}")
            messagebox.showinfo("Diagnostic", f"Report saved to {filename}")
        except Exception as e:
            self.log(f"Failed to save diagnostic report: {e}")

if __name__ == "__main__":
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        
    app = App(root)
    root.mainloop()
