import torch
import safetensors.torch
from diffusers import AutoencoderKL
from PIL import Image
import os
import gc

class LatentDecoder:
    def __init__(self):
        self.vae = None
        self.vae_path = None
        self.device = "cpu"
        self.offload = False
        self.scale_factor = 0.18215 # Default to SD1.5, allow change

    def load_vae(self, vae_path, device="cuda", offload=False, config_path=None):
        self.device = device
        self.offload = offload
        
        if self.vae_path == vae_path and self.vae is not None:
            print("VAE already loaded.")
            self.move_vae_to_device()
            return

        print(f"Loading VAE from {vae_path}...")
        try:
            # Attempt to load as single file (safetensors or ckpt)
            # Use map_location="cpu" to avoid meta device issues
            # Pass config_path if provided to avoid downloading config from Hub
            self.vae = AutoencoderKL.from_single_file(vae_path, map_location="cpu", config=config_path)
        except Exception as e:
            print(f"Failed to load VAE with diffusers: {e}")
            # Fallback or re-raise
            raise e

        self.vae_path = vae_path
        self.move_vae_to_device()
        
        # Enable slicing and tiling to save memory, especially on CPU or low VRAM
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
            
        print("VAE loaded successfully.")

    def move_vae_to_device(self):
        if self.vae is None:
            return
            
        if self.offload:
            # Keep on CPU, move to GPU only when needed (manual offload logic)
            # For simplicity in this script, if offload is True, we keep it on CPU 
            # and move to GPU during decode, then back.
            self.vae.to("cpu")
        else:
            if self.device == "cuda" and torch.cuda.is_available():
                self.vae.to("cuda")
            else:
                self.vae.to("cpu")

    def decode_latent(self, latent_path, output_path=None, scale_factor=None):
        if self.vae is None:
            raise ValueError("VAE not loaded.")

        if scale_factor is not None:
            self.scale_factor = scale_factor

        print(f"Loading latent from {latent_path}...")
        try:
            latent_tensor = self.load_latent_file(latent_path)
        except Exception as e:
            print(f"Error loading latent file: {e}")
            return []
        
        # Latent shape: (B, C, H, W). usually (1, 4, H, W)
        print(f"Latent shape: {latent_tensor.shape}")
        
        # Calculate and print stats
        l_mean = latent_tensor.mean().item()
        l_std = latent_tensor.std().item()
        l_min = latent_tensor.min().item()
        l_max = latent_tensor.max().item()
        print(f"Latent Stats - Mean: {l_mean:.4f}, Std: {l_std:.4f}, Min: {l_min:.4f}, Max: {l_max:.4f}")
        
        if l_std > 0.5:
             print("WARNING: Latent standard deviation is high (> 0.5). It might already be unscaled. Dividing by scale factor might cause artifacts.")
        elif l_std < 0.05:
             print("WARNING: Latent standard deviation is very low (< 0.05). It might be empty or wrong.")

        # Determine target device
        target_device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        
        try:
            return self._execute_decode(latent_tensor, target_device, output_path, latent_path)
        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory. Switching to CPU/Offload mode...")
            torch.cuda.empty_cache()
            return self._execute_decode(latent_tensor, "cpu", output_path, latent_path)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("CUDA Out of Memory (RuntimeError). Switching to CPU...")
                torch.cuda.empty_cache()
                return self._execute_decode(latent_tensor, "cpu", output_path, latent_path)
            else:
                raise e

    def _execute_decode(self, latent_tensor, device, output_path, latent_path):
        # If offloading or fallback to CPU, ensure VAE is on the correct device
        # If device is CPU, move VAE to CPU
        # If device is CUDA, move VAE to CUDA (if not offloaded)
        
        # For simple fallback: If we are here, we want to run on 'device'.
        
        # Ensure VAE is on the target device
        self.vae.to(device)
        latent_tensor = latent_tensor.to(device)
        latent_tensor = latent_tensor.to(self.vae.dtype)

        scaled_latent = latent_tensor / self.scale_factor
        
        print(f"Applying Scale Factor: {self.scale_factor}")
        print(f"Scaled Latent Stats - Mean: {scaled_latent.mean().item():.4f}, Std: {scaled_latent.std().item():.4f}")

        print(f"Decoding on {device}...")
        with torch.no_grad():
            image = self.vae.decode(scaled_latent).sample

        # Post-process image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # If we moved VAE to CUDA and offload is enabled, move back to CPU?
        # But here we might have forced CPU due to OOM.
        # Restore VAE device preference if needed? 
        # For now, just leave it where it is or reset to config preference?
        # Let's leave it. The next run will call move_vae_to_device or we can reset.
        if self.offload and device == "cuda":
             self.vae.to("cpu")
             torch.cuda.empty_cache()
        
        # Save images
        saved_paths = []
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(latent_path))[0]
            output_dir = os.path.dirname(latent_path)
            output_path = os.path.join(output_dir, base_name + ".png")

        # Handle batch
        for i, img in enumerate(image):
            img_pil = Image.fromarray((img * 255).round().astype("uint8"))
            if len(image) > 1:
                save_path = output_path.replace(".png", f"_{i}.png")
            else:
                save_path = output_path
            
            img_pil.save(save_path)
            saved_paths.append(save_path)
            print(f"Saved to {save_path}")

        return saved_paths

    def load_latent_file(self, path):
        # Support .latent (safetensors or torch pickle) and .safetensors
        if path.endswith(".safetensors") or path.endswith(".latent"):
            try:
                # Try loading as safetensors first
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    # ComfyUI usually saves latent in "latent_tensor" or similar key?
                    # Or it might be just the whole dict.
                    # Standard ComfyUI .latent format (safetensors):
                    # keys: "latent_tensor", "latent_format_version"
                    keys = f.keys()
                    if "latent_tensor" in keys:
                        return f.get_tensor("latent_tensor")
                    
                    # If not, maybe it's just the tensor itself with some key?
                    # Let's iterate and find a 4-channel tensor
                    for k in keys:
                        t = f.get_tensor(k)
                        if len(t.shape) == 4 and t.shape[1] == 4:
                            return t
            except Exception as e:
                pass # Try torch.load next

        # Fallback to torch.load (pickle)
        # ComfyUI .latent can be a pickle dict: {'latent_tensor': ...}
        # Safe to use weights_only=False locally if user trusts the file, but standard is True.
        # However, ComfyUI latents might need weights_only=False if they contain custom classes? 
        # Usually they are just tensors.
        try:
            data = torch.load(path, map_location="cpu", weights_only=False) # Use False to be compatible with older files
            if isinstance(data, dict):
                if "latent_tensor" in data:
                    return data["latent_tensor"]
                # Search for tensor
                for v in data.values():
                    if isinstance(v, torch.Tensor) and len(v.shape) == 4 and v.shape[1] == 4:
                        return v
            elif isinstance(data, torch.Tensor):
                return data
        except Exception as e:
            print(f"Failed to load latent with torch.load: {e}")
            raise ValueError(f"Could not load latent from {path}")
            
        raise ValueError(f"No suitable latent tensor found in {path}")

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Latent Decoder Worker")
    parser.add_argument("--latent", type=str, required=True, help="Path to latent file")
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE model")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--scale", type=float, default=0.18215, help="Scale factor")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--offload", action="store_true", help="Enable VRAM offload")
    parser.add_argument("--config", type=str, default=None, help="Path to VAE config json")
    
    args = parser.parse_args()
    
    try:
        print(f"Worker: Starting decode for {args.latent}")
        decoder = LatentDecoder()
        
        # Load VAE
        decoder.load_vae(args.vae, device=args.device, offload=args.offload, config_path=args.config)
        
        # Decode
        output_files = decoder.decode_latent(args.latent, output_path=args.output, scale_factor=args.scale)
        
        if output_files:
            print("Worker: Success")
            sys.exit(0)
        else:
            print("Worker: Failed to generate output")
            sys.exit(1)
            
    except Exception as e:
        print(f"Worker Error: {e}")
        # Print full traceback for debug
        import traceback
        traceback.print_exc()
        sys.exit(1)
