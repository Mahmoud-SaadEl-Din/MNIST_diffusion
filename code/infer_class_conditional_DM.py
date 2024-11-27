import os
import yaml
import torch
from DM import ClassConditionedUnet
from tqdm.auto import tqdm
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from PIL import ImageDraw, Image
# Define the main directory
main_experiments_dir = "../mnist_diffusion_experiments"
output_dir = "../generation_out"
os.makedirs(output_dir, exist_ok=True)

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Function to load model and config
def load_model_and_config(exp_dir):
    with open(os.path.join(exp_dir, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    
    model = ClassConditionedUnet(config["model"]).to(device)
    
    model.load_state_dict(torch.load(os.path.join(exp_dir, "conditional_mnist.pth"), map_location=device))
    return model, config

def create_diffusion_gif(frames, exp_name, output_path):
    images = []
    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
    
    for i, frame in enumerate(frames):  # Each frame is (5, 1, 28, 28)
        # Arrange the 5 numbers side by side
        grid = make_grid(frame, nrow=5, normalize=True, value_range=(-1, 1))  # Shape: (1, H, W*5)
        img = ToPILImage()(grid.cpu())  # Convert the grid to a PIL image

        # Annotate the timestep
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"t={i}", fill="red")  # Add timestep as text in the top-left corner

        # # Save the image frame
        # frame_path = os.path.join(output_path, f"{exp_name}_t{i:03d}.png")
        # img.save(frame_path)

        # Collect for GIF
        images.append(img)

    # Save all frames as a GIF
    gif_path = os.path.join(output_path, f"{exp_name}_diffusion.gif")
    images[0].save(
        gif_path, save_all=True, append_images=images[1:], loop=0, duration=100
    )
    print(f"Saved diffusion GIF for {exp_name} at {gif_path}")

# Loop through experiment folders
all_results = []
for exp in sorted(os.listdir(main_experiments_dir)):
    exp_dir = os.path.join(main_experiments_dir, exp)
    if not os.path.isdir(exp_dir):
        continue
    
    print(f"Processing {exp}...")
    
    # Load model and configuration
    model, config = load_model_and_config(exp_dir)
    model.eval()
    
    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=config["scheduler"]["num_train_timesteps"], beta_schedule="squaredcos_cap_v2")
    
    # Generate numbers 0 to 4
    x = torch.randn(5, 1, 28, 28).to(device)  # Starting noise
    y = torch.arange(5).to(device)  # Labels 0-4
    
    frames = []  # To store intermediate diffusion steps
    # print(noise_scheduler.timesteps)
    for i, t in tqdm(enumerate(noise_scheduler.timesteps), desc=f"{exp} Sampling Loop", total=len(noise_scheduler.timesteps)):
        with torch.no_grad():
            residual = model(x, t, y)
        x = noise_scheduler.step(residual, t, x).prev_sample
        if i % 20 == 0:  # Save every 50 steps for the GIF
            frames.append(x.detach().clone())
    
    # Save GIF of the diffusion process
    create_diffusion_gif(frames, exp, output_dir)
    # Detach and clip generated images
    generated_images = x.detach().cpu().clip(-1, 1)
    all_results.append((exp, generated_images))
    
    # Save individual experiment results
    save_image(generated_images, f"{output_dir}/{exp}_generated.png", nrow=5, normalize=True, value_range=(-1, 1))

# Concatenate results vertically
final_image = torch.cat([res[1] for res in all_results], dim=0)
save_image(final_image, f"{output_dir}/all_experiments_generated.png", nrow=5, normalize=True, value_range=(-1, 1))

print("Inference complete. Results saved in:", output_dir)