from flask import Flask, jsonify, send_file
import os
import torch
import torchvision
from diffusers import DDPMScheduler
from PIL import Image
from io import BytesIO

from tqdm import tqdm
from DM import ClassConditionedUnet  # Assuming this is your model definition
import yaml

app = Flask(__name__)

# Paths and Configuration
EXP_FOLDER = "../mnist_diffusion_experiments/exp-1"
MODEL_PATH = os.path.join(EXP_FOLDER, "conditional_mnist.pth")
CONFIG_PATH = os.path.join(EXP_FOLDER, "config.yaml")

# Load model and scheduler
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
noise_scheduler = None

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_model():
    global model, noise_scheduler

    config = load_config(CONFIG_PATH)
    model = ClassConditionedUnet(config["model"]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    noise_scheduler = DDPMScheduler(**config["scheduler"])


def generate_numbers_image(phone_number):
    """
    Generate an image of the digits in the phone number using the diffusion model.
    """
    digits = [int(d) for d in phone_number if d.isdigit()]
    if len(digits) == 0:
        raise ValueError("No valid digits found in phone number.")

    # Prepare random noise input
    batch_size = len(digits)
    x = torch.randn(batch_size, 1, 28, 28).to(device)
    y = torch.tensor(digits).to(device)

    # Sampling loop
    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(x, t, y)
        x = noise_scheduler.step(residual, t, x).prev_sample

    # Convert to image
    generated_images = x.detach().cpu().clip(-1, 1)
    grid_image = torchvision.utils.make_grid(generated_images, nrow=batch_size, padding=2, pad_value=1)

    # Convert to PIL image
    pil_image = torchvision.transforms.ToPILImage()(grid_image)
    return pil_image


@app.route("/generate_image/<phone_number>", methods=["GET"])
def generate_image_route(phone_number):
    """
    Flask route to generate an image of the phone number passed in the URL.
    """
    try:
        image = generate_numbers_image(phone_number)
        img_io = BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png", as_attachment=True, download_name=f"{phone_number}_generated_image.png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
