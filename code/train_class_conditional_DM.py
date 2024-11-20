import yaml
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from DM import ClassConditionedUnet
from datetime import datetime
import os
from os.path import join


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def save_config(config, path):
    """
    Save the config dictionary to a YAML file.
    """
    with open(path, "w") as f:
        yaml.dump(config, f)


def get_device(config):
    if config["device"]["use_mps"] and torch.backends.mps.is_available():
        return "mps"
    elif config["device"]["use_cuda"] and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_data(dataset_config):
    dataset = torchvision.datasets.MNIST(
        root=dataset_config["root"],
        train=True,
        download=dataset_config["download"],
        transform=torchvision.transforms.ToTensor(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=dataset_config["shuffle"],
    )
    return dataloader


def train_one_epoch(net, dataloader, noise_scheduler, opt, loss_fn, device, writer, global_step):
    epoch_loss = 0.0
    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        x = x.to(device) * 2 - 1
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        pred = net(noisy_x, timesteps, y)
        loss = loss_fn(pred, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        writer.add_scalar("Loss/Batch", loss.item(), global_step)
        global_step += 1

    return epoch_loss / len(dataloader), global_step


def main():
    # Load configuration
    config = load_config()
    
    # Save the config file used for training
    os.makedirs(config["logging"]["config_save_path"],exist_ok=True)
    save_config(config, join(config["logging"]["config_save_path"],"config.yaml"))
    
    # Set device
    device = get_device(config)
    print(f"Using device: {device}")

    # Load dataset
    train_dataloader = load_data(config["dataset"])

    # Initialize model, loss, optimizer, and scheduler
    net = ClassConditionedUnet(config["model"]).to(device)
    
    loss_fn = nn.MSELoss()
    
    opt = torch.optim.Adam(net.parameters(), lr=float(config["training"]["lr"]))
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["scheduler"]["num_train_timesteps"],
        beta_schedule=config["scheduler"]["beta_schedule"],
    )

    # Initialize TensorBoard writer
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"{config['logging']['tensorboard_dir']}/{timestamp}" 
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    global_step = 0
    
    for epoch in range(config["training"]["n_epochs"]):
        avg_loss, global_step = train_one_epoch(
            net, train_dataloader, noise_scheduler, opt, loss_fn, device, writer, global_step
        )
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.5f}")
        writer.add_scalar("Loss/Epoch", avg_loss, epoch)

    # Save loss curve and model
    torch.save(net.state_dict(), join(config["logging"]["model_save_path"], "conditional_mnist.pth"))
    writer.close()


if __name__ == "__main__":
    main()
