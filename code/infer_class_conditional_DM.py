import torch
from DM import ClassConditionedUnet
from tqdm.auto import tqdm
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import torchvision
from prettytable import PrettyTable

def count_parameters_table(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def count_parameters(model):
    trained_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_parameters = sum(p.numel() for p in model.parameters())

    return trained_parameters, all_parameters

load_model_name = "conditional_mnist.pth"
trained_model_path = "../trained_models"
out_generation_path = "../generation_out"
name_to_save = "1st_generation"


device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
net = ClassConditionedUnet().to(device)
net.load_state_dict(torch.load(f"{trained_model_path}/{load_model_name}"))
# trained_model_parameter, all_model_parameter = count_parameters(net)
# print("model trained parameter: ", trained_model_parameter, ", all parameters: ",all_model_parameter)
# count_parameters_table(net)

# net.eval()
# Create a scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

# Prepare random x to start from, plus some desired labels y
x = torch.randn(80, 1, 28, 28).to(device)
y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)

# Sampling loop
for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

    # Get model pred
    with torch.no_grad():
        residual = net(x, t, y)  # Again, note that we pass in our labels y

    # Update sample with step
    x = noise_scheduler.step(residual, t, x).prev_sample

# Show the results
# fig, ax = plt.subplots(1, 1, figsize=(12, 12))

plt.imsave(f"{out_generation_path}/{name_to_save}.png",torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap="Greys")