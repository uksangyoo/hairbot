from utils.util_forcemodel import get_dataloaders, ForceModel, train_model, real_time_inference
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


root_dir = 'XXX'
data_names =['d1', 'd2','d3','d4'  ]

depth_images = []
joint_torques = []
force_vectors = []
for data_name in data_names:
    data_dir = root_dir + data_name + '/'
    num_files = len([name for name in os.listdir(data_dir+'pcd') if os.path.isfile(os.path.join(data_dir + 'pcd', name))])
    print(f'Number of files in {data_dir}: {num_files}')
    for i in range(num_files):
        depth_images.append(np.load(data_dir + f'depth_image/{i}.npy'))
        joint_torques.append(np.load(data_dir + f'currents/{i}.npy'))
        wrench = np.load(data_dir + f'wrench/{i}.npy')
        average_wrench = np.mean(wrench, axis=0)
        Fz = average_wrench[2]
        Fz = np.array([0,0,Fz])
        frame = np.load(data_dir + f'aruco_pose/{i}.npy')
        Fz = frame[0:3, 0:3]@ Fz
        force_vectors.append(Fz)
 
depth_images = np.array(depth_images)
joint_torques = np.array(joint_torques)
force_vectors = np.array(force_vectors)
print(f'depth_images shape: {depth_images.shape}')
print(f'joint_torques shape: {joint_torques.shape}')
print(f'force_vectors shape: {force_vectors.shape}')   

# # Assuming depth_images, joint_torques, force_vectors are numpy arrays
# depth_images = np.random.rand(1000, 480, 640) * 2000  # Example data
# joint_torques = np.random.rand(1000, 4) * 40 - 20     # Example data
# force_vectors = np.random.rand(1000, 3)               # Example data



train_loader, val_loader = get_dataloaders(depth_images, joint_torques, force_vectors)

# Check if CUDA is available and select the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model with a selected encoder (cnn, vgg, or resnet)
model = ForceModel(depth_encoder="vgg")  # Options: "cnn", "vgg", "resnet"

# Specify force weights for weighted MSE

# Calculate the standard deviation of each component in force_vectors
force_std = np.std(force_vectors, axis=0)

# Invert the standard deviations to get the weights
force_weights = 1 / force_std

# Convert the weights to a tensor and move to the appropriate device
force_weights = torch.tensor(force_weights, device=device)
print(f'Force weights: {force_weights}')
#force_weights = torch.tensor([1.0, 0.5, 2.0], device=device)

# Train the model with weighted MSE loss, selected encoder, specified device, run name, and save directory
train_model(model, train_loader, val_loader, device, run_name="vgg_cosine", save_dir="saved_mode/home/frida/Projects/hairbot/model", force_weights=force_weights,  num_epochs=100, T_max=50)
# Real-time inference
depth_img_sample = np.random.rand(480, 640) * 2000
joint_torque_sample = np.random.rand(4) * 40 - 20
predicted_force = real_time_inference(model, depth_img_sample, joint_torque_sample, device)
print(f'Predicted Force: {predicted_force}')
