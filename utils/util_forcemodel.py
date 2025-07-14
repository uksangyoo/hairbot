import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np
import torchvision.models as models

# ForceModel class with encoder option
class ForceModel(nn.Module):
    def __init__(self, depth_encoder="cnn"):
        super(ForceModel, self).__init__()
        
        if depth_encoder == "cnn":
            # Custom CNN Encoder for depth image
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 60 * 80, 128),  # Adjust the input size after conv layers
                nn.ReLU()
            )
            self.depth_output_size = 128
        
        elif depth_encoder == "vgg":
            # Load VGG16 and modify the first conv layer to accept 1-channel input
            vgg = models.vgg16(pretrained=True)
            vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Modify input channels to 1
            self.cnn_encoder = nn.Sequential(*vgg.features, nn.Flatten(), nn.Linear(512 * 15 * 20, 256), nn.ReLU())
            self.depth_output_size = 256

        elif depth_encoder == "resnet":
            # Load ResNet and modify the first conv layer to accept 1-channel input
            resnet = models.resnet18(pretrained=True)
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify input channels to 1
            resnet.fc = nn.Identity()  # Remove the fully connected layer since we'll add our own
            self.cnn_encoder = resnet
            self.depth_output_size = 512

        else:
            raise ValueError(f"Unsupported depth_encoder type: {depth_encoder}")
        
        # MLP Encoder for joint torques
        self.mlp_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU()
        )
        
        # MLP for final force prediction
        self.mlp_predictor = nn.Sequential(
            nn.Linear(self.depth_output_size + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Predict 3D force vector (Fx, Fy, Fz)
        )
    
    def forward(self, depth_img, joint_torques):
        depth_features = self.cnn_encoder(depth_img)
        torque_features = self.mlp_encoder(joint_torques)
        combined_features = torch.cat((depth_features, torque_features), dim=1)
        force_pred = self.mlp_predictor(combined_features)
        return force_pred


# Custom weighted MSE loss function
def weighted_mse_loss(pred, target, weights):
    """
    Calculate the weighted MSE loss for each component (Fx, Fy, Fz).
    """
    loss = weights * (pred - target) ** 2
    return loss.mean()


# Training function with cosine annealing, encoder option, and weighted MSE loss
def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-4, run_name="default_run", save_dir="models", force_weights=None, T_max=None):
    """
    Train the model with cosine annealing, custom depth encoders (CNN, VGG, ResNet), and optionally using weighted MSE loss for force components.
    
    Args:
    - model: The PyTorch model to be trained.
    - train_loader: The training data loader.
    - val_loader: The validation data loader.
    - device: Device to use for training ('cpu' or 'cuda').
    - num_epochs: Number of training epochs.
    - learning_rate: Initial learning rate for the optimizer.
    - project: WandB project name.
    - entity: WandB entity name.
    - run_name: Custom name for the WandB run.
    - save_dir: Directory to save the trained model.
    - force_weights: Tensor of weights for (Fx, Fy, Fz). Default is equal weights (1.0, 1.0, 1.0).
    - T_max: Maximum number of iterations for cosine annealing.
    
    Returns:
    - None
    """
    # Initialize wandb with the run name
    wandb.init(name=run_name)
    wandb.watch(model)
    
    # Default weights if not provided
    if force_weights is None:
        force_weights = torch.tensor([1.0, 1.0, 1.0], device=device)

    criterion = lambda pred, target: weighted_mse_loss(pred, target, force_weights)  # Use weighted MSE loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize Cosine Annealing Scheduler
    if T_max is None:
        T_max = num_epochs  # Default to the number of epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    
    model.to(device)  # Move model to the selected device
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            depth_imgs, joint_torques, force_vectors = batch
            
            # Move inputs and targets to the selected device
            depth_imgs = depth_imgs.to(device)
            joint_torques = joint_torques.to(device)
            force_vectors = force_vectors.to(device)

            optimizer.zero_grad()
            outputs = model(depth_imgs, joint_torques)
            loss = criterion(outputs, force_vectors)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Update the learning rate using the cosine annealing scheduler
        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        wandb.log({"train_loss": avg_train_loss, "lr": scheduler.get_last_lr()[0]})
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                depth_imgs, joint_torques, force_vectors = batch
                
                # Move validation data to the correct device
                depth_imgs = depth_imgs.to(device)
                joint_torques = joint_torques.to(device)
                force_vectors = force_vectors.to(device)

                outputs = model(depth_imgs, joint_torques)
                loss = criterion(outputs, force_vectors)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        wandb.log({"val_loss": avg_val_loss})

    # Save the trained model to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"{run_name}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


# Real-time inference function with device parameter
def real_time_inference(model, depth_img, joint_torque, device):
    #model.to(device)  # Move model to the correct device
    depth_img[depth_img > 2000] = 0  # Clip depth image values to 2000
    depth_img = depth_img / 2000.0  # Normalize depth image
    
    depth_img = (depth_img - 0.5) * 2  # Center depth image
    joint_torque = joint_torque / 20.0  # Normalize joint torques
    
    depth_img_tensor = torch.tensor(depth_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    joint_torque_tensor = torch.tensor(joint_torque, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_force = model(depth_img_tensor, joint_torque_tensor)
    
    return predicted_force.squeeze().cpu().numpy()  # Return to CPU for further use

def load_model(model_class, file_path, device, depth_encoder="cnn"):
    """
    Load the model from a saved .pth file.
    
    Args:
    - model_class: The class of the model to be loaded (e.g., ForceModel).
    - file_path: The path to the saved .pth file.
    - device: The device to load the model onto (e.g., 'cuda' or 'cpu').
    - depth_encoder: The type of depth encoder used in the model (default is 'cnn').
    
    Returns:
    - model: The loaded model.
    """
    model = model_class(depth_encoder=depth_encoder).to(device)
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.eval()
    print(f"Model loaded from {file_path}")
    return model

# Dataset class
class ForceDataset(Dataset):
    def __init__(self, depth_images, joint_torques, force_vectors):
        self.depth_images = depth_images
        self.joint_torques = joint_torques
        self.force_vectors = force_vectors
    
    def __len__(self):
        return len(self.depth_images)
    
    def __getitem__(self, idx):
        depth_img = self.depth_images[idx] / 2000.0  # Normalize depth images (0-2000 -> 0-1)
        depth_img = (depth_img - 0.5) * 2  # Center at 0
        joint_torque = self.joint_torques[idx] / 20.0  # Normalize joint torques (-20 to 20 -> -1 to 1)
        force_vec = self.force_vectors[idx]
        
        return torch.tensor(depth_img, dtype=torch.float32).unsqueeze(0), torch.tensor(joint_torque, dtype=torch.float32), torch.tensor(force_vec, dtype=torch.float32)


# DataLoader function
def get_dataloaders(depth_images, joint_torques, force_vectors, batch_size=32, val_split=0.2):
    dataset = ForceDataset(depth_images, joint_torques, force_vectors)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Simulated example data
    depth_images = np.random.rand(1000, 480, 640) * 2000
    joint_torques = np.random.rand(1000, 4) * 40 - 20
    force_vectors = np.random.rand(1000, 3)
    
    train_loader, val_loader = get_dataloaders(depth_images, joint_torques, force_vectors)
    
    # Check if CUDA is available and select the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model with a selected encoder (cnn, vgg, or resnet)
    model = ForceModel(depth_encoder="resnet")  # Options: "cnn", "vgg", "resnet"
    
    # Specify force weights for weighted MSE
    force_weights = torch.tensor([1.0, 0.5, 2.0], device=device)
    
    # Train the model with cosine annealing, weighted MSE loss, selected encoder, and specified device
    train_model(model, train_loader, val_loader, device, run_name="resnet_cosine_model_run", save_dir="saved_models", force_weights=force_weights, T_max=10)

    # Real-time inference
    depth_img_sample = np.random.rand(480, 640) * 2000
    joint_torque_sample = np.random.rand(4) * 40 - 20
    predicted_force = real_time_inference(model, depth_img_sample, joint_torque_sample, device)
    print(f'Predicted Force: {predicted_force}')
