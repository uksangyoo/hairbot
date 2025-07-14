
import os
import wandb  
import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def sample_random_points(pcd, num_points=2048):
    """
    Randomly sample points from a point cloud.
    
    Parameters:
    - pcd: Open3D point cloud object
    - num_points: The number of points to sample, default is 2048
    
    Returns:
    - sampled_pcd: Open3D point cloud object with sampled points
    """
    
    # Convert the point cloud to numpy array
    points = np.asarray(pcd.points)

    # Check if the point cloud has enough points to sample
    if len(points) < num_points:
        raise ValueError("Point cloud has fewer points than the requested sample size.")
    
    # Randomly sample indices
    sampled_indices = np.random.choice(len(points), num_points, replace=False)

    # Create a new point cloud with the sampled points
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(points[sampled_indices])

    # If the point cloud has colors, sample colors as well
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        sampled_pcd.colors = o3d.utility.Vector3dVector(colors[sampled_indices])

    return sampled_pcd



# Define custom dataset for point cloud data
class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, forces):
        self.point_clouds = point_clouds
        self.forces = forces

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return torch.Tensor(self.point_clouds[idx]), torch.Tensor(self.forces[idx])



# Define a T-Net (Transformation Network)
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1024)

        # Initialize the weights for the transformation matrix
        self.fc3.bias.data.zero_()
        self.fc3.weight.data.zero_()

        self.k = k

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x).squeeze(-1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        id_matrix = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1).to(x.device)
        matrix = x.view(-1, self.k, self.k) + id_matrix
        return matrix

# Define the PointNet model
class PointNet(nn.Module):
    def __init__(self, num_points=2048, output_dim=3):
        super(PointNet, self).__init__()
        
        self.input_transform = TNet(k=3)  # T-Net for input transformation
        self.feature_transform = TNet(k=64)  # T-Net for feature transformation (optional)
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)  # Predicting the 3D force vector
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(num_points)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Input transformation
        transform_matrix = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), transform_matrix).transpose(2, 1)
        
        # Point feature extraction
        x = self.relu(self.conv1(x))
        
        # Feature transformation (optional, improves robustness)
        transform_matrix_64 = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), transform_matrix_64).transpose(2, 1)
        
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x).squeeze(-1)  # Global features
        
        # MLP for force prediction
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output the 3D force vector
        
        return x

# Define the PointNet model without T-Net
class PointNetNoTNet(nn.Module):
    def __init__(self, num_points=2048, output_dim=3):
        super(PointNetNoTNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)  # Predicting the 3D force vector
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(num_points)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Transpose the point cloud from (batch_size, num_points, 3) to (batch_size, 3, num_points)
        x = x.transpose(2, 1)
        
        # Point feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x).squeeze(-1)  # Global features
        
        # MLP for force prediction
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output the 3D force vector
        
        return x

# Save the model and optimizer states
def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch + 1,  # +1 because epochs are zero-indexed
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

# Test the model and compute the average loss
def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for points, forces in tqdm(test_loader, desc="Testing"):
            points, forces = points.to(device), forces.to(device)
            outputs = model(points)
            loss = criterion(outputs, forces)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

# Deploy model on a single point cloud to predict the force vector
def deploy_model(model, point_cloud):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        point_cloud = torch.Tensor(point_cloud).unsqueeze(0).to(device)  # Add batch dimension
        prediction = model(point_cloud)
    return prediction.squeeze(0).cpu().numpy()  # Remove batch dimension for output

# Train the model with tqdm, checkpoint saving, CUDA support, and WandB integration
def train_model(model, train_loader, criterion, optimizer, epochs=20, checkpoint_dir="./checkpoints", save_freq=5, project_name="pointnet_training"):
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists

    # Initialize WandB run
    wandb.init(project=project_name)
    wandb.watch(model, log="all", log_freq=10)

    # Check if CUDA is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Using device: {device}")

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for batch_idx, (points, forces) in enumerate(progress_bar):
            points = points.to(device)
            forces = forces.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, forces)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

        # Save a checkpoint every 'save_freq' epochs
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth.tar")
            save_checkpoint(model, optimizer, epoch, avg_loss, filename=checkpoint_path)

            # Optionally, if you want to track the checkpoint in WandB, ensure the path is valid
            # Comment out or remove the wandb.save() line to avoid errors
            # wandb.save(checkpoint_path)

    wandb.finish()
# Main function to train, test, and deploy the model
def main():
    # Example initialization (replace these with actual data loaders)
    num_points = 1024
    output_dim = 3
    train_loader = ...  # Replace with actual DataLoader for training data
    test_loader = ...  # Replace with actual DataLoader for test data
    
    model = PointNetNoTNet(num_points=num_points, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=20, checkpoint_dir="./checkpoints", save_freq=5, project_name="pointnet_no_tnet_training")

    # Test the model
    test_model(model, test_loader, criterion)

    # Deploy on a single point cloud (example)
    sample_point_cloud = ...  # Replace with an actual Nx3 point cloud
    predicted_force = deploy_model(model, sample_point_cloud)
    print("Predicted Force Vector:", predicted_force)

if __name__ == "__main__":
    main()
