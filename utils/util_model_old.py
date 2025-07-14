import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
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
# Define the PointNet model
class PointNet(nn.Module):
    def __init__(self, num_points=2048, output_dim=3):
        super(PointNet, self).__init__()
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
        x = x.transpose(2, 1)  # Convert (batch_size, N, 3) to (batch_size, 3, N)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x).squeeze(-1)  # (batch_size, 1024)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output the 3D force vector
        
        return x

# Custom dataset for point cloud and force vector
class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, forces):
        self.point_clouds = point_clouds
        self.forces = forces

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = torch.Tensor(self.point_clouds[idx])
        force = torch.Tensor(self.forces[idx])
        return point_cloud, force

# Train the model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (points, forces) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, forces)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Test the model
def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (points, forces) in enumerate(test_loader):
            outputs = model(points)
            loss = criterion(outputs, forces)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

# Deploy model for single prediction
def deploy_model(model, point_cloud):
    model.eval()
    with torch.no_grad():
        point_cloud_tensor = torch.Tensor(point_cloud).unsqueeze(0)  # Add batch dimension
        prediction = model(point_cloud_tensor)
    return prediction.squeeze(0).numpy()  # Remove batch dimension for output


# Main function to run training, testing, and deployment
def main():
    # Example point clouds and force vectors (use your dataset here)
    train_point_clouds = ...  # (N_train, num_points, 3)
    train_forces = ...  # (N_train, 3)
    
    test_point_clouds = ...  # (N_test, num_points, 3)
    test_forces = ...  # (N_test, 3)
    
    train_dataset = PointCloudDataset(train_point_clouds, train_forces)
    test_dataset = PointCloudDataset(test_point_clouds, test_forces)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = PointNet(num_points=train_point_clouds.shape[1], output_dim=3)
    criterion = nn.MSELoss()  # Loss for force prediction
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=20)
    
    # Test the model
    test_loss = test_model(model, test_loader, criterion)

    # Deploy the model on a single point cloud
    sample_point_cloud = test_point_clouds[0]
    predicted_force = deploy_model(model, sample_point_cloud)
    print("Predicted Force Vector:", predicted_force)


if __name__ == "__main__":
    main()
