"""
Machine Learning models for EMI shielding prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        Physics-informed neural network for EMI shielding prediction.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
        """
        super().__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 4)  # 4 outputs: R, A, M, total SE
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor containing:
               - Element properties (atomic number, mass, etc.)
               - Composition ratios
               - Frequency
               - Thickness
               
        Returns:
            Tensor containing:
            - Reflection loss
            - Absorption loss
            - Multiple reflection loss
            - Total shielding effectiveness
        """
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        
        # Split outputs
        r_loss, a_loss, m_loss, total_se = torch.split(x, 1, dim=1)
        
        # Apply physical constraints
        r_loss = torch.abs(r_loss)  # Reflection loss is positive
        a_loss = torch.abs(a_loss)  # Absorption loss is positive
        m_loss = -torch.abs(m_loss)  # Multiple reflection loss is negative
        
        # Total SE should be the sum of components
        total_se = r_loss + a_loss + m_loss
        
        return torch.cat([r_loss, a_loss, m_loss, total_se], dim=1)

class EMIPredictor:
    def __init__(self, input_dim: int):
        """
        EMI shielding prediction model manager.
        
        Args:
            input_dim: Number of input features
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PhysicsInformedNN(input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.mse_loss = nn.MSELoss()
        
    def prepare_input(self, material_data: Dict, frequency: float, 
                     thickness: float) -> torch.Tensor:
        """
        Prepare input data for the model.
        
        Args:
            material_data: Dictionary containing material properties
            frequency: Frequency in Hz
            thickness: Material thickness in meters
            
        Returns:
            Tensor containing formatted input data
        """
        # Extract relevant features
        features = [
            material_data["conductivity"],
            material_data["relative_permeability"],
            material_data["relative_permittivity"],
            material_data["density"],
            frequency,
            thickness
        ]
        
        return torch.tensor(features, dtype=torch.float32).to(self.device)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform one training step.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        y_pred = self.model(x)
        
        # Calculate loss
        loss = self.mse_loss(y_pred, y)
        
        # Physics-based regularization
        physics_loss = self._physics_consistency_loss(y_pred)
        total_loss = loss + 0.1 * physics_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def _physics_consistency_loss(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate physics consistency loss.
        
        Args:
            y_pred: Predicted values tensor
            
        Returns:
            Physics consistency loss
        """
        r_loss, a_loss, m_loss, total_se = torch.split(y_pred, 1, dim=1)
        
        # Physical constraints
        consistency_loss = torch.mean(torch.abs(total_se - (r_loss + a_loss + m_loss)))
        
        # Ensure losses are in reasonable ranges
        range_loss = torch.mean(torch.relu(-r_loss) + torch.relu(-a_loss) + 
                              torch.relu(m_loss) + torch.relu(-total_se))
        
        return consistency_loss + range_loss
    
    def predict(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Make predictions for input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing predicted shielding components
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x)
            r_loss, a_loss, m_loss, total_se = predictions[0].cpu().numpy()
            
            return {
                "reflection_loss": float(r_loss),
                "absorption_loss": float(a_loss),
                "multiple_reflection_loss": float(m_loss),
                "total_se": float(total_se)
            }
    
    def save_model(self, path: str):
        """
        Save model state.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """
        Load model state.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
