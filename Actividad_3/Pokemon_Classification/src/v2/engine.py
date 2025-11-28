"""
Contains the Trainer class for training and testing a PyTorch model.
Refactored to Object-Oriented Design (OOP) for better modularity and state management.
"""
import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

class Trainer:
    """
    Clase para gestionar el entrenamiento y evaluación de modelos PyTorch.
    Encapsula el ciclo de vida del entrenamiento, métricas y optimizaciones.
    """
    def __init__(self, 
                 model: nn.Module, 
                 train_dataloader: torch.utils.data.DataLoader, 
                 test_dataloader: torch.utils.data.DataLoader, 
                 optimizer: torch.optim.Optimizer, 
                 loss_fn: nn.Module, 
                 device: str):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Inicializar Scaler para Automatic Mixed Precision (AMP)
        # Se usa 'torch.amp' para evitar warnings de depreciación
        self.scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    def train_step(self) -> Tuple[float, float]:
        """
        Realiza un paso de entrenamiento (una época) sobre el conjunto de datos de entrenamiento.
        """
        self.model.train()
        train_loss, train_acc = 0, 0
        
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Optimización: Mixed Precision 
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    y_pred = self.model(X)
                    loss = self.loss_fn(y_pred, y)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Código estándar para CPU/MPS sin AMP
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Métricas
            train_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Promediar métricas por batch
        return train_loss / len(self.train_dataloader), train_acc / len(self.train_dataloader)

    def test_step(self) -> Tuple[float, float]:
        """
        Realiza un paso de evaluación sobre el conjunto de datos de prueba.
        """
        self.model.eval()
        test_loss, test_acc = 0, 0
        
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                
                test_pred = self.model(X)
                
                test_loss += self.loss_fn(test_pred, y).item()
                test_pred_labels = test_pred.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
        
        return test_loss / len(self.test_dataloader), test_acc / len(self.test_dataloader)

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        """
        Orquesta el proceso completo de entrenamiento y evaluación durante n épocas.
        """
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            train_loss, train_acc = self.train_step()
            test_loss, test_acc = self.test_step()
            
            
            print(
                f"Epoch: {epoch+1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
        return results