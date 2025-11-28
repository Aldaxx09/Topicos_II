"""
Contiene la lógica del bucle de entrenamiento. Es el "gimnasio" donde el modelo suda y aprende."""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# --- PASO DE ENTRENAMIENTO (UN LOTE) ---
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch."""
  
  model.train() # Pone el modelo en modo "aprendizaje" (activa Dropout, Batchnorm, etc.)
  
  train_loss, train_acc = 0, 0
  
  # Itera sobre cada lote de imágenes
  for batch, (X, y) in enumerate(dataloader):
      # Mueve datos a la GPU/CPU
      X, y = X.to(device), y.to(device)

      # 1. Forward pass: El modelo hace una predicción
      y_pred = model(X)

      # 2. Calcular pérdida: ¿Qué tan mal se equivocó?
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Limpiar gradientes anteriores (esencial en PyTorch)
      optimizer.zero_grad()

      # 4. Backward pass: Calcula cómo ajustar los pesos (Backpropagation)
      loss.backward()

      # 5. Optimizer step: Actualiza los pesos para reducir el error
      optimizer.step()

      # Calcular precisión (Accuracy)
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Promedios finales por época
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

# --- PASO DE PRUEBA (EVALUACIÓN) ---
def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch."""
  
  model.eval() # Pone el modelo en modo "examen" (congela capas de aprendizaje)
  
  test_loss, test_acc = 0, 0
  
  # Inference Mode: Desactiva el cálculo de gradientes (ahorra memoria y es más rápido)
  with torch.inference_mode():
      for batch, (X, y) in enumerate(dataloader):
          X, y = X.to(device), y.to(device)
  
          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calcular pérdida (solo para reportar, no para entrenar)
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()
          
          # Calcular precisión
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
          
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

# --- ORQUESTADOR PRINCIPAL ---
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model."""
  
  # Diccionario para guardar el historial de resultados
  results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
  
  # Bucle por cada época (vuelta completa al dataset)
  for epoch in tqdm(range(epochs)):
      # Entrenar
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      # Evaluar
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      
      # Imprimir progreso
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Guardar resultados
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  return results