"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
Optimized for Windows & GPU throughput.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# En Windows, os.cpu_count() a veces es demasiado agresivo. 
# Usaremos un valor fijo seguro o 2, que suele ser el 'sweet spot'.
NUM_WORKERS = 2 

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int = NUM_WORKERS
):
  """
  Creates training and testing DataLoaders.
  Optimized with persistent_workers to speed up epochs on Windows.
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,          # Acelera la transferencia a la GPU
      persistent_workers=True,  # CRÍTICO EN WINDOWS: Mantiene los workers vivos entre épocas
      prefetch_factor=2         # Pre-carga batches mientras la GPU trabaja
  )
  
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, 
      num_workers=num_workers,
      pin_memory=True,
      persistent_workers=True,
      prefetch_factor=2
  )

  return train_dataloader, test_dataloader, class_names