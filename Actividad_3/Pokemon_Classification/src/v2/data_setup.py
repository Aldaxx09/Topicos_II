"""
Este archivo se encarga de transformar tus carpetas de imágenes en tensores que PyTorch puede "masticar".
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Determina cuántos núcleos de CPU tiene tu máquina para cargar datos en paralelo.
# Es como contratar más ayudantes de cocina para picar verduras más rápido.
NUM_WORKERS = os.cpu_count()
NUM_WORKERS

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Ruta a la carpeta de entrenamiento.
    test_dir: Ruta a la carpeta de prueba.
    transform: Las transformaciones (data augmentation, resize) a aplicar.
    batch_size: Cuántas imágenes procesar a la vez (lote).
    num_workers: Cuántos sub-procesos usar para cargar datos.

  Returns:
    Una tupla: (train_dataloader, test_dataloader, class_names)
  """
  # Crear Datasets usando ImageFolder
  # ImageFolder escanea la carpeta y usa los nombres de las subcarpetas como etiquetas (clases).
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Obtener nombres de clases
  class_names = train_data.classes

  # Convertir Datasets en DataLoaders
  # El DataLoader es el "camarero" que sirve los datos en lotes al modelo.
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True, # Barajar entrenamiento para que el modelo no memorice el orden.
      num_workers=num_workers,
      pin_memory=True, # Acelera la transferencia de datos a la GPU.
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # No hace falta barajar en prueba, solo estamos evaluando.
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
# %%
