"""
Aquí se definen los planos de la red neuronal. En este caso, es una llamada de 
arquitectura TinyVGG(una versión pequeña de la famosa VGG).
"""
import torch
from torch import nn 

# Heredamos de nn.Module, que es la clase base para todas las redes en PyTorch.
class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.
  
  Args:
    input_shape: Canales de entrada (3 para imágenes a color RGB).
    hidden_units: Neuronas en las capas ocultas (ancho de la red).
    output_shape: Número de clases a predecir (150 pokémon).
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      
      # Primer bloque de convolución: Extrae características básicas (bordes, colores).
      self.conv_block_1 = nn.Sequential(
          # Conv2d: El "ojo" que escanea la imagen buscando patrones.
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, # Tamaño del filtro (3x3 pixels)
                    stride=1, 
                    padding=0),  
          nn.ReLU(), # Función de activación (agrega no-linealidad).
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          # MaxPool2d: Reduce la imagen a la mitad, quedándose con lo más importante.
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      
      # Segundo bloque: Busca patrones más complejos combinando los anteriores.
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      
      # Clasificador: Toma las características extraídas y decide qué Pokémon es.
      self.classifier = nn.Sequential(
          nn.Flatten(), # Aplana el mapa de características 2D a un vector 1D.
          # Linear: La capa final que conecta todo a las salidas (clases).
          # Nota: 'hidden_units*13*13' depende del tamaño de entrada de la imagen.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )
    
  # Forward define cómo pasan los datos por la red.
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x) # Pasa por bloque 1
      x = self.conv_block_2(x) # Pasa por bloque 2
      x = self.classifier(x)   # Pasa por el clasificador
      return x