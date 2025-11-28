"""
Este es el guión principal. Importa todos los demás módulos y coordina la ejecución. 
Este es el que tú ejecutas en la terminal ( python train.py)."""

import os
import torch
# IMPORTANTE: Aquí importamos nuestros propios módulos creados arriba
import data_setup, engine, model_builder, utils

from torchvision import transforms

def main():

    # --- HIPERPARÁMETROS ---
    NUM_EPOCHS = 20        # Número de vueltas
    BATCH_SIZE = 32        # Tamaño del lote
    HIDDEN_UNITS = 64      # Neuronas en capas ocultas
    LEARNING_RATE = 0.001  # Velocidad de aprendizaje

    # Directorios de datos
    train_dir = "../data/train"
    test_dir = "../data/test"

    # Configuración agnóstica del dispositivo (GPU, MPS o CPU)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'Used device: {device}')

    # Definir transformaciones básicas para las imágenes
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),      # Redimensionar a 64x64
        transforms.RandomHorizontalFlip(), # Aumento de datos: volteo horizontal
        transforms.RandomRotation(25),     # Aumento de datos: rotación
        transforms.ToTensor(),             # Convertir a Tensor de PyTorch
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizar pixeles
    ])

    # 1. PREPARAR DATOS (Usando data_setup.py)
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # 2. CONSTRUIR MODELO (Usando model_builder.py)
    model = model_builder.TinyVGG(
        input_shape=3, # 3 canales RGB
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names) # Una salida por cada clase de Pokémon
    ).to(device)

    # 3. DEFINIR PÉRDIDA Y OPTIMIZADOR
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # 4. ENTRENAR (Usando engine.py)
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    # 5. GUARDAR (Usando utils.py)
    utils.save_model(model=model,
                     target_dir="../models",
                     model_name="pokemon_tinyvgg_v0.pth")

if __name__ == '__main__':
    main()