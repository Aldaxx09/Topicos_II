"""Versión optimizada del entrenamiento - CORREGIDA"""
import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms
import time

def main():
    # --- HIPERPARÁMETROS OPTIMIZADOS ---
    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    HIDDEN_UNITS = 128
    LEARNING_RATE = 0.001
    PATIENCE = 5

    # Directorios de datos
    train_dir = "../data/train"
    test_dir = "../data/test"

    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Usando dispositivo: {device}')
    
    # Mejorar transformaciones
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. PREPARAR DATOS
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    print(f"Número de clases: {len(class_names)}")
    print(f"Tamaño del dataset de entrenamiento: {len(train_dataloader.dataset)}")
    print(f"Tamaño del dataset de prueba: {len(test_dataloader.dataset)}")

    # 2. CONSTRUIR MODELO MEJORADO
    # Si hay problemas con el modelo mejorado, podemos usar el original primero
    try:
        model = model_builder.ImprovedTinyVGG(
            input_shape=3,
            hidden_units=HIDDEN_UNITS,
            output_shape=len(class_names)
        ).to(device)
    except:
        print("Usando modelo TinyVGG original...")
        model = model_builder.TinyVGG(
            input_shape=3,
            hidden_units=HIDDEN_UNITS,
            output_shape=len(class_names)
        ).to(device)

    print(f"Modelo creado con {sum(p.numel() for p in model.parameters()):,} parámetros")

    # 3. DEFINIR PÉRDIDA Y OPTIMIZADOR MEJORADO
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Scheduler CORREGIDO - sin parámetro verbose
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # 4. ENTRENAMIENTO CON EARLY STOPPING
    start_time = time.time()
    
    best_acc = 0.0
    patience_counter = 0
    current_lr = LEARNING_RATE
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Entrenar
        train_loss, train_acc = engine.train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        # Evaluar
        test_loss, test_acc = engine.test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        # Ajustar learning rate
        scheduler.step(test_loss)
        
        # Verificar si el learning rate cambió
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"⚠️ Learning rate reducido a {new_lr:.6f}")
            current_lr = new_lr
        
        epoch_time = time.time() - epoch_start
        
        print(
            f"Epoch: {epoch+1:02d} | "
            f"Tiempo: {epoch_time:.2f}s | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        
        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            # Guardar mejor modelo
            utils.save_model(
                model=model,
                target_dir="../models",
                model_name="pokemon_best_model.pth"
            )
            print(f"💾 Mejor modelo guardado con accuracy: {test_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping en epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\n✅ Entrenamiento completado en {total_time/60:.2f} minutos")
    print(f"🏆 Mejor precisión: {best_acc:.4f}")

if __name__ == '__main__':
    main()