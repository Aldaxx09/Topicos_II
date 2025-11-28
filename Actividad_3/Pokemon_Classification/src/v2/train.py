# train.py
import torch
import data_setup, model_builder, utils
from engine import Trainer # Importamos la nueva clase
from torchvision import transforms

def main():
    # Hiperparametros (Ajustados para velocidad)
    NUM_EPOCHS = 15     # Con BatchNorm convergerá más rápido
    BATCH_SIZE = 128    # Aumentar de 32 a 128 aprovecha mejor la GPU
    HIDDEN_UNITS = 32   # Reducir ligeramente si la memoria es problema
    LEARNING_RATE = 0.001

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entrenando en: {device}")

    # Data
    train_dir = "../data/train"
    test_dir = "../data/test"
    
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # Normalizacion estándar para convergencia rápida
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DataLoaders: num_workers=2 suele ser el punto óptimo en local
    train_dl, test_dl, class_names = data_setup.create_dataloaders(
        train_dir=train_dir, test_dir=test_dir, 
        transform=data_transform, batch_size=BATCH_SIZE, num_workers=2
    )

    # Inicializar Modelo Optimizado
    model = model_builder.TinyVGG(
        input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
    )

    # Inicializar Engine (Trainer)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    trainer = Trainer(model, train_dl, test_dl, optimizer, loss_fn, device)

    # Entrenar
    results = trainer.fit(epochs=NUM_EPOCHS)

    # Guardar
    utils.save_model(model=model, target_dir="models", model_name="tinyvgg_v2.pth")

if __name__ == "__main__":
    main()