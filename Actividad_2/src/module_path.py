# module_path.py
# Importar librerias estandar
from pathlib import Path

def train_data_path() -> Path:
    # Devuelve la ruta al archivo de entrenamiento.
    cwd= Path("..")
    for folder in (cwd, cwd/"..", cwd/".."/".."):
        data_file = folder / "data/training.csv"
        if data_file.exists() and data_file.is_file():
            print("Datos de entrenamiento encontrados en", data_file)
            return data_file
        else:
            raise Exception("Datos no encontrados")


def test_data_path() -> Path:
    # Devuelve la ruta al archivo de prueba.
    cwd= Path("..")
    for folder in (cwd, cwd/"..", cwd/".."/".."):
        data_file = folder / "data/test.csv"
        if data_file.exists() and data_file.is_file():
            print("Datos de prueba encontrados en", data_file)
            return data_file
        else:
            raise Exception("Datos no encontrados")
        
        
# if __name__ == "__main__":
#     train_data_path()
#     test_data_path()