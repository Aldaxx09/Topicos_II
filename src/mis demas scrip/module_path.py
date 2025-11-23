# module_path.py

from pathlib import Path
import os

def get_project_root() -> Path:
    """
    Encuentra la raíz del proyecto asumiendo que este script
    está en un directorio 'src'.
    """
    # Path de este archivo -> .../TuProyecto/src/module_path.py
    # .parent -> .../TuProyecto/src
    if Path(__file__).parent.name == 'src':
        # .parent.parent -> .../TuProyecto/
        return Path(__file__).parent.parent
    else:
        # Si no, asumimos que estamos en la raíz
        return Path.cwd()

def train_data_path() -> Path:
    """
    Devuelve la ruta al archivo de entrenamiento.
    :return: la ruta al archivo training.csv
    """
    root = get_project_root()
    data_file = root / "data" / "training.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Archivo no encontrado en {data_file}. Asegúrate de que la carpeta 'data' esté en la raíz del proyecto.")
    return data_file

def test_data_path() -> Path:
    """
    Devuelve la ruta al archivo de prueba.
    :return: la ruta al archivo test.csv
    """
    root = get_project_root()
    data_file = root / "data" / "test.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Archivo no encontrado en {data_file}. Asegúrate de que la carpeta 'data' esté en la raíz del proyecto.")
    return data_file