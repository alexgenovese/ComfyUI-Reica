import os, sys

# Installazione automatica delle dipendenze
def install_dependencies():
    try:
        import google.cloud.storage
    except ImportError:
        print("Installazione delle dipendenze per GCP Storage...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-storage"])
        print("Dipendenze installate con successo!")

# Esegui l'installazione delle dipendenze quando il modulo viene caricato
install_dependencies()

# Import nodes after installing dependencies
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

