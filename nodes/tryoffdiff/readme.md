# TryOffDiff ComfyUI Custom Node

Un nodo personalizzato per ComfyUI che implementa TryOffDiff, un modello di diffusione per la generazione di virtual try-off di abbigliamento.

## Caratteristiche

- **4 tipi di modello**: Upper-Body, Lower-Body, Dress, e Multi-Garment
- **Generazione ad alta qualità**: Utilizza modelli di diffusione per risultati realistici
- **Integrazione nativa ComfyUI**: Compatibile con il workflow di ComfyUI
- **Controllo parametri avanzato**: Seed, guidance scale, steps di inferenza personalizzabili

## Installazione

1. Clona questo repository nella cartella `custom_nodes` di ComfyUI:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/comfyui-tryoffdiff
```

2. Installa le dipendenze:
```bash
cd comfyui-tryoffdiff
pip install -r requirements.txt
```

3. Riavvia ComfyUI

## Struttura dei File

```
comfyui-tryoffdiff/
├── __init__.py          # Registrazione nodi
├── nodes.py             # Implementazione nodi principali
├── model.py             # Definizioni modelli TryOffDiff
├── requirements.txt     # Dipendenze Python
└── README.md           # Documentazione
```

## Utilizzo

### 1. TryOffDiff Model Loader

Il nodo `TryOffDiffLoader` carica il modello e tutti i componenti necessari:

**Input:**
- `model_type`: Tipo di modello (upper/lower/dress/multi)
- `repo_id`: Repository Hugging Face (default: "rizavelioglu/tryoffdiff")
- `force_download`: Forza il download del modello

**Output:**
- `pipeline`: Pipeline completa del modello

### 2. TryOffDiff Generator

Il nodo `TryOffDiffGenerator` genera l'immagine virtual try-off:

**Input:**
- `pipeline`: Pipeline dal loader
- `reference_image`: Immagine di riferimento (formato ComfyUI)
- `seed`: Seed per la generazione (default: 42)
- `guidance_scale`: Scala di guidance (1.0-5.0, default: 2.0)
- `num_inference_steps`: Numero di step (5-1000, default: 20)
- `garment_types`: Tipi di abbigliamento (solo per modello multi)

**Output:**
- `generated_image`: Immagine generata

## Workflow di Esempio

1. **Load Image** → Carica l'immagine di riferimento
2. **TryOffDiff Model Loader** → Carica il modello desiderato
3. **TryOffDiff Generator** → Genera l'immagine virtual try-off
4. **Save Image** → Salva il risultato

## Tipi di Modello

- **Upper-Body**: Genera parti superiori (magliette, giacche, ecc.)
- **Lower-Body**: Genera parti inferiori (pantaloni, gonne, ecc.)
- **Dress**: Genera vestiti completi
- **Multi-Garment**: Genera combinazioni di abbigliamento

## Requisiti di Sistema

- CUDA compatible GPU (raccomandato)
- Almeno 8GB VRAM
- Python 3.8+
- ComfyUI installato

## Dipendenze Principali

- torch
- torchvision
- diffusers
- transformers
- huggingface-hub
- PIL (Pillow)

## Crediti

Basato sul lavoro di ricerca:
- [TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models](https://rizavelioglu.github.io/tryoffdiff)
- Repository originale: [rizavelioglu/tryoffdiff](https://github.com/rizavelioglu/tryoffdiff)

## Licenza

Questo nodo segue la licenza del progetto originale TryOffDiff.