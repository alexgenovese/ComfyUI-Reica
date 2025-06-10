import torch, requests, os
from PIL import Image
import numpy as np
from io import BytesIO
from urllib.parse import urlparse

class URLImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("STRING", {
                    "multiline": True,
                    "default": "https://example.com/image1.jpg\nhttps://example.com/image2.png"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filenames")
    FUNCTION = "load_images_from_urls"
    CATEGORY = "image/io"
    
    def load_images_from_urls(self, urls):
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
        
        images = []
        filenames = []
        
        for url in url_list:
            try:
                # Scarica l'immagine
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Estrai il nome del file dall'URL
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename or '.' not in filename:
                    filename = f"image_{len(filenames)}.jpg"
                
                # Converti in PIL Image
                image = Image.open(BytesIO(response.content))
                
                # Converti in RGB se necessario
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Converti in numpy array e poi in tensor
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                
                images.append(image_tensor)
                filenames.append(filename)
                
            except Exception as e:
                print(f"Errore nel caricamento dell'immagine da {url}: {str(e)}")
                continue
        
        if not images:
            # Se nessuna immagine è stata caricata, crea un'immagine vuota
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, ["no_image.jpg"])
        
        # Concatena tutte le immagini in un batch
        batch_images = torch.cat(images, dim=0)
        
        return (batch_images, filenames)

