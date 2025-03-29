import os
import sys
import folder_paths
import importlib.util
import datetime
import numpy as np
from PIL import Image, ImageOps
from google.cloud import storage
import torch

# Definizione dei nodi
class GCPWriteImageNode:
    def __init__(self):
        self.compress_level = 4
        self.type = "output"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.normpath(os.path.join(script_dir, "temp"))
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "bucket_name": ("STRING", {"default": "my-bucket"}),
                "bucket_path": ("STRING", {"default": "some/folder"}),
                "file_name": ("STRING", {"default": "my_image.png"}),
                "gcp_service_json": ("STRING", {"default": "/path/to/service_account.json"}),
                "make_public": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "url", "error_message")
    FUNCTION = "store_image_in_gcp"
    OUTPUT_NODE = True
    CATEGORY = "gcp_storage"

    def store_image_in_gcp(self, images, bucket_name, bucket_path, file_name, gcp_service_json, make_public=True):
        error_message = ""
        url = ""
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.isabs(gcp_service_json):
                gcp_service_json = os.path.join(script_dir, gcp_service_json)
            
            if not os.path.exists(gcp_service_json):
                raise FileNotFoundError(f"Credential file not found: {gcp_service_json}")
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_service_json
            
            # Assicuriamoci che la directory esista
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Salviamo l'immagine localmente
            local_file_path = os.path.join(self.output_dir, file_name)
            
            results = list()
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                img.save(local_file_path, pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file_name,
                    "subfolder": "",
                    "type": self.type
                })
            
            # Carichiamo l'immagine su GCP
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"{bucket_path}/{file_name}".strip("/")
            blob = bucket.blob(blob_path)
            
            print(f"Uploading blob to {bucket_name}/{blob_path}..")
            blob.upload_from_filename(local_file_path)
            
            # Rendiamo il blob pubblico se richiesto
            if make_public:
                blob.make_public()
                url = blob.public_url
            else:
                # Generiamo un URL firmato valido per 7 giorni
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(days=7),
                    method="GET"
                )
            
            print(f"Image uploaded successfully. URL: {url}")
            
            # Puliamo il file locale
            os.remove(local_file_path)
            
            return (images, url, error_message)
        
        except Exception as e:
            error_message = f"Upload failed: {str(e)}"
            print(error_message)
            return (images, url, error_message)

class GCPReadImageNode:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.normpath(os.path.join(script_dir, "temp"))
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bucket_name": ("STRING", {"default": "my-bucket"}),
                "bucket_path": ("STRING", {"default": "some/folder"}),
                "file_name": ("STRING", {"default": "my_image.png"}),
                "gcp_service_json": ("STRING", {"default": "/path/to/service_account.json"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "download_from_gcp"
    OUTPUT_NODE = True
    CATEGORY = "gcp_storage"

    def download_from_gcp(self, bucket_name, bucket_path, file_name, gcp_service_json):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.isabs(gcp_service_json):
                gcp_service_json = os.path.join(script_dir, gcp_service_json)
            
            if not os.path.exists(gcp_service_json):
                raise FileNotFoundError(f"Credential file not found: {gcp_service_json}")
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_service_json
            
            local_file_path = os.path.join(self.output_dir, os.path.basename(file_name))
            gcp_path = f"{bucket_path}/{os.path.basename(file_name)}".strip("/")
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(gcp_path)
            
            print(f"[GCP READ] Downloading gs://{bucket_name}/{gcp_path} → {local_file_path}")
            blob.download_to_filename(local_file_path)
            
            i = Image.open(local_file_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            
            if len(image_np.shape) == 3:
                image_np = image_np[None, ...]
            
            image = torch.from_numpy(image_np)
            image = image.contiguous().float()
            
            os.remove(local_file_path)
            
            return (image,)
        
        except Exception as e:
            print(str(e))
            raise e


