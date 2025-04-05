import os
import sys
import folder_paths
import importlib.util
import datetime
import numpy as np
from PIL import Image, ImageOps
from google.cloud import storage
import torch
from io import BytesIO

# Definizione dei nodi
class GCPWriteImageNode:
    def __init__(self):
        self.compress_level = 4
        self.type = "output"
        # No need for temp directory

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "bucket_name": ("STRING", {"default": "my-bucket"}),
                "bucket_path": ("STRING", {"default": "some/folder"}),
                "file_name": ("STRING", {"default": "my_image.png"}),
                "gcp_service_json": ("STRING", {"default": "/path/to/service_account.json"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "url", "error_message")
    FUNCTION = "store_image_in_gcp"
    OUTPUT_NODE = True
    CATEGORY = "gcp_storage"

    def store_image_in_gcp(self, images, bucket_name, bucket_path, file_name, gcp_service_json):
        error_message = ""
        url = ""
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.isabs(gcp_service_json):
                gcp_service_json = os.path.join(script_dir, gcp_service_json)
            
            if not os.path.exists(gcp_service_json):
                raise FileNotFoundError(f"Credential file not found: {gcp_service_json}")
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_service_json
            
            # Process the first image (you could modify to handle multiple images differently)
            image = images[0]
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Instead of saving to file, use BytesIO to keep in memory
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG', compress_level=self.compress_level)
            img_byte_arr.seek(0)  # Go to the start of the BytesIO object
            
            # Carichiamo l'immagine su GCP direttamente dalla memoria
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"{bucket_path}/{file_name}".strip("/")
            blob = bucket.blob(blob_path)
            
            print(f"Uploading blob to {bucket_name}/{blob_path}..")
            blob.upload_from_file(img_byte_arr, content_type='image/png')
            
            # Rendiamo il blob pubblico se richiesto
            url = blob.public_url
            
            print(f"Image uploaded successfully. URL: {url}")
            
            return (images, url, error_message)
        
        except Exception as e:
            error_message = f"Upload failed: {str(e)}"
            print(error_message)
            return (images, url, error_message)

class GCPReadImageNode:
    def __init__(self):
        pass  # No need for temp directory

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
            
            gcp_path = f"{bucket_path}/{os.path.basename(file_name)}".strip("/")
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(gcp_path)
            
            print(f"[GCP READ] Downloading gs://{bucket_name}/{gcp_path}")
            
            # Download into memory using BytesIO
            img_bytes = BytesIO()
            blob.download_to_file(img_bytes)
            img_bytes.seek(0)  # Go to start of the BytesIO object
            
            i = Image.open(img_bytes)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            
            if len(image_np.shape) == 3:
                image_np = image_np[None, ...]
            
            image = torch.from_numpy(image_np)
            image = image.contiguous().float()
            
            return (image,)
        
        except Exception as e:
            print(str(e))
            raise e


