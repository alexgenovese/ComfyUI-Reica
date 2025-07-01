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
                "file_names": ("STRING", {"default": "image1.png,image2.png"}),
                "gcp_service_json": ("STRING", {"default": "/path/to/service_account.json"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls", "error_messages")
    FUNCTION = "store_image_in_gcp"
    OUTPUT_NODE = True
    CATEGORY = "gcp_storage"

    def store_image_in_gcp(self, images, bucket_name, bucket_path, file_names, gcp_service_json):
        error_messages = []
        urls = []
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.isabs(gcp_service_json):
                gcp_service_json = os.path.join(script_dir, gcp_service_json)
            
            if not os.path.exists(gcp_service_json):
                raise FileNotFoundError(f"Credential file not found: {gcp_service_json}")
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_service_json
            
            # Handle both single and multiple images
            images_count = 1 if len(images.shape) == 3 else len(images)
            
            # Accept file_names as either a string or a list
            if isinstance(file_names, str):
                file_names_array = [name.strip() for name in file_names.split(",") if name.strip()]
            elif isinstance(file_names, list):
                file_names_array = [str(name).strip() for name in file_names if str(name).strip()]
            else:
                raise ValueError("file_names must be a string or a list of strings")
            
            # Strict validation of matching counts
            if len(file_names_array) != images_count:
                raise ValueError(f"Mismatch between number of images ({images_count}) and filenames ({len(file_names_array)}). Please provide exactly one filename per image.")
            
            # Convert single image to list format if needed
            images_list = [images] if images_count == 1 else images
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            # Process each image with its corresponding filename
            for idx, (image, filename) in enumerate(zip(images_list, file_names_array)):
                try:
                    i = image.cpu().numpy()
                    print(f"[DEBUG] Original image shape: {i.shape}, dtype: {i.dtype}")
                    # Rimuovi dimensioni batch/canale extra
                    while i.ndim > 3:
                        i = i[0]
                    print(f"[DEBUG] Shape for PIL: {i.shape}, dtype: {i.dtype}")
                    i = 255. * i
                    # Forza il tipo a uint8 e la shape a (H, W, 3)
                    i = np.clip(i, 0, 255).astype(np.uint8)
                    if i.ndim == 2:  # grayscale
                        img = Image.fromarray(i, mode="L")
                    elif i.ndim == 3 and i.shape[2] == 3:
                        img = Image.fromarray(i, mode="RGB")
                    elif i.ndim == 3 and i.shape[2] == 4:
                        img = Image.fromarray(i, mode="RGBA")
                    else:
                        raise ValueError(f"Unsupported image shape for PIL: {i.shape}")
                    
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG', compress_level=self.compress_level)
                    img_byte_arr.seek(0)
                    
                    blob_path = f"{bucket_path}/{filename}".strip("/")
                    blob = bucket.blob(blob_path)
                    
                    print(f"Uploading blob to {bucket_name}/{blob_path}..")
                    blob.upload_from_file(img_byte_arr, content_type='image/png')
                    
                    urls.append(blob.public_url)
                    error_messages.append("")
                    
                except Exception as e:
                    error_msg = f"Failed to upload image {idx} ({filename}): {str(e)}"
                    error_messages.append(error_msg)
                    urls.append("")
                    print(error_msg)
            
            return (images, ",".join(urls), ",".join(error_messages))
            
        except Exception as e:
            error_msg = f"General upload failure: {str(e)}"
            print(error_msg)
            return (images, "", error_msg)

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


