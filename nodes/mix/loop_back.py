import os
import numpy as np
import torch
import requests
from PIL import Image, ImageOps
from io import BytesIO
import hashlib

class LoadLoopImagesFromURLs:
    last_index = 0  # Class attribute to track the last accessed image

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines input types for loading images from URLs.
        URLs should be provided one per line in the text input.
        """
        return {
            "required": {
                "urls_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "URL", "FILENAME")
    OUTPUT_IS_LIST = (False, False, False, False)  # Only return one image at a time

    FUNCTION = "load_images"

    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Tracks changes and cycles through URLs.
        """
        if kwargs.get("load_always", False):
            return float("NaN")

        urls_text = kwargs.get("urls_text", "")
        urls = cls._parse_urls(urls_text)
        
        if not urls:
            return hash(frozenset(kwargs.items()))

        # Cycle to next URL
        cls.last_index = (cls.last_index + 1) % len(urls)

        return hash((cls.last_index, frozenset(kwargs.items())))

    @classmethod
    def _parse_urls(cls, urls_text):
        """
        Parse URLs from text input, filtering out empty lines and comments.
        """
        if not urls_text.strip():
            return []
        
        urls = []
        for line in urls_text.strip().split('\n'):
            line = line.strip()
            # Skip empty lines and comments (lines starting with #)
            if line and not line.startswith('#'):
                urls.append(line)
        
        return urls

    @classmethod
    def _get_filename_from_url(cls, url):
        """
        Extract filename from URL or generate one if not available.
        """
        try:
            # Remove query parameters and fragments
            clean_url = url.split('?')[0].split('#')[0]
            # Extract filename from path
            filename = os.path.basename(clean_url)
            
            # If no filename or extension, generate one
            if not filename or '.' not in filename:
                # Create a hash-based filename
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"image_{url_hash}.jpg"
            
            return filename
        except Exception:
            # Fallback filename
            return f"image_{cls.last_index + 1}.jpg"

    @classmethod
    def _download_image(cls, url):
        """
        Download image from URL with error handling.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")
            image_data = BytesIO(response.content)
            img = Image.open(image_data)
            return img
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to download image from {url}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to process image from {url}: {str(e)}")

    def load_images(self, urls_text: str, load_always=False):
        """
        Load images from URLs, cycling through them one at a time.
        
        :param urls_text: String containing URLs, one per line
        :param load_always: Boolean flag for always loading
        :return: image, mask, current URL, and filename
        """
        urls = self._parse_urls(urls_text)
        
        if not urls:
            raise ValueError("No valid URLs provided. Please enter URLs one per line.")

        # Get current URL based on last_index
        current_url = urls[self.last_index % len(urls)]
        
        # Extract filename from URL
        filename = self._get_filename_from_url(current_url)
        
        print(f"Loading image {self.last_index + 1} of {len(urls)}: {filename} from {current_url}")
        
        # Download and process image
        try:
            img = self._download_image(current_url)
            img = ImageOps.exif_transpose(img)
            
            # Convert to RGB
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]  # Add batch dimension

            # Load mask (if alpha channel exists)
            if 'A' in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
                mask = mask[None,]  # Add batch dimension
            else:
                # Create empty mask with same dimensions as image
                h, w = image.shape[1:3]
                mask = torch.zeros((1, h, w), dtype=torch.float32, device="cpu")

            return image, mask, current_url, filename
            
        except Exception as e:
            # If current URL fails, try to continue with a placeholder or raise error
            print(f"Error loading image from {current_url}: {str(e)}")
            raise RuntimeError(f"Failed to load image from URL: {current_url}\nError: {str(e)}")

# Alternative version that skips failed URLs instead of stopping
class LoadLoopImagesFromURLsSkipErrors(LoadLoopImagesFromURLs):
    """
    Version that skips failed URLs instead of stopping execution
    """
    
    def load_images(self, urls_text: str, load_always=False):
        """
        Load images from URLs, skipping failed ones and continuing with the next.
        """
        urls = self._parse_urls(urls_text)
        
        if not urls:
            raise ValueError("No valid URLs provided. Please enter URLs one per line.")

        max_attempts = len(urls)  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            current_url = urls[self.last_index % len(urls)]
            
            # Extract filename from URL
            filename = self._get_filename_from_url(current_url)
            
            try:
                print(f"Loading image {self.last_index + 1} of {len(urls)}: {filename} from {current_url}")
                
                img = self._download_image(current_url)
                img = ImageOps.exif_transpose(img)
                
                # Convert to RGB
                image = img.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]  # Add batch dimension

                # Load mask (if alpha channel exists)
                if 'A' in img.getbands():
                    mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1.0 - torch.from_numpy(mask)
                    mask = mask[None,]  # Add batch dimension
                else:
                    # Create empty mask with same dimensions as image
                    h, w = image.shape[1:3]
                    mask = torch.zeros((1, h, w), dtype=torch.float32, device="cpu")

                return image, mask, current_url, filename
                
            except Exception as e:
                print(f"Skipping failed URL {current_url} ({filename}): {str(e)}")
                self.last_index = (self.last_index + 1) % len(urls)
                attempts += 1
                continue
        
        raise RuntimeError("All URLs failed to load. Please check your URLs and internet connection.")

