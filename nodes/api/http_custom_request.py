import requests
import json
from io import BytesIO
from PIL import Image

class HTTPCustomRequestNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint_url": ("STRING", {"default": "http://tryonyou-endpoint.vercel.app/api/update-generation"}),
                "json_body": ("STRING", {
                    "multiline": True, 
                    "default": """{\n    "key": "value"\n}""",
                    "placeholder": "Enter JSON body here..."
                }),
                "images": ("STRING", {"default": "https://example.com/image1.png,https://example.com/image2.png"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "data", "error_message", "display_text")
    FUNCTION = "send_custom_request"
    CATEGORY = "api"

    def send_custom_request(self, endpoint_url, json_body, images):
        response = ""
        error_message = ""
        data = ""
        display_text = f"Processing custom request...\nEndpoint: {endpoint_url}"
        
        try:
            # Validate input
            if not isinstance(images, str):
                raise ValueError("Images input must be a string of comma-separated URLs")
                
            urls = [url.strip() for url in images.split(",") if url.strip()]
            if not urls:
                raise ValueError("No valid URLs provided in 'images' input")

            # Parse JSON body
            payload = json.loads(json_body)
            payload["output_results"] = payload.get("output_results", {})
            payload["output_results"]["urls_generated"] = urls

            # Update display text
            display_text = f"Sending request to:\n{endpoint_url}\n\nPayload:\n{json.dumps(payload, indent=2)}"
            data = json.dumps(payload)

            # Make HTTP POST request
            headers = {
                'Content-Type': 'application/json',
                'Origin': 'http://localhost:3001'
            }
            r = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            response = r.text

            # Format success display text
            display_text = f"✅ Request sent successfully\n\nEndpoint: {endpoint_url}\n\nStatus: {r.status_code}\n\nResponse:\n{response[:200]}"
            if len(response) > 200:
                display_text += "...[truncated]"
            
            error_message = "No errors"
            
            return (response, data, error_message, display_text)
            
        except Exception as e:
            error_message = f"Failed to send request: {str(e)}"
            display_text = f"❌ Request failed\n\nEndpoint: {endpoint_url}\n\nError: {str(e)}"
            return ("", data, error_message, display_text)