import requests
import json
import os

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
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "error_message", "display_text")
    FUNCTION = "send_custom_request"
    CATEGORY = "api"

    def send_custom_request(self, endpoint_url, json_body):
        response = ""
        error_message = ""
        display_text = f"Processing custom request...\nEndpoint: {endpoint_url}"
        
        try:
            # Parse JSON body
            try:
                payload = json.loads(json_body)
            except json.JSONDecodeError as je:
                raise ValueError(f"Invalid JSON format: {str(je)}")
            
            # Update display text
            display_text = f"Sending request to:\n{endpoint_url}\n\nPayload:\n{json.dumps(payload, indent=2)}"
            
            # Make HTTP POST request
            headers = {
                'Content-Type': 'application/json',
                'Origin': 'http://localhost:3001'
            }
            r = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
            
            # Get response
            r.raise_for_status()
            response = r.text
            
            # Format success display text
            display_text = f"✅ Request sent successfully\n\nEndpoint: {endpoint_url}\n\nStatus: {r.status_code}\n\nResponse:\n{response[:200]}"
            if len(response) > 200:
                display_text += "...[truncated]"
            
            error_message = "No errors"
            
            return (response, error_message, display_text)
            
        except Exception as e:
            error_message = f"Failed to send request: {str(e)}"
            display_text = f"❌ Request failed\n\nEndpoint: {endpoint_url}\n\nError: {str(e)}"
            
            return (response, error_message, display_text)
