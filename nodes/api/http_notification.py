import requests
import json
import os

class HTTPNotificationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint_url": ("STRING", {"default": "http://tryonyou-endpoint.vercel.app/api/update-generation"}),
                "image_url": ("STRING", {"default": ""}),
                "user_id": ("STRING", {"default": ""}),
                "product_sku": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "error_message", "display_text")
    FUNCTION = "send_notification"
    CATEGORY = "api"

    def send_notification(self, endpoint_url, image_url, user_id, product_sku):
        response = ""
        error_message = ""
        # Initialize display_text with content to ensure it's never empty
        display_text = f"Processing notification...\nEndpoint: {endpoint_url}\nUser ID: {user_id}\nProduct SKU: {product_sku}"
        
        try:
            # Prepare payload
            payload = {
                "image_url": image_url,
                "user_id": user_id,
                "product_sku": product_sku
            }
            
            # Update display_text before making request
            display_text = f"Sending notification to:\n{endpoint_url}\n\nPayload:\n{json.dumps(payload, indent=2)}"
            
            # Make HTTP POST request
            headers = {'Content-Type': 'application/json'}
            r = requests.post(endpoint_url, data=json.dumps(payload), headers=headers, timeout=30)
            
            # Get response
            r.raise_for_status()  # Raise exception for non-2xx response
            response = r.text
            
            print(f"Notification sent successfully to {endpoint_url}")
            
            # Format the display text with successful response - ensure it's a plain string
            display_text = f"✅ Notification sent successfully\n\nEndpoint: {endpoint_url}\n\nStatus: {r.status_code}\n\nUser ID: {user_id}\nProduct SKU: {product_sku}\n\nResponse:\n{response[:200]}"
            if len(response) > 200:
                display_text += "...[truncated]"
                
            return (response, error_message, display_text)
            
        except Exception as e:
            error_message = f"Failed to send notification: {str(e)}"
            print(error_message)
            
            # Format the display text with error information - ensure it's a plain string
            display_text = f"❌ Notification failed\n\nEndpoint: {endpoint_url}\n\nUser ID: {user_id}\nProduct SKU: {product_sku}\n\nError: {str(e)}"
            
            return (response, error_message, display_text)
