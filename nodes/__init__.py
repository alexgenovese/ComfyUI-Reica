from .storage.google_storage import GCPReadImageNode, GCPWriteImageNode
from .mix.text_image_display_node import ReicaTextImageDisplay
from .mix.read_image_url import ReicaReadImageUrl
from .api.http_notification import HTTPNotificationNode


# Mappatura dei nodi per ComfyUI
NODE_CLASS_MAPPINGS = {
    "ReicaGCPReadImageNode": GCPReadImageNode,
    "ReicaGCPWriteImageNode": GCPWriteImageNode,
    "ReicaTextImageDisplay": ReicaTextImageDisplay,
    "ReicaReadImageUrl": ReicaReadImageUrl,
    "ReicaHTTPNotification": HTTPNotificationNode,
}

# Nomi visualizzati per i nodi
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReicaGCPReadImageNode": "Reica GCP: Read Image",
    "ReicaGCPWriteImageNode": "Reica GCP: Write Image & Get URL",
    "ReicaTextImageDisplay": "Reica Text Image Display",
    "ReicaReadImageUrl": "Reica Read Image URL",
    "ReicaHTTPNotification": "Reica API: Send HTTP Notification"
}