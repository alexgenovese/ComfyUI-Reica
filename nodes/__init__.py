from .storage.google_storage import GCPReadImageNode, GCPWriteImageNode
from .mix.text_image_display_node import ReicaTextImageDisplay
from .api.http_notification import HTTPNotificationNode


# Mappatura dei nodi per ComfyUI
NODE_CLASS_MAPPINGS = {
    "ReicaGCPReadImageNode": GCPReadImageNode,
    "ReicaGCPWriteImageNode": GCPWriteImageNode,
    "ReicaTextImageDisplay": ReicaTextImageDisplay,
    "ReicaHTTPNotification": HTTPNotificationNode,
}

# Nomi visualizzati per i nodi
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReicaGCPReadImageNode": "Reica GCP: Read Image",
    "ReicaGCPWriteImageNode": "Reica GCP: Write Image & Get URL",
    "ReicaTextImageDisplay": "Reica Text Image Display",
    "ReicaFluxImageGenerator": "Flux Image Generator",
    "ReicaHTTPNotification": "Reica API: Send HTTP Notification"
}