from .storage.google_storage import GCPReadImageNode, GCPWriteImageNode
from .mix.text_image_display_node import ReicaTextImageDisplay
from .mix.read_image_url import ReicaReadImageUrl
from .mix.insert_anything import InsertAnythingNode
from .mix.url_image_loader_filename import URLImageLoaderFilename
from .api.http_notification import HTTPNotificationNode


# Mappatura dei nodi per ComfyUI
NODE_CLASS_MAPPINGS = {
    "ReicaGCPReadImageNode": GCPReadImageNode,
    "ReicaGCPWriteImageNode": GCPWriteImageNode,
    "ReicaTextImageDisplay": ReicaTextImageDisplay,
    "ReicaReadImageUrl": ReicaReadImageUrl,
    "ReicaURLImageLoader": URLImageLoaderFilename,
    "ReicaHTTPNotification": HTTPNotificationNode,
    "InsertAnythingNode": InsertAnythingNode
}

# Nomi visualizzati per i nodi
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReicaGCPReadImageNode": "Reica GCP: Read Image",
    "ReicaGCPWriteImageNode": "Reica GCP: Write Image & Get URL",
    "ReicaTextImageDisplay": "Reica Text Image Display",
    "ReicaReadImageUrl": "Reica Read Image URL",
    "ReicaURLImageLoader": "Reica URL Image Loader Filename",
    "ReicaHTTPNotification": "Reica API: Send HTTP Notification",
    "InsertAnythingNode": "Insert Anything"
}