from .storage.google_storage import GCPReadImageNode, GCPWriteImageNode
from .mix.text_image_display_node import ReicaTextImageDisplay
from .mix.insert_anything import InsertAnythingNode
from .mix.url_image_loader_filename import URLImageLoaderFilename
from .tryoffdiff.nodes import TryOffDiffNode, TryOffDiffLoaderNode
from .api.http_notification import HTTPNotificationNode
from .api.http_custom_request import HTTPCustomRequestNode
from .mix.loop_back import LoadLoopImagesFromURLs, LoadLoopImagesFromURLsSkipErrors
from .mix.smart_resizer import SmartResizer


# Mappatura dei nodi per ComfyUI
NODE_CLASS_MAPPINGS = {
    "ReicaGCPReadImageNode": GCPReadImageNode,
    "ReicaGCPWriteImageNode": GCPWriteImageNode,
    "ReicaTextImageDisplay": ReicaTextImageDisplay,
    "ReicaURLImageLoader": URLImageLoaderFilename,
    "ReicaHTTPNotification": HTTPNotificationNode,
    "ReicaHTTPCustomRequest": HTTPCustomRequestNode,
    "ReicaInsertAnythingNode": InsertAnythingNode,
    "ReicaTryOffDiffLoader": TryOffDiffLoaderNode,
    "ReicaTryOffDiffGenerator": TryOffDiffNode,
    "ReicaLoadLoopImagesFromURLs": LoadLoopImagesFromURLs,
    "ReicaLoadLoopImagesFromURLsSkipErrors": LoadLoopImagesFromURLsSkipErrors,
    "ReicaSmartResizer": SmartResizer,
}

# Nomi visualizzati per i nodi
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReicaGCPReadImageNode": "Reica GCP: Read Image",
    "ReicaGCPWriteImageNode": "Reica GCP: Write Image & Get URL",
    "ReicaTextImageDisplay": "Reica Text Image Display",
    "ReicaReadImageUrl": "Reica Read Image URL",
    "ReicaURLImageLoader": "Reica URL Image Loader Filename",
    "ReicaHTTPNotification": "Reica API: Send HTTP Notification",
    "ReicaHTTPCustomRequest": "Reica API: Send Custom Request",
    "ReicaInsertAnythingNode": "Reica Insert Anything",
    "ReicaTryOffDiffLoader": "Reica TryOffDiff Model Loader",
    "ReicaTryOffDiffGenerator": "Reica TryOffDiff Generator",
    "ReicaLoadLoopImagesFromURLs": "Reica Load Loop Images From URLs",
    "ReicaLoadLoopImagesFromURLsSkipErrors": "Reica Load Loop Images From URLs (Skip Errors)",
    "ReicaSmartResizer": "Reica Smart Resizer",
}