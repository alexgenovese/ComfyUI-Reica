import torch
import numpy as np

class ReicaTextImageDisplay:
    """
    A node that displays text images or URLs passed as input.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "show_text"  # This matches the method name below
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def show_text(self, text):
        # Method name now matches FUNCTION attribute
        return {"ui": {"text": text}, "result": (text,)}

    @classmethod
    def IS_DISPLAY(cls):
        return True
