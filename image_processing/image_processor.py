import json
from PIL import Image, ImageOps, ImageFilter

from pathlib import Path
from functools import cache


class ImageProcessingDispatcher:
    @staticmethod
    def dispatch(image, optype, **params): ...


class PILDispatcher(ImageProcessingDispatcher):
    # grayscale: ImageOps.grayscale(image)
    # flip_horizontal: ImageOps.mirror(image)
    # flip_vertical: ImageOps.flip(image)
    # scale: ImageOps.scale(image, factor), including its default bicubic resampling
    # blur: image.filter(ImageFilter.BoxBlur(radius))
    # rotate: image.rotate(angle), using Pillow's defaults, including expand=False
    @staticmethod
    def dispatch(image, optype, **params):
        match optype:
            case "grayscale":
                return ImageOps.grayscale(image)
            case "flip_horizontal":
                return ImageOps.mirror(image)
            case "flip_vertical":
                return ImageOps.flip(image)
            case "scale":
                return ImageOps.scale(image, factor=params.get("factor", 0.5))
            case "blur":
                return image.filter(ImageFilter.BoxBlur(radius=params.get("factor", 0)))
            case "rotate":
                return image.rotate(angle=params.get("factor", 0))
            case _:
                return image


@cache
def transform_config(transform_file):
    with open(transform_file) as tf:
        return json.load(tf)


def transform_image(image_file, transform_file, output_file):
    xforms = transform_config(transform_file)
    with Image.open(image_file) as img:
        for transform in xforms.get("transformations", []):
            img = PILDispatcher.dispatch(img, transform.get("type"), **transform)
        odir = Path(output_file)
        odir.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_file)
