from typing import Callable

from utils.path_utils import get_output_path, walk_files
from itertools import product

from concurrent.futures import ThreadPoolExecutor

from image_processor import transform_image

image_files = {"*.jpg", "*.jpeg", "*.png"}
xform_files = {"*.json"}


def process_images(
    image_dir: str,
    transformation_dir: str,
    get_output_path: Callable[[str, str], str],
) -> None:
    images = walk_files(image_dir, image_files)
    xforms = walk_files(transformation_dir, xform_files)
    ops = (
        (i, t, "/var/tmp/" + get_output_path(i, t)) for i, t in product(images, xforms)
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(lambda p: transform_image(*p), ops):
            pass


if __name__ == "__main__":
    process_images("images", "transforms", get_output_path)
