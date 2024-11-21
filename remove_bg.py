import torch

from PIL import Image
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
from carvekit.ml.files.models_loc import download_all


def remove_exif_single(image: Image.Image) -> Image.Image:

    # Create a new image without EXIF
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)

    return image_without_exif


class RemoveBackground:
    def __init__(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        download_all()
        self.interface = init_interface(
            MLConfig(
                segmentation_network="tracer_b7",
                preprocessing_method="none",
                postprocessing_method="fba",
                seg_mask_size=640,
                trimap_dilation=30,
                trimap_erosion=5,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

    @staticmethod
    def create_canvas(image: Image.Image, canvas_size: tuple[int, int]) -> Image.Image:
        canvas_image = Image.new("RGB", canvas_size, (255, 255, 255))

        image_width, image_height = image.size
        canvas_width, canvas_height = canvas_size

        if canvas_width < image_width or canvas_height < image_height:
            scale_factor = min(canvas_width / image_width, canvas_height / image_height)
            new_width = int(image_width * scale_factor)
            new_height = int(image_height * scale_factor)
            image = image.resize((new_width, new_height))
            image_width, image_height = new_width, new_height

        # x_offset = (canvas_width - image_width) // 2
        # y_offset = (canvas_height - image_height) // 2

        canvas_image.paste(image, (0, 0), image)

        return canvas_image

    def __call__(
        self, images: list[Image.Image], canvas_size: tuple[int, int] = (940, 788)
    ) -> list[Image.Image]:
        """Run a single prediction on the model"""
        for index, image in enumerate(images):
            images[index] = remove_exif_single(image)
        processed_bg = self.interface(images)
        for index, image in enumerate(processed_bg):
            canvas_image = self.create_canvas(image, canvas_size)
            processed_bg[index] = canvas_image
        return processed_bg


# if __name__ == "__main__":
#     import time

#     t1 = time.perf_counter()
#     remove_bg = RemoveBackground()
#     print(f"Time taken to initialize: {time.perf_counter() - t1}")
# image_paths = [
#     "C:/Users/shabr/Downloads/flx67342fea4e61f-inputs/heads67342fead122e.jpg",
#     "C:/Users/shabr/Downloads/flx67342fea4e61f-inputs/upper_body67342feb8642f.jpg",
#     "C:/Users/shabr/Downloads/flx67342fea4e61f-inputs/upper_body67342fecc6fd9.jpg",
# ]
# images = [Image.open(image_path).r for image_path in image_paths]
# remove_bg_images = remove_bg(images)
# for index, image in enumerate(remove_bg_images):
#     image.save(f"output_{index}.png")
