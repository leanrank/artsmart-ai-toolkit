import os
import gc
import torch
import random
import concurrent.futures

from tqdm import tqdm
from PIL import Image

from llm_captioning import generate_caption, write_caption


class Preprocessing:
    @classmethod
    def data_cleaning(
        cls,
        data_dir: str,
        convert: bool = False,
        batch_size: int = 32,
        is_recursive: bool = False,
        use_random_color: bool = False,
    ):
        cls.clean_directory(directory=data_dir)
        images = cls.find_images(directory=data_dir)
        num_batches = len(images) // batch_size + 1

        if convert:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Prepare arguments for each function call
                tasks = []
                for i in range(num_batches):
                    start = i * batch_size
                    end = start + batch_size
                    batch = images[start:end]
                    for image_path in batch:
                        tasks.append((image_path, use_random_color))

                # Use map with the list of tuples as arguments
                for _ in tqdm(
                    executor.map(lambda p: cls.process_image(*p), tasks),
                    total=len(tasks),
                ):
                    pass

            print("All images have been converted")

    @staticmethod
    def clean_directory(directory):
        supported_types = [
            ".png",
            ".jpg",
            ".jpeg",
            ".txt",
        ]

        for item in os.listdir(directory):
            file_path = os.path.join(directory, item)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(item)[1]
                if file_ext not in supported_types:
                    print(f"Deleting file {item} from {directory}")
                    os.remove(file_path)

    @staticmethod
    def process_image(image_path, use_random_color):
        background_colors = [
            (255, 255, 255),
            (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        img = Image.open(image_path)
        img_dir, image_name = os.path.split(image_path)

        if img.mode in ("RGBA", "LA"):
            if use_random_color:
                background_color = random.choice(background_colors)
            else:
                background_color = (255, 255, 255)
            bg = Image.new("RGB", img.size, background_color)
            bg.paste(img, mask=img.split()[-1])

            if image_name.endswith(".webp"):
                bg = bg.convert("RGB")
                new_image_path = os.path.join(
                    img_dir, image_name.replace(".webp", ".jpg")
                )
                bg.save(new_image_path, "JPEG")
                os.remove(image_path)
                print(
                    f" Converted image: {image_name} to {os.path.basename(new_image_path)}"
                )
            else:
                bg.save(image_path, "PNG")
                print(f" Converted image: {image_name}")
        else:
            if image_name.endswith(".webp"):
                new_image_path = os.path.join(
                    img_dir, image_name.replace(".webp", ".jpg")
                )
                img.save(new_image_path, "JPEG")
                os.remove(image_path)
                print(
                    f" Converted image: {image_name} to {os.path.basename(new_image_path)}"
                )
            else:
                img.save(image_path, "PNG")

    @staticmethod
    def find_images(directory):
        images = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".png") or file.endswith(".webp"):
                    images.append(os.path.join(root, file))
        return images

    @staticmethod
    def data_annotation(cls, data_dir: str, custom_token: str = "TOK"):
        images_path = cls.find_images(data_dir)
        for image_path in images_path:
            caption = generate_caption(image_path, custom_token=custom_token)
            file_name = os.path.splitext(image_path)[0]
            caption_file_path = f"{file_name}.txt"
            write_caption(caption, caption_file_path)

        gc.collect()
        torch.cuda.empty_cache()