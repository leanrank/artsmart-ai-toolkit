import os
import sys
import logging
import yaml
import zipfile
import subprocess

from subprocess import call
from pathlib import Path
from cog import BaseModel, Input, Path

from preprocessing import Preprocessing

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class TrainingOutput(BaseModel):
    weights: Path


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def train(
    input_images: Path = Input(
        description="A zip file containing the images that will be used for training. We recommend a minimum of 10 images. If you include captions, include them as one .txt file per image, e.g. my-photo.jpg should have a caption file named my-photo.txt. If you don't include captions, you can use autocaptioning (enabled by default).  File names must be their captions: a_photo_of_TOK.png, etc. Min 12 images required."
    ),
    trigger_word: str = Input(
        description="The trigger word refers to the object, style or concept you are training on. Pick a string that isn't a real word, like TOK or something related to what's being trained, like CYBRPNK. The trigger word you specify here will be associated with all images during training. Then when you use your LoRA, you can include the trigger word in prompts to help activate the LoRA.",
        default="TOK",
    ),
    autocaption: bool = Input(
        description="Whether to use autocaption. Autocaption for images dataset is more accurate but takes longer to train.",
        default=True,
    ),
    autocaption_prefix: str = Input(
        description="The prefix to use for autocaptioning. This is useful if you want to autocaption a series of images, e.g. 'a photo of TOK, a photo of' or 'a drawing of TOK, a drawing of'. Prefixes help set the right context for your captions, and the captioner will use this prefix as context.",
        default=None,
    ),
    autocaption_suffix: str = Input(
        description="The suffix to use for autocaptioning. This is useful if you want to autocaption a series of images, e.g. 'in the style of TOK'. Suffixes help set the right context for your captions, and the captioner will use this suffix as context.",
        default=None,
    ),
    steps: int = Input(
        description="Number of training steps.", ge=10, le=6000, default=1250
    ),
    learning_rate: float = Input(
        description="Learning rate for the model.", ge=0.00001, le=0.01, default=2e-4
    ),
    batch_size: int = Input(
        description="Batch size for the model.", ge=1, le=16, default=1
    ),
    lora_rank: int = Input(
        description="Supports 16, 32, 64, 128. Higher ranks take longer to train but can capture more complex features. Caption quality is more important for higher ranks.",
        default=16,
    ),
    lora_name: str = Input(
        description="The name of the LoRA to use.",
        default="lora_flux_schnell",
    ),
    model_type: str = Input(
        description="The type of model to use.",
        choices=["schnell", "dev"],
        default="schnell",
    ),
) -> TrainingOutput:
    """Trains a LoRA model on the provided images."""
    logger.info(
        f"Training model with {steps} steps, {batch_size} batch size, {lora_rank} LoRA rank, and {learning_rate} learning rate."
    )

    # Cleanup prev runs
    os.system("rm -rf output")

    # Cleanup prev training images
    dataset_dir = "dataset"
    os.system(f"rm -rf {dataset_dir}")

    if model_type == "schnell":
        config_path = Path("config/lora_flux_schnell.yaml")
    else:
        config_path = Path("config/lora_flux_dev.yaml")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    config["config"]["name"] = lora_name
    config["config"]["process"][0]["train"]["steps"] = steps
    config["config"]["process"][0]["save"]["save_every"] = steps + 1
    # config["config"]["process"][0]["sample"]["sample_every"] = steps
    config["config"]["process"][0]["train"]["lr"] = learning_rate
    config["config"]["process"][0]["train"]["batch_size"] = batch_size
    config["config"]["process"][0]["network"]["linear"] = lora_rank
    config["config"]["process"][0]["network"]["linear_alpha"] = lora_rank
    config["config"]["process"][0]["trigger_word"] = trigger_word
    config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_dir

    with config_path.open("w") as f:
        yaml.dump(config, f)

    compressed_image_file_path = str(input_images)
    if compressed_image_file_path.endswith(".zip"):
        logger.info(f"Unzipping {compressed_image_file_path} to {dataset_dir}")
        os.makedirs(dataset_dir, exist_ok=True)
        with zipfile.ZipFile(compressed_image_file_path, "r") as zip_ref:
            zip_ref.extractall(f"{dataset_dir}/")
    elif compressed_image_file_path.endswith(".tar"):
        logger.info(f"Untarring {compressed_image_file_path} to {dataset_dir}")
        os.system(f"tar -xvf {compressed_image_file_path} -C {dataset_dir}")

    Preprocessing.data_cleaning(data_dir=dataset_dir, convert=True)
    if autocaption:
        Preprocessing.data_annotation(
            data_dir=dataset_dir,
            custom_token=trigger_word,
            autocaption_prefix=autocaption_prefix,
            autocaption_suffix=autocaption_suffix,
        )

    # Run trainer
    if model_type == "schnell":
        run_cmd(f"python run.py config/lora_flux_schnell.yaml")
    else:
        run_cmd(f"python run.py config/lora_flux_dev.yaml")

    output_lora = f"output/{lora_name}"

    captions = Preprocessing.find_captions(dataset_dir)
    out_captions = f"{output_lora}/captions"
    os.makedirs(out_captions, exist_ok=True)

    for caption in captions:
        os.system(f"cp {caption} {out_captions}")

    output_zip_path = f"/tmp/{lora_name}.zip"
    os.system(f"zip -r {output_zip_path} {output_lora}")

    os.system(f"rm -rf {dataset_dir}")

    return TrainingOutput(weights=Path(output_zip_path))


# if __name__ == "__main__":
#     import pprint

#     config_path = "config/lora_flux_schnell.yaml"
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#     pprint.pprint(config["config"]["process"][0])
