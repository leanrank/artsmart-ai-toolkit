import os
import sys
import logging
import yaml
import zipfile
import asyncio
import aiohttp
import aiofiles
import shutil

from .base import app

from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent.parent)

from preprocessing import Preprocessing

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


async def run_cmd(command):
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            shell=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        logger.info(f"[{command!r} exited with {proc.returncode}]")
        if stdout:
            logger.info(f"[stdout]\n{stdout.decode()}")
        if stderr:
            logger.info(f"[stderr]\n{stderr.decode()}")
    except KeyboardInterrupt:
        logger.info("Process interrupted")
        sys.exit(1)


async def download_datasets(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            async with aiofiles.open(os.path.basename(url), mode="wb") as file:
                async for chunk in response.content.iter_chunked(1024):
                    await file.write(chunk)

    return os.path.basename(url)


@app.task(name="train_model")
async def train_model(
    input_images: str,
    trigger_word: str,
    autocaption: bool,
    autocaption_prefix: str,
    autocaption_suffix: str,
    steps: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int,
    lora_name: str,
    model_type: str,
):
    try:
        """Trains a LoRA model on the provided images."""
        logger.info(
            f"Training model with {steps} steps, {batch_size} batch size, {lora_rank} LoRA rank, and {learning_rate} learning rate."
        )

        # Cleanup prev runs
        if Path("output").exists():
            shutil.rmtree("output")

        # Cleanup prev training images
        dataset_dir = "dataset"
        if Path(dataset_dir).exists():
            shutil.rmtree(dataset_dir)

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

        compressed_image_file_path = await download_datasets(input_images)
        if compressed_image_file_path.endswith(".zip"):
            logger.info(f"Unzipping {compressed_image_file_path} to {dataset_dir}")
            os.makedirs(dataset_dir, exist_ok=True)
            with zipfile.ZipFile(compressed_image_file_path, "r") as zip_ref:
                zip_ref.extractall(f"{dataset_dir}/")
        elif compressed_image_file_path.endswith(".tar"):
            logger.info(f"Untarring {compressed_image_file_path} to {dataset_dir}")
            shutil.unpack_archive(compressed_image_file_path, dataset_dir)

        Preprocessing.data_cleaning(data_dir=dataset_dir, convert=True)
        Preprocessing.data_annotation(
            data_dir=dataset_dir,
            custom_token=trigger_word,
            autocaption_prefix=autocaption_prefix,
            autocaption_suffix=autocaption_suffix,
            is_autocaption=autocaption,
        )

        # Run trainer
        if model_type == "schnell":
            await run_cmd(f"python run.py config/lora_flux_schnell.yaml")
        else:
            await run_cmd(f"python run.py config/lora_flux_dev.yaml")

        output_lora = f"output/{lora_name}"

        captions = Preprocessing.find_captions(dataset_dir)
        out_captions = f"{output_lora}/captions"
        os.makedirs(out_captions, exist_ok=True)

        for caption in captions:
            shutil.copy(caption, out_captions)

        output_zip_path = f"/tmp/{lora_name}.zip"
        shutil.make_archive(output_zip_path, "zip", output_lora)

        shutil.rmtree(dataset_dir)
    except Exception as e:
        print(e)
