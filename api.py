import os
import uvicorn
import aioboto3
import pynvml
import asyncio

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tasks.train import train_model
from fastapi.routing import APIRouter


class TrainRequest(BaseModel):
    input_images: str = Field(
        description="A zip file containing the images that will be used for training. We recommend a minimum of 10 images. If you include captions, include them as one .txt file per image, e.g. my-photo.jpg should have a caption file named my-photo.txt. If you don't include captions, you can use autocaptioning (enabled by default).  File names must be their captions: a_photo_of_TOK.png, etc. Min 12 images required."
    )
    trigger_word: str = Field(
        description="The trigger word refers to the object, style or concept you are training on. Pick a string that isn't a real word, like TOK or something related to what's being trained, like CYBRPNK. The trigger word you specify here will be associated with all images during training. Then when you use your LoRA, you can include the trigger word in prompts to help activate the LoRA.",
        default="TOK",
    )
    autocaption: bool = Field(
        description="Whether to use autocaption. Autocaption for images dataset is more accurate but takes longer to train.",
        default=False,
    )
    autocaption_prefix: str = Field(
        description="The prefix to use for autocaptioning. This is useful if you want to autocaption a series of images, e.g. 'a photo of TOK, a photo of' or 'a drawing of TOK, a drawing of'. Prefixes help set the right context for your captions, and the captioner will use this prefix as context.",
        default=None,
    )
    autocaption_suffix: str = Field(
        description="The suffix to use for autocaptioning. This is useful if you want to autocaption a series of images, e.g. 'in the style of TOK'. Suffixes help set the right context for your captions, and the captioner will use this suffix as context.",
        default=None,
    )
    steps: int = (
        Field(description="Number of training steps.", ge=10, le=6000, default=1250),
    )
    learning_rate: float = Field(
        description="Learning rate for the model.",
        ge=0.00001,
        le=0.01,
        default=2e-4,
    )
    batch_size: int = Field(
        description="Batch size for the model.", ge=1, le=16, default=1
    )
    lora_rank: int = Field(
        description="Supports 16, 32, 64, 128. Higher ranks take longer to train but can capture more complex features. Caption quality is more important for higher ranks.",
        default=16,
    )
    lora_name: str = Field(
        description="The name of the LoRA to use.",
        default="lora_flux_schnell",
    )
    model_type: str = Field(
        description="The type of model to use.",
        default="schnell",
    )
    webhook_url: str = Field(
        description="The webhook URL to send the training status to.",
        default=None,
    )


router = APIRouter()


async def check_process(command):
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()

    if proc.returncode == 0:
        return stdout.decode().strip()  # Return decoded standard output
    else:
        return None  # Handle the case if the command fails


async def check_gpu_processes():
    loop = asyncio.get_event_loop()

    def sync_check():
        # Initialize NVML
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        processes_running = False

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            print(f"Checking GPU {i}...")

            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if processes:
                    processes_running = True
                    print(f"GPU {i} has {len(processes)} process(es) running.")
                    for process in processes:
                        print(
                            f"Process ID: {process.pid}, Memory Usage: {process.usedGpuMemory / 1024**2:.2f} MB"
                        )
                else:
                    print(f"  GPU {i} has no processes running.")
            except pynvml.NVMLError as err:
                print(f"  Failed to get processes for GPU {i}: {str(err)}")

        # Shutdown NVML
        pynvml.nvmlShutdown()

        return processes_running

    return await loop.run_in_executor(None, sync_check)


async def check_cpu_processes():
    process_running = await check_process(["pgrep", "-f", "run.py"])
    return process_running is not None


@router.get("/health-check")
async def health_check():
    return JSONResponse(content={"message": "OK"})


@router.get("/check-running-processes")
async def check_running_processes():
    cpu_processes_running = await check_cpu_processes()
    gpu_processes_running = await check_gpu_processes()

    return JSONResponse(
        content={
            "process_running": gpu_processes_running or cpu_processes_running,
        }
    )


@router.post("/train")
async def train(request: TrainRequest):
    await train_model.apply(
        kwargs={
            "input_images": request.input_images,
            "trigger_word": request.trigger_word,
            "autocaption": request.autocaption,
            "autocaption_prefix": request.autocaption_prefix,
            "autocaption_suffix": request.autocaption_suffix,
            "steps": request.steps,
            "learning_rate": request.learning_rate,
            "batch_size": request.batch_size,
            "lora_rank": request.lora_rank,
            "lora_name": request.lora_name,
            "model_type": request.model_type,
            "webhook_url": request.webhook_url,
        }
    )

    return JSONResponse(content={"message": "Training started"})


@router.delete("/delete-model/{model_name}")
async def delete_model(model_name: str):
    session = aioboto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION_NAME"),
    )
    object_key = f"public/loras/{model_name}.safetensors"
    async with session.client("s3") as s3:
        object_list = await s3.list_objects_v2(
            Bucket=os.environ.get("AWS_BUCKET_NAME"), Prefix=object_key
        )
        if object_list["KeyCount"] == 0:
            return JSONResponse(content={"message": "Model not found"}, status_code=404)

        await s3.delete_object(Bucket=os.environ.get("AWS_BUCKET_NAME"), Key=object_key)

    return JSONResponse(content={"message": "Model deleted"})


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
