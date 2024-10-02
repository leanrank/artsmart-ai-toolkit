from tasks.base import app


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
        print("Training model")
    except Exception as e:
        print(e)
