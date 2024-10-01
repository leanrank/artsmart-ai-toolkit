from .base import app


@app.task(name="train_model")
async def train_model():
    try:
        print("Training model")
    except Exception as e:
        print(e)
