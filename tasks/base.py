import os

from datetime import timedelta

from samping.driver.sqs import SQSDriver
from samping.app import App
from samping.backoff import Exponential
from samping.routes import Rule


def driver_factory():
    return SQSDriver(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
        batch_window=os.environ.get("SQS_BATCH_WINDOW", 5),
        backoff=Exponential(max_elapsed_time=timedelta(seconds=30)),
        batch_size=os.environ.get("SQS_BATCH_SIZE", 10),
        prefetch_size=os.environ.get("SQS_QUEUE_PREFETCH_SIZE", 50),
        endpoint_url=os.environ.get("SQS_ENDPOINT"),
        use_ssl=os.environ.get("SQS_USE_SSL"),
        visibility_timeout=os.environ.get("SQS_VISIBILITY_TIMEOUT", 3600),
    )


app = App(
    driver_factory,
    task_timeout=os.environ.get("TASK_TIMEOUT", 3600),
    queue_size=os.environ.get("QUEUE_SIZE", 100),
    default_queue=os.environ.get("SQS_QUEUE_NAME"),
)


app.routes = [
    # prefix task with inprocess_ to execute it in the same proces
    Rule("train_*", os.environ.get("SQS_IN_PROCESS_QUEUE_NAME")),
]
