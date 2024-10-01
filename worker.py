import asyncio
import logging
import argparse

from tasks import task_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_worker(queues: str, num_workers: int):
    registered_tasks = "\n".join([key for key in task_app._tasks])
    logger.info(f"Registered tasks: {registered_tasks}")
    logger.info(f"Starting {num_workers} workers for queues: {queues}")

    try:
        asyncio.run(task_app.run_worker(queues=queues, num_worker=num_workers))
    except Exception as e:
        logger.error(f"Error starting worker: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", help="queue to process", default="default")
    parser.add_argument("--num-worker", type=int, default=3)
    args = parser.parse_args()
    run_worker(args.queue, args.num_worker)
