# {
#     "created_at": "2024-10-16T12:40:32.23Z",
#     "data_removed": false,
#     "error": null,
#     "id": "qy2dts8twsrj20cjjnc87atzd0",
#     "input": {
#         "autocaption": true,
#         "autocaption_prefix": "A photo of TOK",
#         "batch_size": 1,
#         "input_images": "https://artsmart-storage-bucket-v2.s3.amazonaws.com/public/zip/high_steps-inputs.zip",
#         "learning_rate": 0.0002,
#         "lora_name": "high_steps",
#         "lora_rank": 128,
#         "model_type": "dev",
#         "steps": 10,
#         "trigger_word": "TOK",
#     },
#     "logs": "",
#     "model": "smartartdev/artsmart-flux-trainer",
#     "output": null,
#     "started_at": "2024-10-16T12:46:40.156272412Z",
#     "status": "processing",
#     "urls": {
#         "cancel": "https://api.replicate.com/v1/predictions/qy2dts8twsrj20cjjnc87atzd0/cancel",
#         "get": "https://api.replicate.com/v1/predictions/qy2dts8twsrj20cjjnc87atzd0",
#     },
#     "version": "1521cfb274804aedd5aaeb0c4b2ee2cf0698a86967db5dbaed79b53392ed9c0f",
#     "webhook": "https://webhook.site/79cb4bbc-3033-46d9-945f-b09a003301b0",
#     "webhook_events_filter": ["start", "completed"],
# }


# {
#     "completed_at": "2024-10-16T12:49:31.018764799Z",
#     "created_at": "2024-10-16T12:40:32.23Z",
#     "data_removed": false,
#     "error": null,
#     "id": "qy2dts8twsrj20cjjnc87atzd0",
#     "input": {
#         "autocaption": true,
#         "autocaption_prefix": "A photo of TOK",
#         "batch_size": 1,
#         "input_images": "https://artsmart-storage-bucket-v2.s3.amazonaws.com/public/zip/high_steps-inputs.zip",
#         "learning_rate": 0.0002,
#         "lora_name": "high_steps",
#         "lora_rank": 128,
#         "model_type": "dev",
#         "steps": 10,
#         "trigger_word": "TOK",
#     },
#     "logs": "get it from_sub_process",
#     "metrics": {"predict_time": 170.862492429},
#     "model": "smartartdev/artsmart-flux-trainer",
#     "output": {
#         "version": "smartartdev/flux-schnell-avatar:d75a7e1050973cc9ff6106a204ac968690d0eae9ec12760bffdfa3c395d4aa29",
#         "weights": "https://replicate.delivery/yhqm/Y9w24EI0XuKYOlcQYD6oFK9n3wK559m5QLPWfKAn96VtQrzJA/high_steps.zip",
#     },
#     "started_at": "2024-10-16T12:46:40.156272412Z",
#     "status": "succeeded",
#     "urls": {
#         "cancel": "https://api.replicate.com/v1/predictions/qy2dts8twsrj20cjjnc87atzd0/cancel",
#         "get": "https://api.replicate.com/v1/predictions/qy2dts8twsrj20cjjnc87atzd0",
#     },
#     "version": "1521cfb274804aedd5aaeb0c4b2ee2cf0698a86967db5dbaed79b53392ed9c0f",
#     "webhook": "https://webhook.site/79cb4bbc-3033-46d9-945f-b09a003301b0",
#     "webhook_events_filter": ["start", "completed"],
# }


from pydantic import BaseModel, AnyHttpUrl
from typing import Optional, List


class InputData(BaseModel):
    autocaption: bool
    autocaption_prefix: str
    batch_size: int
    input_images: AnyHttpUrl
    learning_rate: float
    lora_name: str
    lora_rank: int
    model_type: str
    steps: int
    trigger_word: str


class PredictionSchema(BaseModel):
    created_at: str
    error: Optional[str]
    id: str
    input: InputData
    logs: Optional[str]
    metrics: Optional[dict]
    model: str
    output: Optional[dict]
    started_at: str
    status: str
    webhook: AnyHttpUrl
