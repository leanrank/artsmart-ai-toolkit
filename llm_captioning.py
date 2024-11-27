import torch
import re
import base64
import requests
import time
import subprocess

from pathlib import Path
from io import BytesIO
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"

weights = [
    {
        "dest": "liuhaotian/llava-v1.5-13b",
        # git commit hash from huggingface
        "src": "llava-v1.5-13b/006818fc465ebda4c003c0998674d9141d8d95f8",
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model-00001-of-00003.bin",
            "pytorch_model-00002-of-00003.bin",
            "pytorch_model-00003-of-00003.bin",
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ],
    },
    {
        "dest": "openai/clip-vit-large-patch14-336",
        "src": "clip-vit-large-patch14-336/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "files": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
    },
]


def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")


def download_weights(baseurl: str, basedest: str, files: list[str]):
    base_dir = Path(basedest)
    start = time.time()
    print("downloading to: ", base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = base_dir / f
        url = f"{REPLICATE_WEIGHTS_URL}/{baseurl}/{f}"
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix == ".json":
                download_json(url, dest)
            else:
                subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)


def first_lower(s):
    if len(s) == 0:
        return s
    else:
        return s[0].lower() + s[1:]


def extract_response_pairs(text):
    turns = re.split(r"(USER:|ASSISTANT:)", text)[1:]
    turns = [turn.strip() for turn in turns if turn.strip()]
    conv_list = []
    for i in range(0, len(turns[1::2]), 2):
        if i + 1 < len(turns[1::2]):
            conv_list.append(
                [turns[1::2][i].lstrip(":"), turns[1::2][i + 1].lstrip(":")]
            )

    return conv_list


def downscale_image(image_path: str, max_size: int) -> str:
    with Image.open(image_path) as img:
        if img.width and img.height:
            longer_axis = max(img.width, img.height)
            if longer_axis > max_size:
                aspect_ratio = img.width / img.height
                if img.width > img.height:
                    new_width = max_size
                    new_height = int(max_size / aspect_ratio)
                else:
                    new_height = max_size
                    new_width = int(max_size * aspect_ratio)
                img = img.resize((new_width, new_height))

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    return None


class LLMCaptioner:
    def load_models(self):
        for weight in weights:
            download_weights(weight["src"], weight["dest"], weight["files"])
        disable_torch_init()

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                "liuhaotian/llava-v1.5-13b",
                model_name="llava-v1.5-13b",
                model_base="None",
                load_8bit=False,
                load_4bit=False,
            )
        )

    @staticmethod
    def write_caption(caption, dest_dir):
        caption = re.sub(r"\n", " ", caption)
        print(f"Writing caption for {dest_dir}")
        with open(dest_dir, "w") as file:
            file.write(caption)

    def generate_caption(
        self,
        image_path: list[str],
        custom_token: str = None,
        custom_instruction: str = None,
        inherent_attributes: str = None,
        current_caption: str = None,
    ):
        system_prompt = """
        You are an AI assistant that captions images for training purposes. Your task is to create clear, detailed captions.
        """
        if custom_token:
            system_prompt += (
                f' that incorporate the custom token "{custom_token}" at the beginning.'
            )

        system_prompt += """
        The following guide outlines the captioning approach:

        ### Captioning Principles:
        1. **Avoid Making Main Concepts Variable**: Exclude specific traits of the main teaching point to ensure it remains consistent across the dataset.
        2. **Include Detailed Descriptions**: Describe everything except the primary concept being taught.
        3. **Use Generic Classes as Tags**:
        - Broad tags (e.g., "man") can bias the entire class toward the training data.
        - Specific tags (e.g., character name or unique string like "m4n") can reduce impact on the general class while creating strong associations.

        ### Caption Structure:
        1. **Globals**: Rare tokens or uniform tags{f' (e.g., {custom_token})' if custom_token else ''}.
        1.5. **Natural Language Description**: A concise description shorter than a sentence but longer than a tag describing the entire scene.
        2. **Type/Perspective**:
        - Broad description of the image type and perspective (e.g., "photograph," "full body," "from side").
        3. **Action Words**:
        - Verbs describing actions or states (e.g., "sitting," "looking at viewer," "smiling").
        4. **Subject Descriptions**:
        - Detailed descriptions excluding the main teaching concept (e.g., "short brown hair," "pale pink dress").
        5. **Notable Details**:
        - Unique or emphasized elements not classified as background (e.g., "sunlight through windows").
        6. **Background/Location**:
        - Layered background context (e.g., "brown couch," "wooden floor," "refrigerator in background").
        7. **Loose Associations**:
        - Relevant associations or emotions (e.g., "dreary environment").
        Combine all of these to create a detailed caption for the image. Do not include any other text or formatting.
        """

        if inherent_attributes:
            system_prompt += (
                f"\n### Inherent Attributes to Avoid:\n{inherent_attributes}\n"
            )

        if custom_instruction:
            system_prompt += f"\n{custom_instruction}\n"

        user_message = "Here is an image for you to describe. Please describe the image in detail and ensure it adheres to the guidelines. Do not include any uncertainty (i.e. I don't know, appears, seems) or any other text. Focus exclusively on visible elements and not conceptual ones."

        if current_caption:
            user_message += f' The user says this about the image: "{current_caption}". Consider this information while creating your caption, but don\'t simply repeat it. Provide your own detailed description.'

        user_message += " Thank you very much for your help!"

        # conversation = [
        #     {
        #         "role": "assistant",
        #         "content": [
        #             {"type": "text", "text": system_prompt},
        #         ],
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": user_message},
        #             {"type": "image"},
        #         ],
        #     },
        # ]

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        raw_image = Image.open(image_path).convert("RGB")
        image_tensor = (
            self.image_processor.preprocess(raw_image, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )

        input_user_message = DEFAULT_IMAGE_TOKEN + "\n"
        input_user_message += user_message
        conv.append_message(conv.roles[0], input_user_message)
        conv.append_message(conv.roles[1], system_prompt)

        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                top_p=1.0,
                max_new_tokens=512,
                use_cache=True,
            )

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()

            print(f"Caption for {image_path}: {output}")
        try:
            caption = first_lower(str(output))
        except:
            caption = ""

        return caption
