import torch
import re
import base64

from io import BytesIO

from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    AutoProcessor,
)


model_id = "llava-hf/llava-1.5-13b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_id)


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


def write_caption(caption, dest_dir):
    caption = re.sub(r"\n", " ", caption)
    print(f"Writing caption for {dest_dir}")
    with open(dest_dir, "w") as file:
        file.write(caption)


def generate_caption(
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
        system_prompt += f' that incorporate the custom token "Photo of {custom_token} man or woman" if is person else "Photo of {custom_token}" the beginning.'
        system_prompt += """
    Make it natural and blend in with the entire caption.
    Example:
    - Photo of TOK woman sitting on a beige couch. She is wearing a dark blue matching outfit with a geometric...
    - Photo of TOK man stands outdoors on a paved pathway with a grass patch and flowerbed beside him...
    - Photo of TOK standing, smiling and looking at the camera...
    """

    system_prompt += """
    The following guide outlines the captioning approach:

    ### Captioning Principles:
    1. **Avoid Making Main Concepts Variable**: Exclude specific traits of the main teaching point to ensure it remains consistent across the dataset.
    2. **Include Detailed Descriptions**: Describe everything except the primary concept being taught.
    3. **Use Generic Classes as Tags**:
    - Broad tags (e.g., f' (e.g., Photo of {custom_token})' if custom_token else 'man',f' (e.g., Photo of {custom_token})' if custom_token else 'woman') can bias the entire class toward the training data.
    - Specific tags (e.g., character name or unique string like "m4n") can reduce impact on the general class while creating strong associations.

    ### Caption Structure:
    1. **Globals**: Rare tokens or uniform tags{f' (e.g., Photo of {custom_token})' if custom_token else ''}.
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
        system_prompt += f"\n### Inherent Attributes to Avoid:\n{inherent_attributes}\n"

    if custom_instruction:
        system_prompt += f"\n{custom_instruction}\n"

    user_message = "Here is an image for you to describe. Please describe the image in detail and ensure it adheres to the guidelines. Do not include any uncertainty (i.e. I don't know, appears, seems) or any other text. Focus exclusively on visible elements and not conceptual ones."

    if current_caption:
        user_message += f' The user says this about the image: "{current_caption}". Consider this information while creating your caption, but don\'t simply repeat it. Provide your own detailed description.'

    user_message += " Thank you very much for your help!"

    conversation = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image"},
            ],
        },
    ]

    raw_image = Image.open(image_path)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(
        0, torch.float16
    )
    output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    text = processor.decode(output[0][2:], skip_special_tokens=True)
    res = extract_response_pairs(text)
    caption = res[0][1]

    return caption
