from typing import List
from runware import Runware, IImageInference
from dotenv import load_dotenv
import os
import asyncio
from runware.server import RunwareServer
import streamlit as st


load_dotenv()

RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
RUNWARE_LOG_LEVEL = os.getenv("RUNWARE_LOG_LEVEL")


async def initialize_runware():
    runware = Runware(api_key=RUNWARE_API_KEY, log_level=RUNWARE_LOG_LEVEL)
    await runware.connect()
    return runware


async def fetch_images(
    _runware: RunwareServer, prompt: str, number_of_images: int
) -> List[str]:

    try:
        request_image = IImageInference(
            positivePrompt=prompt,
            model="runware:100@1",
            # model="urn:air:flux1:checkpoint:civitai:618692@691639",
            # model="civitai:618692@691639",
            # lora=["civitai:67941@72606"], # 80s
            # model="civitai:54233@125985", # ghibli backgrounds
            numberResults=number_of_images,
            # negativePrompt="cloudy, rainy",
            # useCache=False,
            height=512,
            width=512,
        )
        images = await _runware.imageInference(requestImage=request_image)
        image_urls = [image.imageURL for image in images]
        return image_urls
    except Exception as e:
        print(f"An error occurred in fetch_images: {e}")
        return []


async def main():
    st.title("Runware UI")

    runware = await initialize_runware()

    form = st.form(key="submit_form")

    placeholder_prompt = "superman t-rex"

    prompt_text_box = form.text_area(
        label="Enter your prompt here:", value=placeholder_prompt
    )

    number_of_images_to_create = form.slider(
        label="How many images to create", min_value=1, max_value=4
    )

    submit = form.form_submit_button(label="Submit")

    if submit:
        try:
            image_urls = await fetch_images(
                runware, prompt_text_box, number_of_images_to_create
            )

            for image_url in image_urls:
                print(f"Received image URL: {image_url}")
                st.image(image_url, use_column_width=True)
        except Exception as e:
            print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
