from time import sleep
from typing import List
from runware import Runware, IImageInference, IImage
from dotenv import load_dotenv
import os
import asyncio
import runware
from runware.server import RunwareServer
import streamlit as st


load_dotenv()

RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
RUNWARE_LOG_LEVEL = os.getenv("RUNWARE_LOG_LEVEL")

# runware = None
# connection_ready = asyncio.Event()


# @st.cache_resource
async def initialize_runware():
    # global runware
    # if runware is None:
    runware = Runware(api_key=RUNWARE_API_KEY, log_level=RUNWARE_LOG_LEVEL)
    await runware.connect()
    # connection_ready.set()
    return runware


# @st.cache_data
async def fetch_images(
    _runware: RunwareServer, prompt: str, number_of_images: int
) -> List[str]:
    # await connection_ready.wait()

    try:
        request_image = IImageInference(
            positivePrompt=prompt,
            model="runware:100@1",
            numberResults=number_of_images,
            # negativePrompt="cloudy, rainy",
            useCache=False,
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
    # asyncio.run(initialize_runware())

    # with st.spinner("Waiting for Runware to connect"):
    #     asyncio.run(connection_ready.wait())

    placeholder_prompt = "anime girl smoking a cigarette with a smug expression"

    prompt_text_box = st.text_area(
        label="Enter your prompt here:", value=placeholder_prompt
    )

    number_of_images_to_create = st.slider("How many images to create", 1, 4)

    if st.button("Submit", type="primary"):
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
