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
    runware: RunwareServer,
    positive_prompt: str,
    number_of_images: int,
    # negative_prompt: str,
    height: int,
    width: int,
) -> List[str]:

    try:
        request_image = IImageInference(
            positivePrompt=positive_prompt,
            model="runware:100@1",
            # model="urn:air:flux1:checkpoint:civitai:618692@691639",
            # model="civitai:618692@691639",
            # lora=["civitai:67941@72606"], # 80s
            # model="civitai:54233@125985", # ghibli backgrounds
            numberResults=number_of_images,
            # negativePrompt=negative_prompt,
            # useCache=False,
            height=height,
            width=width,
        )
        images = await runware.imageInference(requestImage=request_image)
        image_urls = [image.imageURL for image in images]
        return image_urls
    except Exception as e:
        print(f"An error occurred in fetch_images: {e}")
        return []


async def main():
    st.set_page_config(page_title="Runware UI", layout="wide")

    st.title("Runware UI")

    runware = await initialize_runware()

    col1, col2 = st.columns(2)

    placeholder_prompt = "90s manga picture of a couple walking along a path next to a stream, enjoying the seasons"

    with col1:

        form = st.form(key="submit_form")

        positive_prompt_text_box = form.text_area(
            label="Enter your prompt here:", value=placeholder_prompt
        )

        # negative_prompt_text_box = form.text_area(
        #     label="Negative prompt here (optional):"
        # )

        number_of_images_to_create = form.slider(
            label="How many images to create", min_value=1, max_value=10
        )

        # size_of_images = form.selectbox(
        #     "Choose a size", ["512x512", "1024x1024", "2048x2048"]
        # )

        # width = int(size_of_images.split("x")[0])
        # height = int(size_of_images.split("x")[1])

        submit = form.form_submit_button(label="Submit")

    with col2:

        if submit:
            try:
                image_urls = await fetch_images(
                    runware=runware,
                    positive_prompt=positive_prompt_text_box,
                    number_of_images=number_of_images_to_create,
                    # negative_prompt=negative_prompt_text_box,
                    height=1024,
                    width=1024,
                )
            except Exception as e:
                print(f"An error occurred in main: {e}")
            st.session_state["image_urls"] = image_urls

        # for image_url in image_urls:
        #     print(f"Received image URL: {image_url}")
        #     st.image(image_url, use_column_width=True)
        if "image_urls" in st.session_state:
            image_urls = st.session_state["image_urls"]
            st.image(image_urls, use_column_width=True)


if __name__ == "__main__":
    asyncio.run(main())
