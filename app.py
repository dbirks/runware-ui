from runware import Runware, IImageInference
from dotenv import load_dotenv
import os
import asyncio
import streamlit as st


load_dotenv()

RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
RUNWARE_LOG_LEVEL = os.getenv("RUNWARE_LOG_LEVEL")


async def fetch_images() -> list:
    try:
        runware = Runware(api_key=RUNWARE_API_KEY, log_level=RUNWARE_LOG_LEVEL)
        await runware.connect()

        request_image = IImageInference(
            positivePrompt="""
            a group of settlers looking out over a vast plain, 
            wondering where the next meal will come from, storm on the horizon, 
            and for some reason they're all wearing bright cyan shirts with pink polka dots
            """,
            model="runware:100@1",
            numberResults=1,
            # negativePrompt="cloudy, rainy",
            useCache=False,
            height=512,
            width=512,
        )

        images = await runware.imageInference(requestImage=request_image)
        return images
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def main():
    print("Hello world")

    st.title("Runware UI")

    try:
        images = asyncio.run(fetch_images())
        for image in images:
            print(f"Image: {image.imageURL}")
            st.image(image.imageURL)
    except Exception as e:
        print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    main()
