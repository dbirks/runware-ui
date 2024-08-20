from runware import Runware, IImageInference
from dotenv import load_dotenv
import os
import asyncio


load_dotenv()

RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
RUNWARE_LOG_LEVEL = os.getenv("RUNWARE_LOG_LEVEL")


async def fetch_images() -> list:
    try:
        runware = Runware(api_key=RUNWARE_API_KEY, log_level=RUNWARE_LOG_LEVEL)
        await runware.connect()

        request_image = IImageInference(
            positivePrompt="a beautiful sunset over the mountains",
            model="civitai:36520@76907",
            numberResults=4,
            negativePrompt="cloudy, rainy",
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
    try:
        images = asyncio.run(fetch_images())
        for image in images:
            print(f"Image: {image.imageURL}")
    except Exception as e:
        print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    main()
