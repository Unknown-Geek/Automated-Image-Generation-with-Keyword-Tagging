# Stable Diffusion Image Generation with Keyword Tagging

This Python script uses the Stable Diffusion XL model from the Hugging Face Diffusers library to generate images based on text prompts. Additionally, it utilizes the Google Generative AI Gemini model to generate relevant keywords for each prompt, which are then added as IPTC metadata to the generated images.

## Prerequisites

Before running the script, ensure that you have the following:

1. Python 3.7 or later installed on your system.
2. A Google Cloud Platform account with the Gemini API enabled and an API key generated. Replace `YOUR_GEMINI_API_KEY` in the code with your actual Gemini API key.
3. Google Colab or a local environment with CUDA support for GPU acceleration (recommended for faster image generation).

## Installation

1. Install the required Python packages by running the following command:
!pip install diffusers torch accelerate clip pyexiv2
Copy code
2. If running on Google Colab, mount your Google Drive:

python

from google.colab import drive
drive.mount('/content/drive')

## Usage 

Prepare a text file containing the prompts for image generation, with one prompt per line.
Upload the text file to the script when prompted.
The script will generate num_images (default is 5) for each prompt and save them in the /content/drive/MyDrive/Automation/CET_Images/ folder on your Google Drive.
The generated images will have relevant keywords added as IPTC metadata, which can be useful for image search and organization.

## Customization

You can customize the following parameters in the script:
num_images: The number of images to generate for each prompt (default is 5).
image_folder: The path to the folder where the generated images will be saved (default is /content/drive/MyDrive/Automation/CET_Images/).

## Note

The script assumes that you have a CUDA-enabled GPU available. If you're running on a CPU, the image generation process will be significantly slower.
The script requires a stable internet connection to generate images and fetch keywords from the Gemini model.
Generating high-resolution images may consume significant computational resources and time.
