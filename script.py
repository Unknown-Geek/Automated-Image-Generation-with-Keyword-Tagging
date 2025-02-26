!pip install diffusers torch accelerate clip pyexiv2

import pkg_resources
REQUIRED_PACKAGES = ['diffusers','torch','accelerate','clip','pyexiv2']

for package in REQUIRED_PACKAGES:
   try:
       dist = pkg_resources.get_distribution(package)
       print('{} ({}) is installed'.format(dist.key, dist.version))
   except pkg_resources.DistributionNotFound:
       print('{} is NOT installed'.format(package))

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)

from IPython.display import clear_output
clear_output()

if torch.cuda.is_available():
   pipeline.to("cuda")
   pipeline.enable_attention_slicing()
   print("Using CUDA")
else:
   print("Using CPU")

import os
import csv
import pyexiv2
from PIL import Image
from google.colab import drive, files
import google.generativeai as genai

# Number of images per prompt
num_images = 5

# Gemini API Key
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"

# Configure Google Generative AI
genai.configure(api_key = GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Mount Google Drive
drive.mount('/content/drive')
!rm -rf /content/drive/MyDrive/Automation/Images/

# Upload text file containing prompts
uploaded = files.upload()

# Get the uploaded file name
uploaded_filename = next(iter(uploaded), None)

# Extract prompts from file
with open(uploaded_filename, 'r') as f:
   prompts = f.read().splitlines()

# Check if prompts list is not empty
if not prompts:
   raise ValueError("Prompts list is empty.")

#Gemini
def get_gemini_response(question):
   response = model.generate_content(question)
   return response.text


# Path to the 'Images' folder in your Google Drive
image_folder = "/content/drive/MyDrive/Automation/Images/"

# Create the 'Images' folder if it doesn't exist
if not os.path.exists(image_folder):
   os.makedirs(image_folder)

# Iterate over each prompt
for prompt in prompts:
 prompt = prompt.replace('.','')

 #Generate keywords
 question = f"Generate atleast 20 relevant, simple and most-searched keywords in a single line each seperated by comma as delimiter for the given prompt. Make sure the keywords are single words with no special characters in them. Generate the keywords taking the prompt as reference : {prompt}"
 keywords = get_gemini_response(question)
 keywords = keywords.split(',')
 keywords = [keyword.strip() for keyword in keywords]
 keywords.append('_ai_generated')
 print(prompt)
 print(keywords)

 for i in range(num_images):
   try:
     # Generate image
     image = pipeline(prompt).images[0]
     image_path = os.path.join(image_folder, f"image-{prompt.replace(' ', '-')}-{i}.png")
     width, height = image.size
     image = image.resize((width * 4, height * 4), resample=Image.BICUBIC)

     # Convert image to JPG before saving
     jpg_path = os.path.join(image_folder, f"image-{prompt.replace(' ', '-')}-{i}.jpg")
     jpg_path = jpg_path
     image.save(jpg_path, format="JPEG")

     #add keywords to file
     image = pyexiv2.Image(jpg_path)
     image.modify_iptc({'Iptc.Application2.Keywords': keywords})
     image.close()

   except Exception as e:
     print(f"Error generating image for prompt {prompt}: {e}")
     continue

print("Images generated and keywords added  successfully!")