import os
from PIL import Image
import torch

# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast

pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer, AutoModelForImageTextToText

tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model = AutoModelForImageTextToText.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")