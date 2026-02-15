from diffusers import StableDiffusionPipeline
import torch
import os

# OUTPUT DIRECTORY (your exact path)
OUTPUT_DIR = "/Users/towfeeqrameez/Documents/pranay/image_verifier/dataset/ai"

# Create folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Use GPU if available, else CPU
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Prompts to generate images
prompts = [
    "A realistic photo of a person",
    "A professional DSLR portrait",
    "A natural outdoor photograph",
    "A city street captured on camera",
    "A candid photo taken with a smartphone"
]

# Generate images
for i in range(200):
    image = pipe(prompts[i % len(prompts)]).images[0]
    image.save(f"{OUTPUT_DIR}/ai_{i}.png")
    print(f"Saved: ai_{i}.png")

print("✅ AI image generation completed.")
