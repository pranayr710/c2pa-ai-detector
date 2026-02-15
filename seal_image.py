from utils import embed_seal

input_image = "img/sample_image.jpg"
output_image = "img/sealed_image.png"

hash_value = embed_seal(input_image, output_image)
print("Image sealed with hash:", hash_value)
