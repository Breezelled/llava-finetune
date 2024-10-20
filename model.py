from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "llava-next-8b"  # Replace with the actual model name if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

input_text = "Explain the process of phase transitions in neural networks."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

from PIL import Image
import requests

# Load an image
img_url = "https://example.com/path/to/image.jpg"
image = Image.open(requests.get(img_url, stream=True).raw)

# Encode image + text and pass them to the model (requires vision support)
# vision_model = SomeVisionModel(...)
# inputs = vision_model(image, text_input)
# outputs = model.generate(inputs)