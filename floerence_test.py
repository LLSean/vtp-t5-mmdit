import requests
from PIL import Image
from modelscope import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

img_url = './danbooru_surtr_solo_100k/1.jpg' 
raw_image = Image.open(img_url).convert('RGB')

question = "Describe this image."
inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())