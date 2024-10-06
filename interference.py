import torch
from cnn import Cnn
import torchvision.transforms as transforms
from transformers import pipeline
import pickle as pkl
import numpy as np
import random

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="mps",
)

model = Cnn(targets=2, in_size=(32, 32, 1))
model.load_state_dict(torch.load("./models/cnn.pt"))
model.to("mps")
model.eval()

# From here on you can just use the model, inputting the data using torch tensors
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Resize(
            (32, 32)
        ),  # Make the CFMS a bit larger (maybe check this as a hparam)
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), May reenable to improve results
    ]  # Normalize the images
)

## Random test sample (please remove in the real interference)
with open("./data/raw/cfms.pkl", "rb") as file_:
    cfms = pkl.load(file_)

with open("./data/raw/labels.pkl", "rb") as file_:
    labels = pkl.load(file_)

i = random.randint(0, cfms.shape[0])
cfms_img = cfms[i].astype(np.float32)
label = labels[i]  # Valence first then Arausal

# Apply the trainsforms and run through the model, it expects an input of 1,1,32,32
img = transform(cfms_img).unsqueeze(0).to("mps")  # Unsqueeze bc we want b,c,w,h
pred = model(img)

# Setup the prompt input
messages = [
    {
        "role": "system",
        "content": "You are a poem writer that writes poems based on the values level of Valence and Arousal. You will interpret these values and tie an emotion to them, based on that emotion you will write a poem with the emotion in the title.",
    },
    {
        "role": "user",
        "content": f"The Valence: {pred[0][0].item()}, The Arousal: {pred[0][1].item()}",
    },
]

# Write the poem
outputs = pipe(
    messages,
    max_new_tokens=512,
)
with open("./poem.txt", "w") as file_:
    file_.write(outputs[0]["generated_text"][-1]["content"])
