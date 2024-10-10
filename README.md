# CLIP-ONNX
It is a simple library to speed up CLIP inference up to 3x (K80 GPU)!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lednik7/CLIP-ONNX/blob/main/examples/readme_example.ipynb)
Open AI CLIP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lednik7/CLIP-ONNX/blob/main/examples/RuCLIP_onnx_example.ipynb)
RuCLIP Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lednik7/CLIP-ONNX/blob/main/examples/ru_CLIP_tiny_onnx.ipynb)
RuCLIP tiny Example

## Usage
Install clip-onnx module and requirements first. Use this trick
```python3
!pip install git+https://github.com/Lednik7/CLIP-ONNX.git
!pip install git+https://github.com/openai/CLIP.git
!pip install onnxruntime-gpu
```
## Example in 3 steps
0. Download CLIP image from repo
```python3
!wget -c -O CLIP.png https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true
```
1. Load standard CLIP model, image, text on cpu
```python3
import clip
from PIL import Image
import numpy as np

# onnx cannot work with cuda
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

# batch first
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).cpu() # [1, 3, 224, 224]
image_onnx = image.detach().cpu().numpy().astype(np.float32)

# batch first
text = clip.tokenize(["a diagram", "a dog", "a cat"]).cpu() # [3, 77]
text_onnx = text.detach().cpu().numpy().astype(np.int32)
```
2. Create CLIP-ONNX object to convert model to onnx
```python3
from clip_onnx import clip_onnx

visual_path = "clip_visual.onnx"
textual_path = "clip_textual.onnx"

onnx_model = clip_onnx(model, visual_path=visual_path, textual_path=textual_path)
onnx_model.convert2onnx(image, text, verbose=True)
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_model.start_sessions(providers=["CPUExecutionProvider"]) # cpu mode
```
3. Use for standard CLIP API. Batch inference
```python3
image_features = onnx_model.encode_image(image_onnx)
text_features = onnx_model.encode_text(text_onnx)

logits_per_image, logits_per_text = onnx_model(image_onnx, text_onnx)
probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421067 0.00299571]]
```

**Enjoy the speed**

## Load saved model
Example for ViT-B/32 from Model Zoo
```python3
!wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/visual.onnx
!wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx
```
```python3
onnx_model = clip_onnx(None)
onnx_model.load_onnx(visual_path="visual.onnx",
                     textual_path="textual.onnx",
                     logit_scale=100.0000) # model.logit_scale.exp()
onnx_model.start_sessions(providers=["CPUExecutionProvider"])
```

## Model Zoo
Models of the original CLIP can be found on this [page](https://github.com/jina-ai/clip-as-service/blob/main/server/clip_server/model/clip_onnx.py).\
They are not part of this library but should work correctly.

## If something doesn't work
It happens that onnx does not convert the model the first time, in these cases it is worth trying to run it again.

If it doesn't help, it makes sense to change the export settings.

Model export options in onnx looks like this:
```python3
DEFAULT_EXPORT = dict(input_names=['input'], output_names=['output'],
                      export_params=True, verbose=False, opset_version=12,
                      do_constant_folding=True,
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
```

You can change them pretty easily.
```python3
from clip_onnx.utils import DEFAULT_EXPORT

DEFAULT_EXPORT["opset_version"] = 15
```

Alternative option (change only visual or textual):
```python3
from clip_onnx import clip_onnx
from clip_onnx.utils import DEFAULT_EXPORT

visual_path = "clip_visual.onnx"
textual_path = "clip_textual.onnx"

textual_export_params = DEFAULT_EXPORT.copy()
textual_export_params["dynamic_axes"] = {'input': {1: 'batch_size'},
                                         'output': {0: 'batch_size'}}
textual_export_params["opset_version"] = 12

Textual = lambda x: x

onnx_model = clip_onnx(model.cpu(), visual_path=visual_path, textual_path=textual_path)
onnx_model.convert2onnx(dummy_input_image, dummy_input_text, verbose=True,
                        textual_wrapper=Textual,
                        textual_export_params=textual_export_params)
```

## Best practices
See [benchmark.md](https://github.com/Lednik7/CLIP-ONNX/tree/main/benchmark.md)
## Examples
See [examples folder](https://github.com/Lednik7/CLIP-ONNX/tree/main/examples) for more details \
Some parts of the code were taken from the [post](https://twitter.com/apeoffire/status/1478493291008172038). Thank you [neverix](https://github.com/neverix) for this notebook.
