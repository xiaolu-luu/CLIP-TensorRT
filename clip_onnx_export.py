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

from clip_onnx import clip_onnx
from clip_onnx import clip_onnx, attention
clip.model.ResidualAttentionBlock.attention = attention

visual_path = "clip_visual.onnx"
textual_path = "clip_textual.onnx"

from clip_onnx.utils import DEFAULT_EXPORT

DEFAULT_EXPORT["opset_version"] = 14 # 17

onnx_model = clip_onnx(model, visual_path=visual_path, textual_path=textual_path)
onnx_model.convert2onnx(image, text, verbose=True)
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_model.start_sessions(providers=["CPUExecutionProvider"]) # cpu mode

image_features = onnx_model.encode_image(image_onnx)
text_features = onnx_model.encode_text(text_onnx)

logits_per_image, logits_per_text = onnx_model(image_onnx, text_onnx)
probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421067 0.00299571]]