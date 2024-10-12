# CLIP-TensorRT
This is a simple repository for accelerating CLIP inference using TensorRT, which includes directly converting the ONNX model to a TRT inference engine, as well as importing a self-written **LayerNorm** Plugin to achieve inference acceleration.

**CLIP** model is a multimodal pre-trained neural network, which is an efficient and scalable method for learning from natural language supervision. The core idea of the model is to use a large number of image and text pairing data for pre-training to learn the alignment relationship between image and text. CLIP model has two modes, one is text mode, one is visual mode, including two main parts:

1. **Text Encoder**: Used to convert text into a low-dimensional vector.
2. **Image Encoder**: It is used to transform the image into a similar vector representation.


![](CLIP.png)


In the prediction phase, the CLIP model generates predictions by calculating the **cosine similarity (CS)** between text and image vectors.


<table>
    <tr>
        <td>
            <div style="position: relative; width: 100%;">
                <img src="clip_trt/image/cat1.jpg" width="100%" style="height: 200px; object-fit: cover;">
                <div style="position: absolute; bottom: 0; width: 100%; background: rgba(0,0,0,0.5); color: white; padding: 5px;">
                    <p>Cat</p>
                    <p>CS: 98%</p>
                </div>
            </div>
        </td>
        <td>
            <div style="position: relative; width: 100%;">
                <img src="clip_trt/image/dog1.jpg" width="100%" style="height: 200px; object-fit: cover;">
                <div style="position: absolute; bottom: 0; width: 100%; background: rgba(0,0,0,0.5); color: white; padding: 5px;">
                    <p>Dog</p>
                    <p>CS: 99%</p>
                </div>
            </div>
        </td>
        <td>
            <div style="position: relative; width: 100%;">
                <img src="clip_trt/image/chicken1.jpg" width="100%" style="height: 200px; object-fit: cover;">
                <div style="position: absolute; bottom: 0; width: 100%; background: rgba(0,0,0,0.5); color: white; padding: 5px;">
                    <p>Chicken</p>
                    <p>CS: 97%</p>
                </div>
            </div>
        </td>
        <td>
            <div style="position: relative; width: 100%;">
                <img src="clip_trt/image/people1.jpg" width="100%" style="height: 200px; object-fit: cover;">
                <div style="position: absolute; bottom: 0; width: 100%; background: rgba(0,0,0,0.5); color: white; padding: 5px;">
                    <p>People</p>
                    <p>CS: 98%</p>
                </div>
            </div>
        </td>
    </tr>
</table>

## Requirements

We have tested it on CUDA 12.1 ,TensorRT 8.6.1 ,ONNX 1.17.0 ,onnxruntime 1.13.1 .

## Usage

Export the visual encoder and text encoder of CLIP to the `clip_visual.onnx` file and `clip_textual.onnx`, respectively

```bash
git clone https://github.com/xiaolu-luu/CLIP-TensorRT.git
python clip_onnx_export.py
```

### Running Foundational CLIP-TensorRT

```bash
cd clip_trt
python clip-inference.py >result.log 2>&1
```

### Running Modified CLIP-TensorRT (Using Plugin)

1. Compile the handwritten TextualLayerNorm plugin and VisualLayerNorm plugin, then test the correctness of both plugins.

```bash
cd LayerNormPlugin
make clean
make
python testLayerNormPlugin.py >test.log 2>&1
```

2. Build the engine and inference.

```bash
cd clip_trt
python clip-inference-plugin.py >result-plugin.log 2>&1
```

## If something doesn't work when export onnx

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

## Additional Information

Introduces some of the uses of trtexec and poly for model export and performance verification

### 1. Using trtexec

Export engine by using trtexec.

```bash
trtexec --onnx=../clip_textual.onnx \
        --memPoolSize=workspace:2048 \
        --saveEngine=./engines/clip_textual_trt.engine \
        --profilingVerbosity=detailed \
        --dumpOutput \
        --dumpProfile \
        --dumpLayerInfo \
        --exportOutput=./build/log/build_output_textual.log \
        --exportProfile=./build/log/build_profile_textual.log \
        --exportLayerInfo=./build/log/build_layer_info_textual.log \
        --warmUp=200 \
        --iterations=50 \
        --verbose \
        > ./build/log/result_trt_build_textual.log
```

### 2. Using polygraphy

Export engine by using polygraphy.

```bash
polygraphy run ../clip_textual.onnx \
    --trt \
    --save-engine ./engines/clip_textual_poly.plan \
    --save-timing-cache ./engines/clip_textual_poly.cache \
    --save-tactics ./engines/clip_textual_poly_tactics.json \
    --trt-min-shapes 'input:[1,77]' \
    --trt-opt-shapes 'input:[4,77]' \
    --trt-max-shapes 'input:[16,77]' \
    --fp16 \
    --pool-limit workspace:1G \
    --builder-optimization-level 5 \
    --max-aux-streams 4 \
    --input-shapes   'input:[4,77]' \
    --verbose \
    > ./build/log/result-poly-01.log 2>&1
```

## ACKNOWLEDGE

The code for exporting the ONNX model is sourced from this repository[CLIP-ONNX](https://github.com/Lednik7/CLIP-ONNX), and we are very grateful for the support it has provided to our work.
