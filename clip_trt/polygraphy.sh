# 01-Run polygraphy from ONNX file in onnxruntime without any more option
# polygraphy run ../visual.onnx \
#     --onnxrt \
#     > result-01-ploy.log 2>&1


# build engine
# polygraphy run ../clip_textual.onnx \
#     --trt \
#     --save-engine ./engines/clip_textual_poly.plan \
#     --save-timing-cache ./engines/clip_textual_poly.cache \
#     --save-tactics ./engines/clip_textual_poly_tactics.json \
#     --trt-min-shapes 'input:[1,77]' \
#     --trt-opt-shapes 'input:[4,77]' \
#     --trt-max-shapes 'input:[16,77]' \
#     --fp16 \
#     --pool-limit workspace:1G \
#     --builder-optimization-level 5 \
#     --max-aux-streams 4 \
#     --input-shapes   'input:[4,77]' \
#     --verbose \
#     > ./build/log/result-poly-02.log 2>&1

# 04-Compare the output of each layer between Onnxruntime and TensorRT
polygraphy run ../clip_textual.onnx \
    --onnxrt --trt \
    --save-engine=./engines/clip_textual_poly.plan \
    --onnx-outputs mark all \
    --trt-outputs mark all \
    --trt-min-shapes 'input:[1,77]' \
    --trt-opt-shapes 'input:[4,77]' \
    --trt-max-shapes 'input:[16,77]' \
    --input-shapes   'input:[4,77]' \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    > ./build/log/result-poly-3.log 2>&1
