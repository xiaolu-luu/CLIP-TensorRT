trtexec --onnx=../textual.onnx \
        --memPoolSize=workspace:2048 \
        --saveEngine=./engines/clip_textual.engine \
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
        > ./build/log/build_textual.log

# trtexec --loadEngine=./engines/clip_visual.engine \
#         --dumpOutput \
#         --dumpProfile \
#         --dumpLayerInfo \
#         --exportOutput=./build/log/infer/infer_output.log \
#         --exportProfile=./build/log/infer/infer_profile.log \
#         --exportLayerInfo=./build/log/infer/infer_layer_info.log \
#         --warmUp=200 \
#         --iterations=50 \
#         > ./build/log/infer/infer.log

# trtexec \
#     --loadEngine=./engines/clip_visual.engine \
#     --dumpProfile \
#     --exportTimes="./clip_visual-exportTimes.json" \
#     --exportProfile="./clip_visual-exportProfile.json" \
#     > result-05.log 2>&1
