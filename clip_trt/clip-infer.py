
import os
from datetime import datetime as dt
from glob import glob

# import calibrator
import cv2
import numpy as np
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from cuda import cudart
from torch.autograd import Variable
import clip
from PIL import Image

def run_visual_encode(onnxFile,trtFile,inferenceImage,nHeight, nWidth):
    logger = trt.Logger(trt.Logger.VERBOSE)
    if os.path.isfile(trtFile):
        with open (trtFile,"rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return 
        print("Succeeded getting serialized engine!")
    else:
        # Parse network, rebuild network and do inference in TensorRT ------------------
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if bUseFP16Mode:
            config.set_flag(trt.BuilderFlag.FP16)
        if bUseINT8Mode:
            config.set_flag(trt.BuilderFlag.INT8)
            # config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnxFile):
            print("Failed finding ONNX file!")
            exit()
        print("Succeeded finding ONNX file!")
        with open(onnxFile, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing .onnx file!")

        inputTensor = network.get_input(0)
        profile.set_shape(inputTensor.name, [1, 3, nHeight, nWidth], [4, 3, nHeight, nWidth], [8, 3, nHeight, nWidth])
        config.add_optimization_profile(profile)

        # network.unmark_output(network.get_output(0))  # remove output tensor "y"
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [1, 3, nHeight, nWidth])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []

    image = cv2.imread(inferenceImage)
    data = cv2.resize(image,(224,224))
    resized_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    # 转换图像数据类型为浮点数
    resized_image = resized_image.astype(np.float32)

    # 将图像像素值归一化到[0, 1]
    resized_image /= 255.0

    resized_image = resized_image.transpose(2, 0, 1)  # 从HWC转换为CHW
    resized_image = resized_image[np.newaxis, :, :, :]  # 添加批次维度
    # model, preprocess = clip.load("ViT-B/32", device="cpu")
    # image = preprocess(Image.open(inferenceImage)).unsqueeze(0)
    # resized_image = image.numpy()
    bufferH.append(np.ascontiguousarray(resized_image))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nIO):
        print(lTensorName[i])
        print(bufferH[i])

    for b in bufferD:
        cudart.cudaFree(b)
    print("Succeeded running model in TensorRT!")
    return bufferH[1]


def run_textual_encode(onnxFile,trtFile,inferencetext,text_len = 77):
    logger = trt.Logger(trt.Logger.VERBOSE)
    if os.path.isfile(trtFile):
        with open (trtFile,"rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return 
        print("Succeeded getting serialized engine!")
    else:
        # Parse network, rebuild network and do inference in TensorRT ------------------
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if bUseFP16Mode:
            config.set_flag(trt.BuilderFlag.FP16)
        if bUseINT8Mode:
            config.set_flag(trt.BuilderFlag.INT8)
            # config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnxFile):
            print("Failed finding ONNX file!")
            exit()
        print("Succeeded finding ONNX file!")
        with open(onnxFile, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing .onnx file!")

        inputTensor = network.get_input(0)
        profile.set_shape(inputTensor.name, [1, text_len], [4, text_len], [8, text_len])
        config.add_optimization_profile(profile)

        # network.unmark_output(network.get_output(0))  # remove output tensor "y"
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [3, text_len])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []

    bufferH.append(np.ascontiguousarray(inferencetext))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nIO):
        print(lTensorName[i])
        print(bufferH[i])

    for b in bufferD:
        cudart.cudaFree(b)
    print("Succeeded running model in TensorRT!")
    return bufferH[1]
if __name__ == "__main__":

    np.set_printoptions(precision=3, linewidth=200, suppress=True)
    cudart.cudaDeviceSynchronize()
    nHeight = 224
    nWidth = 224
    visual_onnxFile = "../visual.onnx"
    textual_onnxFile = "../textual.onnx"
    
    visual_trtFile = "./model.plan"
    textual_trtFile = "./model2.plan"

    inferenceImage = "cat.jpg"

    # for FP16 mode
    bUseFP16Mode = False
    # for INT8 model
    bUseINT8Mode = False
    nCalibration = 1

    text_encode =np.array(clip.tokenize(["a dog","cat","Squirrel"]))

    visual_encoder = run_visual_encode(visual_onnxFile,visual_trtFile,inferenceImage,nHeight,nWidth)
    text_encoder = run_textual_encode(textual_onnxFile,textual_trtFile,text_encode)
    print(np.array(visual_encoder).shape)
    print(np.array(text_encoder).shape)
    
    result = np.dot(visual_encoder,text_encoder.T)  # 形状变为(512, 3)
    print(result.shape)
    softmax_output = (np.exp(result) / np.sum(np.exp(result), axis=1, keepdims=True)).squeeze()

    # print(softmax_output)
    print(softmax_output)


