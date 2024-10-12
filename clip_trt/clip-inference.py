import os
import cv2
import numpy as np
import time
import clip
import tensorrt as trt
from cuda import cudart


class VisualEncoder:
    def __init__(self, onnxFile, trtFile, nHeight, nWidth, bUseFP16Mode=False, bUseINT8Mode=False):
        self.onnxFile = onnxFile
        self.trtFile = trtFile
        self.nHeight = nHeight
        self.nWidth = nWidth
        self.bUseFP16Mode = bUseFP16Mode
        self.bUseINT8Mode = bUseINT8Mode
        self.engine = None
        self.context = None
        self.bufferH = []
        self.bufferD = []

    def build_engine(self):
        logger = trt.Logger(trt.Logger.VERBOSE)
        if os.path.isfile(self.trtFile):
            with open(self.trtFile, "rb") as f:
                engineString = f.read()
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        else:
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            profile = builder.create_optimization_profile()
            config = builder.create_builder_config()
            if self.bUseFP16Mode:
                config.set_flag(trt.BuilderFlag.FP16)
            if self.bUseINT8Mode:
                config.set_flag(trt.BuilderFlag.INT8)
                # config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, self.nHeight, self.nWidth), cacheFile)

            parser = trt.OnnxParser(network, logger)
            if not os.path.exists(self.onnxFile):
                print("Failed finding ONNX file!")
                return False
            with open(self.onnxFile, "rb") as model:
                if not parser.parse(model.read()):
                    print("Failed parsing .onnx file!")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False
            print("Succeeded parsing .onnx file!")

            inputTensor = network.get_input(0)
            # config.max_workspace_size = 1 << 30  # 1GB
            profile.set_shape(inputTensor.name, [1, 3, self.nHeight, self.nWidth], [4, 3, self.nHeight, self.nWidth], [8, 3, self.nHeight, self.nWidth])
            config.add_optimization_profile(profile)
            engineString = builder.build_serialized_network(network, config)
            if engineString == None:
                print("Failed building engine!")
                exit()
            print("Succeeded building engine!")
            with open(self.trtFile, "wb") as f:
                f.write(engineString)
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

        if self.engine is None:
            print("Failed building engine!")
            return False
        self.context = self.engine.create_execution_context()
        
        return True

    def infer_visual(self, inferenceImage):
        if self.engine is None:
            print("Engine not built!")
            return None
        
        nIO = self.engine.num_io_tensors
        lTensorName = [self.engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [self.engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
        self.context.set_input_shape(lTensorName[0],[1,3,self.nHeight,self.nWidth])
        
        # for i in range(nIO):
        #     print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), self.engine.get_tensor_dtype(lTensorName[i]), self.engine.get_tensor_shape(lTensorName[i]), self.context.get_tensor_shape(lTensorName[i]), lTensorName[i])
        
        bufferH = []
        image = cv2.imread(inferenceImage)
        data = cv2.resize(image, (self.nWidth, self.nHeight))
        resized_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        resized_image = resized_image.astype(np.float32) / 255.0
        resized_image = resized_image.transpose(2, 0, 1)[np.newaxis, :, :, :]

        start_time = time.time()
        bufferH.append(np.ascontiguousarray(resized_image))
        for i in range(nInput, nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(self.engine.get_tensor_dtype(lTensorName[i]))))

        bufferD = [cudart.cudaMalloc(bufferH[i].nbytes)[1] for i in range(nIO)]
        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            self.context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        
        self.context.execute_async_v3(0)
        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        
        # for i in range(nIO):
        #     print(lTensorName[i])
        #     print(bufferH[i])
        
        for b in bufferD:
            cudart.cudaFree(b)

        # print("Succeeded running model in TensorRT!")
        return bufferH[nInput] , (time.time() - start_time)*1000 # Assuming the first output tensor is the one we want
    
    
    def infer_visual_multiple_times(self,image_paths):
        if self.engine is None:
            print("Engine not build!")
            return None
        total_time = 0
        inference_count = len(image_paths)
        out_visual_encode = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Image file {image_path} not found!")
                continue
            output,inference_time = self.infer_visual(image_path)
            print(image_path)
            out_visual_encode.append(output)
            total_time += inference_time

        out_visual_encode = np.array(out_visual_encode)
        out_visual_encode = np.squeeze(out_visual_encode)
        print(f"Succeeded running model {inference_count} times in TensorRT!")
        print(f"Average inference time: {total_time / inference_count:.4f} ms")
        return out_visual_encode
    
    def infer_visual_test_times(self,inferenceImage,num_inferences = 1000):
        if self.engine is None:
            print("Engine not build!")
            return None
        total_time = 0
        for _ in range (num_inferences):
            output,inference_time = self.infer_visual(inferenceImage)
            total_time += inference_time
        print("======="*5)
        print(f"visual_running_test")
        print(f"Succeeded running model {num_inferences} times in TensorRT!")
        print(f"Average inference time: {total_time / num_inferences:.4f} ms")
        print(f"Total time for 100 inferences: {total_time:.4f} ms")
        print("======="*5)



class TextualEncoder:
    def __init__(self, onnxFile, trtFile, text_len= 77, bUseFP16Mode=False, bUseINT8Mode=False):
        self.onnxFile = onnxFile
        self.trtFile = trtFile
        self.text_len = text_len
        self.bUseFP16Mode = bUseFP16Mode
        self.bUseINT8Mode = bUseINT8Mode
        self.engine = None
        self.context = None
        self.bufferH = []
        self.bufferD = []

    def build_engine(self):
        logger = trt.Logger(trt.Logger.VERBOSE)
        if os.path.isfile(self.trtFile):
            with open(self.trtFile, "rb") as f:
                engineString = f.read()
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        else:
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            profile = builder.create_optimization_profile()
            config = builder.create_builder_config()
            if self.bUseFP16Mode:
                config.set_flag(trt.BuilderFlag.FP16)
            if self.bUseINT8Mode:
                config.set_flag(trt.BuilderFlag.INT8)
                # config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, self.nHeight, self.nWidth), cacheFile)

            parser = trt.OnnxParser(network, logger)
            if not os.path.exists(self.onnxFile):
                print("Failed finding ONNX file!")
                return False
            with open(self.onnxFile, "rb") as model:
                if not parser.parse(model.read()):
                    print("Failed parsing .onnx file!")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False
            print("Succeeded parsing .onnx file!")

            inputTensor = network.get_input(0)
            # config.max_workspace_size = 1 << 30  # 1GB
            profile.set_shape(inputTensor.name, [1,self.text_len], [4,self.text_len ], [8,self.text_len ])
            config.add_optimization_profile(profile)
            engineString = builder.build_serialized_network(network, config)
            if engineString == None:
                print("Failed building engine!")
                exit()
            print("Succeeded building engine!")
            with open(self.trtFile, "wb") as f:
                f.write(engineString)
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

        if self.engine is None:
            print("Failed building engine!")
            return False
        self.context = self.engine.create_execution_context()
        
        return True

    def infer_textual(self, textual_input):
        if self.engine is None:
            print("Engine not built!")
            return None
        
        nIO = self.engine.num_io_tensors
        lTensorName = [self.engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [self.engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
        self.context.set_input_shape(lTensorName[0],[4,self.text_len])
        
        # for i in range(nIO):
        #     print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), self.engine.get_tensor_dtype(lTensorName[i]), self.engine.get_tensor_shape(lTensorName[i]), self.context.get_tensor_shape(lTensorName[i]), lTensorName[i])
        
        bufferH = []
       
        start_time = time.time()
        bufferH.append(np.ascontiguousarray(textual_input))
        for i in range(nInput, nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(self.engine.get_tensor_dtype(lTensorName[i]))))

        bufferD = [cudart.cudaMalloc(bufferH[i].nbytes)[1] for i in range(nIO)]
        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            self.context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        
        self.context.execute_async_v3(0)

        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        
        # for i in range(nIO):
        #     print(lTensorName[i])
        #     print(bufferH[i])
        
        for b in bufferD:
            cudart.cudaFree(b)

        # print("Succeeded running model in TensorRT!")
        return bufferH[nInput] , (time.time() - start_time)*1000 # Assuming the first output tensor is the one we want
    
    
    def infer_text_multiple_times(self,image_paths):
        # TODO 
        if self.engine is None:
            print("Engine not build!")
            return None
        total_time = 0
        inference_count = len(image_paths)
        out_visual_encode = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Image file {image_path} not found!")
                continue
            output,inference_time = self.infer_visual(image_path)
            out_visual_encode.append(output)
            total_time += inference_time

        out_visual_encode = np.array(out_visual_encode)
        out_visual_encode = np.squeeze(out_visual_encode)
        print(f"Succeeded running model {inference_count} times in TensorRT!")
        print(f"Average inference time: {total_time / inference_count:.4f} ms")
        return out_visual_encode
    
    def infer_text_test_times(self,textual_input,num_inferences = 1000):
        if self.engine is None:
            print("Engine not build!")
            return None
        total_time = 0
        for _ in range (num_inferences):
            output,inference_time = self.infer_textual(textual_input)
            total_time += inference_time
        print("======="*5)
        print(f"textual_running_test")
        print(f'batch size = {output.shape[0]}')
        print(f"Succeeded running model {num_inferences} times in TensorRT!")
        print(f"Average inference time: {total_time / num_inferences:.4f} ms")
        print(f"Total time for 100 inferences: {total_time:.4f} ms")
        print("======="*5)


if __name__ == "__main__":

    np.set_printoptions(precision=3, linewidth=200, suppress=True)
    cudart.cudaDeviceSynchronize()
    # Usage
    visual_onnxFile = "../clip_visual1.onnx"
    textual_onnxFile = "../clip_textual1.onnx"
    
    visual_trtFile = "./engines/clip_visual_ln.plan"
    textual_trtFile = "./engines/clip_textual_ln.plan"

    inferenceImage = "cat.jpg"
    # onnx_file = '../visual.onnx'
    # trt_file = './clip_trt/model.plan'
    # image_path = './clip_trt/cat.jpg'
    image_folder = './image'

    height = 224
    width = 224
    text_len = 77

    use_fp16 = False
    use_int8 = False

    visual_encoder = VisualEncoder(visual_onnxFile, visual_trtFile, height, width, use_fp16, use_int8)
    textual_encoder = TextualEncoder(textual_onnxFile, textual_trtFile, text_len, use_fp16, use_int8)
    if visual_encoder.build_engine():

        #=========Test a picture==============
        output, _ = visual_encoder.infer_visual(inferenceImage)
        # print(output)

        #=========Test a image folder=========
        image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('jpg', 'png', 'jpeg', 'bmp'))]#[:-1] 
        visual_feature = visual_encoder.infer_visual_multiple_times(image_paths)

        # ==========Test the time it takes to run the inference 100 times========
        # output = visual_encoder.infer_visual_test_times(image_path,100)
    
    if textual_encoder.build_engine():

        #=========Test a textual list==============
        text_encode =np.array(clip.tokenize(["dog","cat","people","chicken"]))
        text_feature, _ = textual_encoder.infer_textual(text_encode)
        
        print('visual_feature:',np.array(visual_feature).shape)
        print('text_feature:',np.array(text_feature).shape)

        # ==========Test the time it takes to run the inference 100 times========
        # textual_encoder.infer_text_test_times(text_encode)

    result = np.dot(visual_feature,text_feature.T)  # shape = (10, 4)
    print(result.shape)
    softmax_output = (np.exp(result) / np.sum(np.exp(result), axis=1, keepdims=True)).squeeze()

    # print(softmax_output)
    print(softmax_output)

