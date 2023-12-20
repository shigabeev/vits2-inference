# Russian Text to Speech

TTS models are stored in ONNX format. ONNX is a computational graph with weights, platform agnostic. To run inference from ONNX model you need any ONNX framework and the dictionary to convert text to token ids.

### ONNX model takes 4 inputs:
1. input - text ids. Model takes integers as inputs. Each integer corresponds to id of a letter. Mapping between them is provided in text_to_seq.py
2. input_lengths - length of text. Should match legth of input.
3. scales - array of [noise scale, length scale, noise scale of duration predictor]. Modify length_scale to make speech faster and shorter. Noise scale for audio and duration predictor are vital for natural sound. Not changing them is recommended.
4. sid - Speaker id. Integer. Optional and unused in this setup, but the final system would have multiple speakers available. 

### Directory structure:
1. model_repository - contains the model. Structure is defined by Triton inference server.
2. text_to_seq.py - contains a function to convert text to input ids.
3. infer_onnx.py - contains an example python function to run inference from model.
4. infer_onnx.cs - contains the translation of infer_onnx.py by ChatGPT. Use at your own risk.

## Usage

### Triton server
To serve model use triton docker container:
````
sudo docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver --model-repository=/models
````

Usage instructions can be found on github. 
https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/ONNX

### Python

Use infer_onnx.py in your setup. Modify variable `text` to change input for the model.

### C#

Use infer_onnx.cs in your setup. Modify variable `text` to change input for the model. Probably would require debugging. For reference check ONNX runtime instructions for C#

https://onnxruntime.ai/docs/tutorials/csharp/bert-nlp-csharp-console-app.html

