# Russian Text to Speech

TTS models are stored in ONNX format. ONNX is a computational graph with weights, platform agnostic. To run inference from ONNX model you need any ONNX framework and the dictionary to convert text to token ids.

### ONNX model takes 4 inputs:
1. input - text ids. Model takes integers as inputs. Each integer corresponds to id of a letter. Mapping between them is provided in text_to_seq.py
2. input_lengths - length of text. Should match legth of input.
3. scales - array of [noise scale, length scale, noise scale of duration predictor]. Modify length_scale to make speech faster and shorter. Noise scale for audio and duration predictor are vital for natural sound. Not changing them is recommended.
4. sid - Speaker id. Integer. Optional and unused in this setup, but the final system would have multiple speakers available. 

### Directory structure:

1. infer_onnx.py - contains an example python function to run inference from model.


## Python Usage

Setup

```
pip install -r requirements.txt
```

Now you can use infer_onnx.py in your setup. Modify variable `text` to change input for the model.

```
python infer_onnx.py --text "Что-то совсем старушка распоясалася." --model /path/to/model
```