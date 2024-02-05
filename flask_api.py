from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import numpy as np
import io
import numpy as np
import onnxruntime
import pickle
import base64
from io import BytesIO
from scipy.io import wavfile
from russian_normalization import normalize_russian
from symbols import symbols

app = FastAPI(title="TTS API", description="A simple Text-to-Speech API", version="1.0")

global text_processor, model

MODEL_PATH = '/home/frappuccino/dev/MB-iSTFT-VITS2/exported_models/shergin_feb1.onnx'  # Specify the path to your ONNX model

# _pad = '_'
# _punctuation = ' !+,-.:;?«»—'
# _letters = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# # Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# # Mappings from symbol to numeric ID and vice versa:
# _symbol_to_id = {s: i for i, s in enumerate(symbols)}

class TTSRequest(BaseModel):
    text: str

def convert_audio_to_base64(audio):
    # Convert numpy audio array to base64 encoded audio
    buffered = BytesIO()
    wavfile.write(buffered, 22050, audio)
    audio_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return audio_base64

def load_model(checkpoint_path):
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession(checkpoint_path, sess_options=sess_options)
    return model


def load_accentuator():
    """
    Text normalizer. Converts "На горе стояла елка" into "На гор+е сто+яла +ёлка"
    """
    try:
        from ruaccent import RUAccent 
        text_processor = RUAccent()
        text_processor.load(omograph_model_size='medium_poetry', 
                            use_dictionary=True, 
                            custom_dict={}, 
                            custom_homographs={})
        return text_processor
    except ImportError:
        return None


def process_sentence(text):
    text = normalize_russian(text)
    if text_processor:
        text = text_processor.process_all(text)
    text = text.replace(', ', ', , , ')
    text = text.replace('... ', '. ')
    text = text.replace('„', '"')
    text = text.replace( ':',  ': : :')
    return text

def split_text(text):
    # Implement your text splitting logic here
    # For simplicity, let's assume this function yields sentences or phrases from the text
    yield from text.split('. ')

def inference(text, 
              model, 
              sid=None,
              scales=np.array([0.667, 1.0, 0.8], dtype=np.float32)):
    phoneme_ids = text_to_sequence(text)
    text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    text_lengths = np.array([text.shape[1]], dtype=np.int64)

    audio = model.run(
        None,
        {
            "input": text,
            "input_lengths": text_lengths,
            "scales": scales,
            "sid": sid,
        },
    )[0].squeeze((0, 1))
    audio *=32767
    audio = audio.astype("int16")
    return audio, 22050


def text_to_sequence(text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    clean_text = text.lower()
    for symbol in clean_text:
        symbol_id = _symbol_to_id.get(symbol, None)
        if symbol_id != None:
            sequence += [symbol_id]
    return np.array(sequence)


@app.post("/synthesize/")
async def synthesize(request: TTSRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    text_normalized = process_sentence(text)
    audio, sample_rate = inference(text_normalized, model)

    # Log the size of the generated audio data
    print(f"Generated audio size: {len(audio)} bytes")

    if len(audio) == 0:
        raise HTTPException(status_code=500, detail="Generated audio is empty")

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio.astype(np.int16))
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")


# @app.post("/synthesize_async/")
# async def synthesize_async(request: TTSRequest):
#     text = request.text
#     if not text:
#         raise HTTPException(status_code=400, detail="No text provided")

#     def generate_audio():
#         for chunk in split_text(text):
#             text_normalized = process_sentence(chunk)  # Process each chunk of text

#             audio, sample_rate = inference(text_normalized, model)  # Get audio for the chunk
#             audio = audio.astype(np.int16)
#             yield audio.tobytes()  # Yield audio bytes

#     return StreamingResponse(generate_audio(), media_type="audio/wav")


if __name__ == '__main__':
    import uvicorn
    # Load models and other resources outside of request functions to avoid reloading them on each request
    text_processor = load_accentuator()
    model = load_model(MODEL_PATH)  # Load your TTS model
    uvicorn.run(app, host="0.0.0.0", port=8000)