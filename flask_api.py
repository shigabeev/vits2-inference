from flask import Flask, request, jsonify
import numpy as np
import onnxruntime
import pickle
import base64
from io import BytesIO
from scipy.io import wavfile
from russian_normalization import normalize_russian
# Import other necessary modules and functions like text_to_sequence, inference, etc.

app = Flask(__name__)

global text_processor, model

MODEL_PATH = '/home/frappuccino/dev/vits2_nov26/exported_models/shergin_146k_frozen_te.onnx'  # Specify the path to your ONNX model

# Символы для Наташи
# _pad = '_'
# _punctuation = ' !+,-.:;?«»—'
# _letters = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# # Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Символы для Шергина
_pad = '_'
_punctuation = ' !()+,-./:;<>?«»́‑–—’“”„…'
_letters = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

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



def inference(text, 
              model, 
              sid=[3],
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

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    text_normalized = process_sentence(text)
    audio, sample_rate = inference(text_normalized, model)
    audio_base64 = convert_audio_to_base64(audio)

    return jsonify({'audio': audio_base64, 'sample_rate': sample_rate})

if __name__ == '__main__':
    # Load models and other resources outside of request functions to avoid reloading them on each request
    text_processor = load_accentuator()
    model = load_model(MODEL_PATH)  # Load your TTS model
    app.run(debug=True, port=8080)
