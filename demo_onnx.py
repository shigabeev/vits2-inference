# run with streamlit run demo.py --server.port 8080

import os
import streamlit as st

import pickle

from symbols import symbols #, text_to_sequence

from ruaccent import RUAccent

import nltk
import onnxruntime
import numpy as np
from nemo_text_processing.text_normalization.normalize import Normalizer


# _pad = '_'
# _punctuation = ' !()+,-./:;<>?«»́‑–—’“”„…'
# _letters = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"


# # Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)


_pad = '_'
_punctuation = ' !+,-.:;?«»—'
_letters = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


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


@st.cache_resource
def load_model(checkpoint_path):
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession(checkpoint_path, sess_options=sess_options)
    return model

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

@st.cache_resource
def load_normalizer():
    if os.path.isfile('normalizer.pkl'):
        with open('normalizer.pkl', 'rb') as f:
            normalizer = pickle.load(f)
        return normalizer
    return Normalizer(input_case='lower_cased',
                        deterministic=False,
                        lang='ru')

def process_sentence(text):
    text = text.replace(', ', ', , , ')
    text = text.replace('... ', '. ')
    text = text.replace('„', '"')
    text = text.replace( ':',  ': : :')
    return text

if __name__ == "__main__":
    st.title("Синтез речи на русском языке.")

    text_processor = RUAccent()
    text_processor.load(omograph_model_size='medium_poetry', use_dictionary=True, custom_dict={}, custom_homographs={})

    normalizer = load_normalizer()
    
    checkpoint_path1='/home/frappuccino/dev/MB-iSTFT-VITS2/exported_models/shergin_feb1.onnx'
    checkpoint_path2='/home/frappuccino/dev/vits2_nov26/exported_models/natasha.onnx'
    checkpoint_path = st.selectbox("pick the model", [checkpoint_path1])#, checkpoint_path1])
    
    model = load_model(checkpoint_path)

    text = st.text_input(
        'Type text to synthesize (recommended min length 10 characters, max len 200 characters, one sentence!)',
        max_chars=1000)

    length_scale = st.slider("Choose speed:", 0.5, 2.0, value=0.9, step=0.05)
    
    noise_scale = st.slider("Choose noise:", 0.0, 1.0, value=0.2, step=0.1)

    scales = np.array([noise_scale, length_scale, noise_scale], dtype=np.float32)
    if text.strip() and 5 <= len(text) < 1000:
        text_normalized = normalizer.normalize(text, verbose=True, punct_post_process=True)
        text_processed = text_processor.process_all(text_normalized)
        sentences = nltk.sent_tokenize(text_processed)

        final_wav = np.array([])
        for sentence in sentences:
            sentence = process_sentence(sentence)#.replace(', ', ', , ')
            audio, sample_rate = inference(sentence,
                                            model, 
                                            sid=None, 
                                            scales=scales)
            final_wav = np.append(final_wav, audio)
            final_wav = np.append(final_wav, np.zeros(int(0.35*sample_rate)))


        st.audio(final_wav, sample_rate=sample_rate)
