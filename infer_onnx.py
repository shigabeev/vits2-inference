import argparse

import onnxruntime
import numpy as np
from russian_normalization import normalize_russian
from scipy.io import wavfile

_pad = '_'
_punctuation = ' !()+,-./:;<>?«»́‑–—’“”„…'
_letters = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def normalize_text(text, text_processor):
    text = normalize_russian(text)
    if text_processor:
        text = text_processor.process_all(text)
    return text


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


def load_model(checkpoint_path):
    """
    TTS model
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=str, help="Text to synthesize")
    parser.add_argument("--model", required=True,
                        help="Path to model (.onnx)")
    parser.add_argument(
        "--output", default='output.wav', help="Path to write WAV file"
    )
    
    args = parser.parse_args()
    model = load_model(args.model)
    text_processor = load_accentuator()
    
    text = normalize_text(args.text,
                          text_processor=text_processor)
    
    audio, sample_rate = inference(text, model, sid=[3])
    
    wavfile.write(args.output, rate=sample_rate, data=audio)


if __name__ == "__main__":
    main()
