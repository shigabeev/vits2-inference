import onnxruntime
import numpy as np
from scipy.io import wavfile

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

sess_options = onnxruntime.SessionOptions()
model = onnxruntime.InferenceSession('model_repository/natasha/1/model.onnx', sess_options=sess_options)
text = '+это раб+отает к+ак н+адо abc!'

phoneme_ids = text_to_sequence(text)
text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
text_lengths = np.array([text.shape[1]], dtype=np.int64)
scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
sid = None

audio = model.run(
    None,
    {
        "input": text,
        "input_lengths": text_lengths,
        "scales": scales,
        "sid": sid,
    },
)[0].squeeze((0, 1))

wavfile.write('audio.wav', rate=22050, data=audio)



