using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class TextToAudio : MonoBehaviour
{
    private readonly string _pad = "_";
    private readonly string _punctuation = " !+,-.:;?«»—";
    private readonly string _letters = "абвгдежзийклмнопрстуфхцчшщъыьэюяё";
    private readonly string _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘";
    private Dictionary<string, int> _symbol_to_id;
    
    void Start()
    {
        string symbols = _pad + _punctuation + _letters + _letters_ipa;
        _symbol_to_id = new Dictionary<string, int>();
        for (int i = 0; i < symbols.Length; i++)
        {
            _symbol_to_id.Add(symbols[i].ToString(), i);
        }

        var sessOptions = new SessionOptions();
        var modelPath = "model_repository/natasha/1/model.onnx"; // Make sure you specify the right path
        var model = new InferenceSession(modelPath, sessOptions);
        
        string text = "+это раб+отает к+ак н+адо abc!";
        Tensor<long> phonemeIds = TextToSequence(text);
        Tensor<long> textInput = new DenseTensor<long>(new[] { 1, phonemeIds.Length });
        textInput[0] = phonemeIds;

        var textLengths = new DenseTensor<long>(new long[] { phonemeIds.Length });
        var scales = new DenseTensor<float>(new float[] { 0.667f, 1.0f, 0.8f });
        var sid = new DenseTensor<float>(new float[] {}); // Assuming None translates to an empty tensor

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", textInput),
            NamedOnnxValue.CreateFromTensor("input_lengths", textLengths),
            NamedOnnxValue.CreateFromTensor("scales", scales),
            NamedOnnxValue.CreateFromTensor("sid", sid)
        };

        var results = model.Run(inputs);
        var audio = results[0].AsTensor<float>().ToArray();
        
        // TODO: Write audio array to WAV file (Unity doesn't have scipy.io.wavfile)
    }

    private Tensor<long> TextToSequence(string text)
    {
        List<long> sequence = new List<long>();
        foreach (var ch in text.ToLower())
        {
            int symbolId;
            if (_symbol_to_id.TryGetValue(ch.ToString(), out symbolId))
            {
                sequence.Add(symbolId);
            }
        }

        return new DenseTensor<long>(sequence.ToArray(), new int[] { sequence.Count });
    }
}
