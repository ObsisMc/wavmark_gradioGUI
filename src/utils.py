import numpy as np
import soundfile
import torch
import src.wavmark as wavmark
from src.wavmark.utils.wm_add_util import fix_pattern

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)

pattern_bits = 16
payload = np.random.choice([0, 1], size=32 - pattern_bits)


def encode_audio(audio_path):
    signal, sample_rate = soundfile.read(audio_path)
    watermarked_signal, info_encode = wavmark.encode_watermark(model, signal, payload, pattern_bit_length=pattern_bits,
                                                               show_progress=True)
    print(info_encode)
    return watermarked_signal, payload, info_encode


def decode_audio(audio_path, watermark=None):
    signal, sample_rate = soundfile.read(audio_path)
    payload_decoded, _ = wavmark.decode_watermark(model, signal, len_start_bit=pattern_bits,
                                                  show_progress=True)
    BER = None
    if watermark is not None:
        BER = (watermark != payload_decoded).mean() * 100
        print('Decode BER: {:.1f}%'.format(BER))

    return payload_decoded, BER
