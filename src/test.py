import numpy as np
import soundfile
import torch
import src.wavmark as wavmark
import matplotlib.pyplot as plt
from src.wavmark.utils.wm_add_util import fix_pattern

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = "cpu"
model = wavmark.load_model().to(device)

pattern_bits = 16
payload = np.random.choice([0, 1], size=32 - pattern_bits)
# payload = fix_pattern[pattern_bits:]
print(f"Payload: {payload}")
print(f"Pattern: {fix_pattern[:pattern_bits]}")

audio_path = "/home/zrh/Downloads/speech.wav"
signal, sample_rate = soundfile.read(audio_path)
# plt.plot(signal)
# plt.show(0)

watermarked_signal, info_encode = wavmark.encode_watermark(model, signal, payload, pattern_bit_length=pattern_bits,
                                                 show_progress=True)
print(info_encode)

# add noise TODO: how to add noise
# noise = np.random.normal(0, 0.001, watermarked_signal.shape)
# watermarked_signal_noise = watermarked_signal + noise
watermarked_signal_noise = watermarked_signal

payload_decoded, _ = wavmark.decode_watermark(model, watermarked_signal_noise, len_start_bit=pattern_bits, show_progress=True)
BER = (payload != payload_decoded).mean() * 100

print('Decode BER: {:.1f}%'.format(BER))
