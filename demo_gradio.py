import gradio as gr
import numpy as np
import soundfile
import torch
import src.wavmark as wavmark

import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)
watermark_len = 32
fix_payload_len = 16


audio_input_type = 0
samples_root = os.path.join(os.path.split(os.path.abspath(__file__))[0], "samples")


def select_upload():
    global audio_input_type
    audio_input_type = 0


def select_preload():
    global audio_input_type
    audio_input_type = 1


def select_sample_audio(evt: gr.SelectData):
    return evt.value


def generate_rand_wm():
    global fix_payload_len
    wm_l = fix_payload_len
    return "".join(list(map(str, np.random.randint(0, 2, (wm_l,)))))


def encode_watermark(audio_upload, audio_preload,  watermark_str):
    global audio_input_type
    if audio_input_type == 0:
        audio = audio_upload
    elif audio_input_type == 1:
        audio = audio_preload
    else:
        raise ValueError(f"Unknow audio input_type {audio_input_type}")

    watermark = np.array(list(map(int, watermark_str)), dtype=np.int32)
    pattern_bits = watermark_len - len(watermark)

    signal, sr = soundfile.read(audio)
    if len(signal.shape) > 1:
        signal = signal[:, 0].reshape(-1, )
    watermarked_signal, info_encode = wavmark.encode_watermark(model, signal, watermark,
                                                                pattern_bit_length=pattern_bits,
                                                                show_progress=True)
    output_info = f"Time: {info_encode['time_cost']:.2f}s\n" \
                  f"SNR: {info_encode['snr']:.2f}"
    return (sr, watermarked_signal), output_info


def decode_watermark(audio, watermark=""):
    global fix_payload_len

    if watermark != "":
        watermark = np.array(list(map(int, watermark))).astype(np.float16)
        wm_len = len(watermark)
    else:
        watermark = None
        wm_len = fix_payload_len

    signal, sr = soundfile.read(audio)
    if len(signal.shape) > 1:
        signal = signal[:, 0].reshape(-1, )
    watermark_decoded, info = wavmark.decode_watermark(model, signal, len_start_bit=watermark_len - wm_len,
                                                        show_progress=True)
    more_info = f"Time: {info['time_cost']:.2f}"

    if watermark is not None and watermark_decoded is not None:
        BER = (watermark != watermark_decoded).mean()
        acc = 1 - BER
        BER = f"{BER * 100:.2f}%"
        acc = f"{acc * 100:.2f}%"
    else:
        BER = acc = "No watermark to verify"

    return "".join(
        list(map(str, watermark_decoded))) if watermark_decoded is not None else "No watermark", acc, BER, more_info


with gr.Blocks() as demo:
    with gr.Tab("Add watermark"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Input")

                with gr.Tab("Upload") as upload_tab:
                    origin_audio = gr.Audio(label="Audio", type="filepath", sources=["upload"])
                with gr.Tab("Preloaded") as preload_tab:
                    audio_samples_radio = gr.Radio(
                        choices=[("music1", os.path.join(samples_root, "music1_clip.wav")),
                                 ("music2", os.path.join(samples_root, "music2_clip.wav")),
                                 ("music3", os.path.join(samples_root, "music3_clip.wav")),
                                 ("music4", os.path.join(samples_root, "music4_clip.mp3")),
                                 ("music5", os.path.join(samples_root, "music5_clip.mp3")),
                                 ("music6", os.path.join(samples_root, "music6_clip.mp3"))])
                    samples_audio = gr.Audio(label="Audio", type="filepath", sources=[])

                    audio_samples_radio.select(select_sample_audio, None, outputs=[samples_audio])
                upload_tab.select(select_upload)
                preload_tab.select(select_preload)

                generate_btn = gr.Button("Generate Watermark")
                with gr.Row():
                    with gr.Column(scale=1):
                        random_watermark = gr.Textbox(label=f"Watermark ({fix_payload_len} bits)",
                                                      placeholder="Click button to generate a watermark",
                                                      )
                    # with gr.Column(scale=1):
                    #     watermark_len_textbox = gr.Textbox(label="Watermark Length")

                encode_watermark_btn = gr.Button("Encode Watermark")
                generate_btn.click(generate_rand_wm,
                                   inputs=None,
                                   outputs=[random_watermark])
            with gr.Column(scale=1):
                gr.Markdown("## Output")
                watermarked_audio = gr.Audio(label="Watermarked Audio")
                info = gr.Textbox(label="More info")

            encode_watermark_btn.click(encode_watermark,
                                       inputs=[origin_audio,
                                               samples_audio,
                                               random_watermark],
                                       outputs=[watermarked_audio,
                                                info])

    with gr.Tab("Reveal watermark"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Input")
                audio2decode = gr.Audio(label="Audio", type="filepath", sources=["upload"])

                gr.Markdown(f"You can input a watermark to verify or directly click the 'Reveal watermark' button")
                with gr.Row():
                    wm2verified = gr.Textbox(label="Watermark to be verified",
                                             placeholder=f"The watermark's length should be {fix_payload_len}")
                    # wm_len_num = gr.Number(label="Watermark length")
                # gr.Markdown("If you input a watermark, you don't need to input the watermark length, "
                #             "otherwise the watermark length is needed")

                decode_btn = gr.Button("Reveal watermark")

            with gr.Column():
                gr.Markdown("## Output")
                revealed_wm = gr.Textbox(label="Revealed watermark")
                with gr.Row():
                    acc_text = gr.Textbox(label="Accuracy")
                    ber_text = gr.Textbox(label="Bit Error Rate")
                more_info_text = gr.Textbox(label="More info")

            decode_btn.click(decode_watermark,
                             inputs=[audio2decode,
                                     wm2verified],
                             outputs=[revealed_wm,
                                      acc_text,
                                      ber_text,
                                      more_info_text])

demo.queue().launch()
