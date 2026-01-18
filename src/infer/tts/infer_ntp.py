
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import os
import time
import argparse

import logging
import torch
import glob
from tqdm import tqdm

import torchaudio
from Amphion.models.codec.ns3_codec import FACodecEncoderV2, FACodecDecoderV2

device = "cuda"

device_model = device

device_decoder = device

device_encoder = device


def preprocess_input(data_point):
    template = templates.get(data_point['task'], None)
    if template:
        sequence = template.format(input=data_point['text'])
        return sequence
    else:
        raise ValueError(f"Not supported for now! ")

def get_logger(logger_path):
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    # filename='time-short-bf16.log',
                    filename=logger_path,
                    # filename='time-long-fp16-flashattn.log',
                    # filename='time-short.log',
                    filemode='a')
    logger = logging.getLogger(__name__)
    return logger



def get_lm_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_model,
        torch_dtype=torch.bfloat16,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

def get_fa_encoder(encoder_path):
    fa_encoder = FACodecEncoderV2(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)
    fa_encoder.load_state_dict(torch.load(encoder_path))
    return fa_encoder.to(device_encoder)


def get_fa_decoder(decoder_path):
    fa_decoder = FACodecDecoderV2(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )
    fa_decoder.load_state_dict(torch.load(decoder_path))
    return fa_decoder.to(device_decoder)



def infer_tts(logger):
    for path in text_paths:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.readlines()[0].strip()
            # text = normalizer(text)
            data.append(text)
            wav_name = os.path.basename(path).split('.')[0] + '.wav'
            wav_names.append(wav_name)
    
    success_num = 0
    sum_tps = 0.
    for i in tqdm(range(len(data))):
        # with open(text_path, "r") as f:
        #     text = f.read()
        d = data[i]
        wav_name = wav_names[i]
        logger.info(wav_name)
        wav_path = wav_paths[i]
        wav_path = prompt_map[wav_name.split("_")[0]]
        wav_path = prompt_map[wav_name.split(".")[0]]

        wav_ref, sr = torchaudio.load(wav_path)
        wav_ref = wav_ref.to(device_encoder).unsqueeze(0)
        wav_ref = torchaudio.functional.resample(wav_ref, orig_freq=sr, new_freq=16000)  # resampling
        prosody_ref = fa_encoder.get_prosody_feature(wav_ref)
        enc_out_ref = fa_encoder(wav_ref)
        _, _, _, _, spk_emb, _ = fa_decoder(enc_out_ref.to(device_decoder), prosody_ref.to(device_decoder), eval_vq=False, vq=True)
        data_point = {
            "task": "tts",
            "text": d
        }
        # ref = d['tokens']

        seq = preprocess_input(data_point)

        input_ids = tokenizer(seq, return_tensors='pt').input_ids.to(device_model)
        input_size = input_ids.shape[-1]
        generation_config = GenerationConfig(**generate_kwargs)

        start = time.time()
        generated_ids = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        end = time.time()

        generated_speech_tokens = generated_ids.sequences[:, input_size:]
        tps = generated_speech_tokens.shape[-1] / (end - start)
        sum_tps += tps
        speech = tokenizer.batch_decode(generated_speech_tokens.cpu(), skip_special_tokens=True)[0]
        logger.info(speech)
        logger.info(f"TPS: {tps:.2f}")
        logger.info("--------------------------------")
        if "<eosp>" in speech:
            idx_eosp = speech.index("<eosp>")
        else:
            print("No <eosp>!")
            continue
        # a, b content, c prosody !!!
        speech_tokens = [int(token[1:]) for token in speech.split("><")[1:-1]]
        # print(speech_tokens)
        len_token = len(speech_tokens)
        # print(len_token)
        if len_token % 3 != 0:
            print("Not divisible by 3")
            continue
        
        success_num += 1
        # # NOTICE: for P-C
        # content_tokens_layer_1 = [speech_tokens[i] for i in range(1, len_token, 3)]
        # content_tokens_layer_2 = [speech_tokens[i] for i in range(2, len_token, 3)]
        # prosody_tokens = [speech_tokens[i] for i in range(0, len_token, 3)]
        # # NOTICE: for C-P
        content_tokens_layer_1 = [speech_tokens[i] for i in range(0, len_token, 3)]
        content_tokens_layer_2 = [speech_tokens[i] for i in range(1, len_token, 3)]
        prosody_tokens = [speech_tokens[i] for i in range(2, len_token, 3)]

        content_ids = torch.tensor([content_tokens_layer_1, content_tokens_layer_2]).to(device_decoder).unsqueeze(1)
        # print(content_ids.shape)
        prosody_ids = torch.tensor([prosody_tokens]).to(device_decoder).unsqueeze(1)

        out = 0
        out += fa_decoder.quantizer[0].vq2emb(
            prosody_ids
        )
        out += fa_decoder.quantizer[1].vq2emb(
            content_ids
        )
        recon_wav = fa_decoder.inference(out, spk_emb.to(device_decoder))
        recon_wav = recon_wav.detach().cpu().squeeze(0)
        save_path = os.path.join(save_dir, wav_name)
        torchaudio.save(save_path, recon_wav, sample_rate=16000)
    print(success_num, len(data))
    avg_tps = sum_tps / len(data)
    logger.info(f"avg TPS: {avg_tps:.2f}")


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--logger-path', type=str, required=True)
    parser.add_argument('--text-path', type=str, required=True)
    parser.add_argument('--wav-ref-path', type=str, required=True)
    parser.add_argument('--task', type=str, default='tts', choices=["asr", "tts"])

    args = parser.parse_args()

    model_path = args.model_path
    encoder_path = 'YOUR_FACODEC_ENCODER_PATH'
    decoder_path = 'YOUR_FACODEC_DECODER_PATH'
    save_dir = args.save_dir
    logger_path = args.logger_path
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(logger_path)
    model, tokenizer = get_lm_model(model_path)
    fa_encoder = get_fa_encoder(encoder_path)
    fa_decoder = get_fa_decoder(decoder_path)
    templates = {
        'tts': tokenizer.bos_token + "[Human]: Read this sentence: {input}.\n [Assistant]: The speech is: ",
    }
    data = []
    wav_names = []
    # normalizer = BasicTextNormalizer()

    text_paths = glob.glob(f"{args.text_path}/*.txt")
    wav_paths = glob.glob(f"{args.wav_ref_path}/*.wav")
    text_paths.sort()
    # wav_paths.sort()

    prompt_version = "YOUR_PATH_TO_REF/*.wav"
    # print(prompt_version)
    # print(model_path)
    prompt_wav_paths = glob.glob(prompt_version)
    prompt_map = {}
    # for prompt_wav_path in wav_paths :
    #     speaker_id = os.path.basename(prompt_wav_path).split(".")[0]
    #     prompt_map[speaker_id] = prompt_wav_path
    # for prompt_wav_path in prompt_wav_paths:
    #     speaker_id = os.path.basename(prompt_wav_path).split(".")[0]
    #     prompt_map[speaker_id] = prompt_wav_path

    # print(prompt_map)
    
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(logger_path)
    model, tokenizer = get_lm_model(model_path)

    generate_kwargs = {
        "max_new_tokens": 5120,
        "min_new_tokens": 10,
        # "temperature": 1.2,
        "do_sample": False, 
        # "top_p": 0.9,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "repetition_penalty": 1.2,
        "use_cache": True
    }
    if args.task == "tts":
        infer_tts(logger)


    
    
    

