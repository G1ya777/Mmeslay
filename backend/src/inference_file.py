from Squeezeformer import MySqueezeformer
from torchaudio.models.decoder import ctc_decoder
from torchaudio.transforms import MelSpectrogram
from torch import log,mean,tensor
import torch
import sentencepiece as spm
from torchaudio import load
import os 
import re
import numpy as np
import kenlm

dirname = os.path.dirname(__file__)

sp = spm.SentencePieceProcessor()
sp_model = os.path.join(dirname, '../ressources/tokenizer/128_v7.model')
sp.load(sp_model)

lm = kenlm.Model('./ressources/kenLM_model/kab_5k_6-gram_v2.bin')
sp_lm = spm.SentencePieceProcessor()
sp_lm_model = os.path.join(dirname, '../ressources/tokenizer/5K.model')
sp_lm.load(sp_lm_model)


model = MySqueezeformer()
acoustic_model = os.path.join(dirname, '../ressources/e2e_model/squeezeformer')
model.load_state_dict(torch.load(acoustic_model))
model.eval()


tokens_file = os.path.join(dirname, '../ressources/tokenizer/128_v7.txt')


decoder = ctc_decoder(
    tokens=tokens_file,
    lexicon=None,
    # lexicon = './ressources/lexicon_v7.txt',
    beam_size=128,
    beam_threshold=10,
    beam_size_token=10,
    nbest=50,
    log_add=True,
    blank_token='_', 
    sil_token='|', 
    unk_word='<unk>',
)



with torch.no_grad():

    def inference(audiofile):
        # raise ValueError("This is a custom error message")
        waveform,sr = load(audiofile,normalize=True)

        outputs,_ = model(waveform,tensor([len(waveform[0])]))

        results_array = decoder(outputs)[0]
        transcriptions = []
        scores = []
        for result in results_array:
            transcription = "".join(decoder.idxs_to_tokens(result.tokens))
            transcription = "".join(transcription.split('_'))
            transcription = "".join(transcription.split('|'))
            transcription = " ".join(transcription.split("▁"))
            transcription = transcription.strip()
            transcription = ' '.join(transcription.split())
            transcription = re.sub(r'-{2,}', '-', transcription)
            transcriptions.append(transcription)
            transcription = " ".join(sp_lm.EncodeAsPieces(transcription))
            transcription.replace("- ",'-').replace(" -",'-')
            lm_score = lm.score(transcription)
            score = lm_score*0.25+result.score*0.75
            scores.append(score)

        transcription = transcriptions[np.argmax(scores, axis=0)]
        return(transcription)

    # print(transcription)
    # output_words.append(output_word)
       