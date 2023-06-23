from Data_Loading_v2 import test_dataloader
from Squeezeformer import MySqueezeformer
from torchmetrics.functional import word_error_rate,char_error_rate
from torchaudio.models.decoder import ctc_decoder
import torch
import sentencepiece as spm
import numpy as np
import kenlm
import re 

lm = kenlm.Model('./ressources/kenLM_model/kab_5k_trigram.bin')
sp_lm = spm.SentencePieceProcessor()
sp_lm.load("ressources/tokenizer/5K.model")

sp = spm.SentencePieceProcessor()
sp.load("ressources/tokenizer/128_v7.model")


model = MySqueezeformer()
model.load_state_dict(torch.load("ressources/e2e_model/squeezeformer"),strict=False)
model.eval()

tokens_file = 'ressources/tokenizer/128_v7.txt'
# kenlm_path = "kenLM_model/kab_lex.bin"
# lexicon_path='lexicon_v7.txt'


decoder = ctc_decoder(
    tokens=tokens_file,
    lexicon=None,
    # lexicon=lexicon_path,
    # lm=kenlm_path,
    # lm_weight=1.25,
    beam_size=1,
    beam_threshold=1,
    beam_size_token=1,
    nbest=1,
    log_add=True,
    blank_token='_', 
    sil_token='|', 
    unk_word='<unk>',
)

with torch.no_grad():
    all_transcriptions = []
    all_targets = []
    for batch in test_dataloader:
        inputs, targets, input_lengths, target_lengths  = batch
        outputs, _ = model(inputs, input_lengths)
        results = []
        for i in range(len(outputs)):
            result = decoder(outputs[i].unsqueeze(0))[0]
            results.append(result)

        for target, i in zip(targets, range(len(targets))):
            target_sentence = sp.DecodeIds(torch.Tensor.tolist(target[:target_lengths[i]]))
            all_targets.append(target_sentence)

        for results_array in results:
            transcriptions = []
            scores = []
            for result in results_array:
                transcription = "".join(decoder.idxs_to_tokens(result.tokens))
                # transcription = sp.DecodeIdsWithCheck(torch.Tensor.tolist(result.tokens[1:-1]))
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
            print(transcription)
            all_transcriptions.append(transcription)
    # print(all_transcriptions)

    wer = word_error_rate(all_transcriptions, all_targets)
    cer = char_error_rate(all_transcriptions,all_targets)
    print("Average Word Error Rate: {:.2f}%".format(wer * 100))
    print("Average Character Error Rate: {:.2f}%".format(cer * 100))
