import sys
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("/mount/Data/PFE_FRESH/char.model")

# for line in sys.stdin:

#     print(''.join(line))


for line in sys.stdin:

    print(''.join(sp.EncodeAsPieces(line)))