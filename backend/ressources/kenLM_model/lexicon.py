import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("/mount/Data/PFE_FRESH/char.model")


with open ("/mount/Data/PFE_FRESH/kenLM_model/kab_v5.arpa",'r')as i:
    with open("lexicon_v5.txt","w") as o:

        for line in i :
            line = line.split()
            if len(line)==3:
                line = line[1]
                line_tokenized = " ".join(sp.EncodeAsPieces(line)[1:])
                line = str(line+' '+line_tokenized+' |\n')
                o.write(line)



# with open("lexicon_v5.txt",'r') as i:
#     with open("lm_words_v5.txt",'w')as o:
#         for line in i :
#             o.write("".join((line).split()[0])+"\n")



# with open("lexicon_v4.txt",'r') as i:
#     with open("lexicon_v4_2.txt",'w')as o:
#         for line in i :
#             test_valid = sp.EncodeAsIds(line.split()[0])
#             test_valid = sp.DecodeIds(test_valid)
#             if "⁇" not in test_valid:
#                 o.write(line)




# with open("lexicon_v4_2.txt",'r')as i:
#     with open("lexicon_v4_2_words.txt",'w')as o:
#         for line in i :
#                 o.write(str(line.split()[0])+"\n")


