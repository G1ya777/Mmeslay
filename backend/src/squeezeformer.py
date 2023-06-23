from data_loading import train_dataloader,validation_dataloader
from pytorch_lightning import LightningModule
from torch.nn.functional import ctc_loss,log_softmax
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
import torch
import datetime
import os
from torch.optim import RAdam
from torchaudio.models.decoder import ctc_decoder
torch.set_float32_matmul_precision("high")
import re
from torchmetrics.functional import word_error_rate,char_error_rate
import sentencepiece as spm
from nemo.collections.asr.modules import SqueezeformerEncoder,ConvASRDecoder
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules import SpectrogramAugmentation
from nemo.core import typecheck
typecheck.set_typecheck_enabled(False) 


sp = spm.SentencePieceProcessor()
sp.load("ressources/tokenizer/128_v7.model")


lr = 2e-4
none_count=0


tokens_file = 'ressources/tokenizer/128_v7.txt'

decoder = ctc_decoder(
    lexicon=None,
    tokens=tokens_file,
    beam_size=1,
    beam_threshold=1,
    # lm = kenlm_path,
    beam_size_token=1,
    nbest=1,
    log_add=True,
    blank_token='_', 
    sil_token='|', 
    unk_word='<unk>',
)


class MySqueezeformer(LightningModule):


    def __init__(self,lr=lr):
        super(MySqueezeformer, self).__init__()

        # self.squeezeformer = Squeezeformer(
        #     input_dim=80,
        #     encoder_dim=144,
        #     num_attention_heads=4,
        #     num_encoder_layers=16,
        #     num_classes=128,
        #     input_dropout_p=0,
        #     conv_dropout_p=0,

        # )
        self.processor = AudioToMelSpectrogramPreprocessor(sample_rate=16000,features=80,n_fft=512,window_size=0.025,window_stride=0.01,log=True,frame_splicing=True)
        self.augmentation = SpectrogramAugmentation(2,5,27,0.05)
        self.encoder = SqueezeformerEncoder(feat_in=80,
                                       feat_out=-1,
                                       n_layers=16,
                                       d_model=144,
                                       adaptive_scale=True,
                                       time_reduce_idx=7,
                                       dropout_emb=0,
                                       dropout_att=0.1,
                                       subsampling_factor=4,
                                       )
        # self.encoder.load_state_dict(torch.load("pt_ckpt/encoder.ckpt"))
        self.decoder = ConvASRDecoder(feat_in=144,num_classes=128)

        self.lr = lr

    def forward(self, x, lengths):
        # logits,logits_lengths = self.squeezeformer(x,lengths)
        spec,lengths = self.processor.forward(x,lengths)
        if self.encoder.training:
            spec = self.augmentation.forward(spec,lengths)
        encoded = self.encoder.forward(spec,lengths)

        decoded = self.decoder.forward(encoded[0])
        logits_lengths = []
        for item in decoded:
            logits_lengths.append(len(item))


        return decoded,torch.tensor(logits_lengths)
        # return logits,logits_lengths
    
    def training_step(self, batch, batch_idx):
        spectrograms, transcriptions, specs_lengths, transcriptions_lengths = batch

        outputs, logits_lengths = self(spectrograms, specs_lengths)
        outputs = outputs.transpose(0, 1)
        outputs = log_softmax(outputs, dim=2)
        loss = ctc_loss(outputs, transcriptions, logits_lengths, transcriptions_lengths, blank=1)

        if loss.isnan() or loss.isinf():
            global none_count
            none_count+=1
            self.log('N_c', float(none_count), prog_bar=True,sync_dist=True)
            
            return None
    
        self.log("loss",loss,sync_dist=True,on_epoch=True,on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        spectrograms, transcriptions, specs_lengths, transcriptions_lengths = batch
        with torch.no_grad():
            
            outputs,logits_lengths = self(spectrograms, specs_lengths)
            all_transcriptions=[]
            all_targets=[]
            
            for target, i in zip(transcriptions, range(len(transcriptions))):
                target_sentence = sp.DecodeIds(torch.Tensor.tolist(target[:transcriptions_lengths[i]]))
                all_targets.append(target_sentence)
            
            
            for i in range(len(outputs)):
                result = decoder(outputs[i].to('cpu').unsqueeze(0))[0][0]
                result = "".join(decoder.idxs_to_tokens(result.tokens))
                transcription = "".join(result.split('_'))
                transcription = "".join(transcription.split('|'))
                transcription = " ".join(transcription.split("▁"))
                transcription = transcription.strip()
                transcription = ' '.join(transcription.split())
                transcription = re.sub(r'-{2,}', '-', transcription)
                all_transcriptions.append(transcription)

                        

            wer = word_error_rate(all_transcriptions, all_targets)
            cer = char_error_rate(all_transcriptions, all_targets)
            outputs = outputs.transpose(0, 1)
            outputs = log_softmax(outputs, dim=2)
            val_loss = ctc_loss(outputs, transcriptions, logits_lengths, transcriptions_lengths,blank=1,zero_infinity=True)
        self.log("val_loss",val_loss,sync_dist=True,on_epoch=True)
        self.log('wer', wer,sync_dist=True,prog_bar=True,on_epoch=True)
        self.log('cer', cer,sync_dist=True,on_epoch=True)
        # return wer
    

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # def on_train_epoch_start(self):
        
    def on_train_epoch_start(self) -> None:
        self.log('N_c', float(none_count), prog_bar=True,sync_dist=True,on_step=False,on_epoch=True)

    def training_epoch_end(self,outputs) -> None:
        global none_count
        none_count = 0
        checkpoint = {
            # 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        checkpoint_dir = './checkpoints_vZ2/sched_ckpt/'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_' + str(datetime.datetime.now()) + '.pt')
        torch.save(checkpoint, checkpoint_path)
        


    def configure_optimizers(self):
        self.optimizer =RAdam(self.parameters(), lr=self.lr,betas=[0.9, 0.98],weight_decay=1e-6,eps=1e-9)
        # self.scheduler = CosineAnnealingLR(self.optimizer,300,eta_min=1e-6)
        
        # ckpt_path = './checkpoints_vZ2/sched_ckpt/checkpoint_2023-05-29 18:25:30.591458.pt'
        # checkpoint = torch.load(ckpt_path)
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 


        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
        }
    

if __name__ == '__main__':
    callbacks= [
        LearningRateMonitor(logging_interval='epoch'),
        # GradientAccumulationScheduler(scheduling={5:2,20:4,40:8}),
        ModelCheckpoint(dirpath="./checkpoints_vZ2/val_loss",verbose=False,save_on_train_epoch_end=True,save_top_k=1,save_last=True,monitor='val_loss'),
        ModelCheckpoint(dirpath="./checkpoints_vZ2/wer",verbose=False,save_on_train_epoch_end=True,save_top_k=1,save_last=False,monitor='wer'),
        ModelCheckpoint(dirpath="./checkpoints_vZ2/cer",verbose=False,save_on_train_epoch_end=True,save_top_k=1,save_last=False,monitor='cer'),
        ]



    model = MySqueezeformer()
    trainer = Trainer(accelerator='auto',
    precision="bf16",
    callbacks=callbacks,
    default_root_dir="./checkpoints_vZ2wSA/logs",
    reload_dataloaders_every_n_epochs=1,
    max_epochs=300,
    )
    
    # torch.save(model.state_dict(),"squeezeformer")
    # trainer.fit(model, train_dataloaders=train_dataloader,val_dataloaders=validation_dataloader)

