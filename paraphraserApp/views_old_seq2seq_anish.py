from django.shortcuts import render
from . import func
import pickle

import torch

from transformers import MT5Tokenizer
from transformers import MT5ForConditionalGeneration, AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import joblib
pl.seed_everything(42)


def generate(request):
    paraphrase_model = joblib.load('..\paraphrase_model.sav')
    tokenizer_name='google/mt5-small'
    tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)
    if request.method=='POST':
        sentence = str(request.POST['sentence'])
        print(sentence)
        inputs_encoding =  tokenizer(
            sentence,
            add_special_tokens=True,
            max_length= 256,
            padding = 'max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
            )
        generate_ids = paraphrase_model.generate(
            input_ids = inputs_encoding["input_ids"],
            attention_mask = inputs_encoding["attention_mask"],
            do_sample=True,
            max_length=64,
            top_k=40,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences = 1,
            no_repeat_ngram_size=2,
            )

        preds = [
            tokenizer.decode(gen_id,
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True)
            for gen_id in generate_ids
        ]

        par="".join(preds)
        print(par)
        return render(request,'main1.html',{'output_text':par,'input_text':sentence})
        
    return render(request,'main1.html')



# Create your views here.
# def generate(request):
#     if request.method=='POST':
#         sentence = str(request.POST['sentence'])
#         print(type(sentence))
#         paraphrase = paraphrase_model.generate_paraphrase(sentence)
#         print(sentence)
#         return render(request,'main.html',{'result':paraphrase},{'sen':sentence})
#     return render(request,'main.html')


# def formInfo(request):
#     sentence = str(request.POST['sentence'])
#     par = paraphrase_model.generate_paraphrase(sentence)
#     return render(request,'result.html',{{'paraphrase':par}})