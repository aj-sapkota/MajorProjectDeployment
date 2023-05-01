def generate_paraphrase(question1):

    inputs_encoding =  tokenizer(
        question1,
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

    return "".join(preds)