from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import pipeline


class sentiment:
    def __init__(self, mode):
        if mode=='Translation':
            model_size = "large"
            model_name = f"persiannlp/mt5-{model_size}-parsinlu-opus-translation_fa_en"
            self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
            self.sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
    def get_sent(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        res = self.model.generate(input_ids)
        output = self.tokenizer.batch_decode(res, skip_special_tokens=True)
        return self.sentiment_analysis(output)