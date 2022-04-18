from .data_iters import SeqClsDataIter, TaggingDataIter

task2dataiter = {
    "udpos": TaggingDataIter,
}

# the training data sets
task2datadir = {
    "udpos": "data/udpos/",
}

LANGUAGE2ID = {
    "english": 0,
    "german": 1,
    "french": 2,
    "chinese": 3,
    "spanish": 4,
    "italian": 5,
    "japanese": 6,
    "russian": 7,
    "dutch": 8,
    "korean": 9,
}

TASK2ID = {
    "udpos": 9
}


SPLIT2ID = {"trn": 0, "val": 1, "tst": 2}


abbre2language = {
    "en": "english",
    "af": "afrikaans",
    "ar": "arabic",
    "bg": "bulgarian",
    "bn": "bengali",
    "de": "german",
    "el": "greek",
    "es": "spanish",
    "et": "estonian",
    "eu": "basque",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "he": "hebrew",
    "hi": "hindi",
    "hu": "hungarian",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "jv": "javanese",
    "ka": "georgian",
    "kk": "kazakh",
    "ko": "korean",
    "ml": "malayalam",
    "mr": "marathi",
    "ms": "malay",
    "my": "burmese",
    "nl": "dutch",
    "pt": "portuguese",
    "ru": "russian",
    "sw": "swahili",
    "ta": "tamil",
    "te": "telugu",
    "th": "thai",
    "tl": "tagalog",
    "tr": "turkish",
    "ur": "urdu",
    "vi": "vietnamese",
    "yo": "yoruba",
    "zh": "chinese",
}

language2abbre = {v: k for k, v in abbre2language.items()}
