
import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
import os
from collections import Counter
import string
import re
from tqdm.auto import tqdm
from  data_processing import load_data


UNANSWERABLE_REPLIES = ["unanswerable","unanswer", "n/a", "idk", "i don't know", "not known", "answer not in context", "the answer is unknown","cannot find", "no answer", "unknown", "none","none of the above", "none of the above choices", "it is unknown", "no answer is given", "no answer is provided", "no answer is available", "no answer is given in the text", "no answer is provided in the text", "no answer is available in the text", "no answer is given in the passage", "no answer is provided in the passage", "no answer is available in the passage", "no answer is given in the context", "no answer is provided in the context", "no answer is available in the context", "no answer is given in the paragraph", "no answer is provided in the paragraph", "no answer is available in the paragraph", "no answer is given in the article", "no answer is provided in the article", "no answer is available in the article", "no answer is given in the story", "no answer is provided in the story", "no answer is available in the story", "no answer is given in the document", "no answer is provided in the document", "no answer is available in the document", "no answer is given in the passage", "no answer is provided in the passage", "no answer is available in the passage", "no answer is given in the context", "no answer is provided in the context", "no answer is available in the context", "no answer is given in the paragraph", "no answer is provided in the paragraph", "no answer is available in the paragraph", "no answer is given in the article", "no answer is provided in the article", "no answer is available in the article", "no answer is given in the story", "no answer is provided in the story", "no answer is available in the story", "no answer is given in the document", "no answer is provided in the document", "no answer is available in the document", "no answer is given in the passage", "no answer is provided in the passage", "no answer is available in the passage", "no answer is given in the context", "no answer is provided in the context", "no answer is available in the context", "no answer is given in the paragraph", "no answer is provided in the paragraph", "no answer is available in the paragraph", "no answer is given in the article", "no answer is provided in the article", "no answer is available in"]
UNANSWERABLE_REPLIES_EXACT = ['nan','none', 'no information','unknown', 'no answer', 'it is unknown', "the answer is unknown", 'none of the above choices', 'none of the above']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_output(output):
    clean_out = ""
    # Extract the first output, convert to lowercase and strip whitespace
    text = output.lower().strip()
    try:
        if any(reply in text for reply in UNANSWERABLE_REPLIES):
            return  ""
        first_segment = text.split('answer',1)[1]
        first_segment = first_segment.split('\n')[0]
        first_segment = first_segment.split('.')[0]
        cleaned_segment = re.sub(r'[^\w\s]', '', first_segment)
        if cleaned_segment in UNANSWERABLE_REPLIES_EXACT:
            clean_out = ""
        # Check if the response contains any of the 'unanswerable' keywords
        elif any(reply in cleaned_segment for reply in UNANSWERABLE_REPLIES):
            clean_out = ""
        else:
            clean_out = cleaned_segment  # Append the cleaned first segment if it's considered 'answerable'
    except:
        return ""
    # Check if the response is exactly one of the 'unanswerable' replies
    return  clean_out


