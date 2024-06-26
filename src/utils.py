
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)