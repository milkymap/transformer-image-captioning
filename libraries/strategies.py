import io 
import json
from multiprocessing.sharedctypes import Value 
import pickle as pk 

import clip 

import cv2 
import numpy as np 
import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import operator as op 
import itertools as it, functools as ft 

from os import path 
from glob import glob 

from PIL import Image 
from rich.progress import track 
from torch.nn.utils.rnn import pad_sequence
from torchvision import models 
from torchvision import transforms as T 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator 

from libraries.log import logger 

SPECIALS2IDX = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}

def pull_files(endpoint, extension):
    file_paths = sorted(glob(path.join(endpoint, extension)))
    return file_paths

def build_tokenizer(tok_name='spacy', lang='en_core_web_sm'):
    tokenizer = get_tokenizer(tokenizer=tok_name, language=lang)
    return tokenizer 

def yield_tokens(data_iter, tokenizer):
    for sample in track(data_iter, description=f'tokenization process'):
        sample = sample.strip().lower()  # remove trailing keys and lowercase 
        yield tokenizer(sample)
    
def make_vocab(data_iter, tokenizer, map_specials2idx):
    vocab = build_vocab_from_iterator(
        iterator=yield_tokens(data_iter, tokenizer), 
        specials=list(map_specials2idx.keys()), 
        min_freq=1, 
        special_first=True
    )
    vocab.set_default_index(map_specials2idx['<unk>'])  # index of the <unk> token 
    return vocab 

def serialize(path2dump, data):
    with open(path2dump, mode='wb') as fp:
        pk.dump(data, fp)

def deserialize(path2dump):
    with open(path2dump, mode='rb') as fp:
        return pk.load(fp)

def serialize(path2dump, data):
    with open(path2dump, mode='wb') as fp:
        pk.dump(data, fp)

def deserialize(path2dump):
    with open(path2dump, mode='rb') as fp:
        return pk.load(fp)

def load_vectorizer(path2models):
    if path.isfile(path2models):
        features_extractor = th.load(path2models)
    else:
        model_name = path.split(path2models)[1]
        real_name, _ = model_name.split('.')
        endpoint = op.attrgetter(real_name)(models)
        if endpoint is not None:
            features_extractor = endpoint(pretrained=True, progress=True)
            features_extractor = nn.Sequential(*list(features_extractor.children())[:-2])
            for prm in features_extractor.parameters():
                prm.requires_grad = False 
            th.save(features_extractor, path2models)
        else:
            raise Value(f'{real_name} is not a valid option for torchvision.models')
    return features_extractor

def load_ranker(path2models, device):
    if path.isfile(path2models):
        with open(path2models, 'rb') as fp:
            model, processor = pk.load(fp)  # (model, processor)
    else:
        model, processor = clip.load("ViT-B/32")
        with open(path2models, 'wb') as fp:
            pk.dump((model, processor), fp)
    return model.to(device), processor 

def pull_images(path2images, exts='*.jpg'):
    return sorted( glob(path.join(path2images, '**' ,exts), recursive=True) )

def th2cv(th_image):
    red, green, blue = th_image.numpy()
    return cv2.merge((blue, green, red))

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def cv2pil(cv_image):
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

def pil2cv(pil_image):
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

def read_image(path2image, size=None):
    pl_image = Image.open(path2image).convert('RGB')
    cv_image = cv2.cvtColor(np.array(pl_image), cv2.COLOR_RGB2BGR)
    if size is not None:
        return cv2.resize(cv_image, size, interpolation=cv2.INTER_CUBIC)
    return cv_image

def save_image(cv_image, path2location):
    cv2.imwrite(path2location, cv_image)

def prepare_image(th_image):
    normalied_th_image = th_image / th.max(th_image)
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(normalied_th_image)

def extract_features(extractor, batch_of_images):
    with th.no_grad():
        features = extractor(batch_of_images)
        return features 

def rank_solutions(pil_image, sentences, ranker, processor, device):
    image = processor(pil_image).unsqueeze(0).to(device)
    tokens = clip.tokenize(sentences).to(device)
    with th.no_grad():
        logits, _ = ranker(image, tokens)
        probabilities = th.softmax(logits, dim=1).cpu().squeeze(0)
        lowest = th.min(probabilities)
        largest = th.max(probabilities)
        normalized_scores = (probabilities - lowest) / (largest - lowest)
        return normalized_scores.tolist()

def custom_fn(batch):
    features, token_ids = list(zip(*batch)) 
    features = th.stack(features)     # 3d 
    token_ids = list(token_ids)
    token_ids = pad_sequence(token_ids, batch_first=True, padding_value=SPECIALS2IDX['<pad>'])
    return features, token_ids 

def build_mask(seq):
    seq_length = seq.shape[1]
    mask = np.fromfunction(lambda i,j: j > i, shape=(seq_length, seq_length))
    return th.as_tensor(mask) 

def build_key_padding_mask(seq, pad_idx):
    seq_key_padding_mask = (seq == pad_idx)
    return seq_key_padding_mask

def greedy_search(model, source, BOS, EOS, max_len, device):
    memory = model.encode(source.to(device))
    target = th.tensor([[BOS]])
    keep_generating = True 
    while keep_generating:
        output = model.decode(target.to(device), memory).squeeze(0)
        logits = model.generator(output[-1, :])
        scaled_logits = th.log_softmax(logits, dim=-1).squeeze(0)
        candidate = th.argmax(scaled_logits)
        target = th.cat([target, th.tensor([[candidate]])], dim=1)
        keep_generating = (candidate != EOS) and (target.shape[1] < max_len)
    return th.flatten(target) 

def beam_search(model, source, BOS, EOS, max_len, device, beam_width, alpha=0.7):
    memory = model.encode(source.to(device))
    target = th.tensor([[BOS]])
    with th.no_grad():
        output = model.decode(target.to(device), memory)
        output = th.stack([ model.generator(out[0, -1, :]) for out in output ])
        logits = th.mean(output, dim=0)

    scaled_logits = th.log_softmax(logits[None, :], dim=1).cpu().squeeze(0) # over vocab size 
    weights, candidates = th.topk(input=scaled_logits, k=beam_width, largest=True)
    
    response_tracker = []  # for valid final sequence 
    sequence_tracker = []  # for current active sequence
    for idx in candidates:
        option = th.tensor([[idx]])  # a new option into the search tree 
        sequence = th.cat([target, option], dim=1)
        sequence_tracker.append(sequence)
    
    keep_generating = True 
    while keep_generating:
        input_batch = th.vstack(sequence_tracker)
        with th.no_grad():
            input_memory = [m.repeat(input_batch.shape[0], 1, 1) for m in memory ]
            output = model.decode(input_batch.to(device), input_memory)
            logits = th.mean(th.stack([ model.generator(out[:, -1, :]) for out in output ]), dim=0)
            
        scaled_logits = th.log_softmax(logits, dim=1).cpu()
        
        length = input_batch.shape[1]
        vocab_size = scaled_logits.shape[1]
        weighted_logits = (scaled_logits + weights[:, None]) / length ** alpha  
        weights, candidates = th.topk(th.flatten(weighted_logits), k=beam_width, largest=True)
        weights = weights * length ** alpha  # denormalize

        weights_tmp = []
        sequence_tmp = []
        for idx, pos in enumerate(candidates):
            row = th.div(pos, vocab_size, rounding_mode='floor') # get relative position over nb_sequences 
            col = pos % vocab_size  # get relative position over vocab_size 
            sequence = th.cat([sequence_tracker[row], th.tensor([[col]])], dim=1)
            if col == EOS:
                logger.success('a sentence was generated :)')
                flattened_sequence = th.flatten(sequence).tolist()
                sequence_score = weights[idx] / len(flattened_sequence) ** alpha 
                response_tracker.append((flattened_sequence, sequence_score))  # a sentence was built 
                if len(response_tracker) == beam_width:
                    keep_generating = False 
                    break  # end the for loop over candidates
            elif sequence.shape[1] < max_len - 1:
                weights_tmp.append(weights[row])
                sequence_tmp.append(sequence)
        # end for loop over candidates ...!

        if len(sequence_tmp) == 0: 
            keep_generating = False 
        else:               
            weights = th.tensor(weights_tmp)
            sequence_tracker = sequence_tmp
    # end while search loop ...! 
    return response_tracker          
