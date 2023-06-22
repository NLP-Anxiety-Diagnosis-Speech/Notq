import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
import statistics
import numpy as np
from transformers import BertModel, BertTokenizerFast
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm


tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()


def get_emb(sentences):
    sent_inputs = tokenizer(sentences, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**sent_inputs)

    embeddings = outputs.pooler_output
    return embeddings


def get_all_emb(df, type):
    data = df.to_numpy()
    out_data = []

    for i in tqdm(data):
        i[0] = i[0].replace('"', '')
        i[0] = i[0].replace('\n', '')
        tmp = i[0].split('.')
        emb = get_emb(i[0], type)
        out_data.append([i[0], i[1], i[2], emb])
    return out_data


def cal_single_metrics(name, embeddings):
    if name == 'euclidean':
        return np.linalg.norm(embeddings[0]-embeddings[1])

    elif name == 'cosine':
        return dot(embeddings[0], embeddings[1])/(norm(embeddings[0])*norm(embeddings[1]))


def get_emb(sentences, type):
    if type == 'labse':
        sent_inputs = tokenizer(sentences, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**sent_inputs)

        embeddings = outputs.pooler_output
        return embeddings


def get_8x8(df, type):
    data = df.to_numpy()
    out_data = []

    for i in tqdm(data):
        i[0] = i[0].replace('"', '')
        i[0] = i[0].replace('\n', '')

    for i in tqdm(range(0, len(data), 8)):
        tmp_euclidean = np.zeros((8, 8))
        tmp_cosine = np.zeros((8, 8))
        embs = []
        for j in range(8):
            embs.append(get_emb(data[i+j][0], type))

        for j in range(8):
            for k in range(8):
                tmp_euclidean[j][k] = cal_single_metrics(
                    'euclidean', [embs[j][0], embs[k][0]])
                tmp_cosine[j][k] = cal_single_metrics(
                    'cosine', [embs[j][0], embs[k][0]], type)
        tmp_euclidean = tmp_euclidean[np.triu_indices(8, k=1)]
        tmp_cosine = tmp_cosine[np.triu_indices(8, k=1)]

        out_data.append([tmp_euclidean, tmp_cosine, data[i][1]])
    return out_data


def cal_metrics(name, embeddings):
    if name == 'euclidean':
        tmp = []
        for i in range(len(embeddings)-1):
            tmp.append(np.linalg.norm(embeddings[i]-embeddings[i+1]))
        if len(tmp) > 1:
            return statistics.mean(tmp), statistics.stdev(tmp)
        else:
            return statistics.mean(tmp), None

    elif name == 'cosine':
        tmp = []
        for i in range(len(embeddings)-1):
            tmp.append(dot(embeddings[i], embeddings[i+1]) /
                       (norm(embeddings[i])*norm(embeddings[i+1])))
        if len(tmp) > 1:
            return statistics.mean(tmp), statistics.stdev(tmp)
        else:
            return statistics.mean(tmp), None

    elif name == 'euclidean_sum':
        s = 0
        for i in range(len(embeddings)-1):
            s += np.linalg.norm(embeddings[i]-embeddings[i+1])
        return s

    elif name == 'cosine_sum':
        s = 0
        for i in range(len(embeddings)-1):
            s += dot(embeddings[i], embeddings[i+1]) / \
                (norm(embeddings[i])*norm(embeddings[i+1]))
        return s


def get_jump(df, type):
    data = df.to_numpy()
    out_data = []

    for i in tqdm(data):
        i[0] = i[0].replace('"', '')
        i[0] = i[0].replace('\n', '')
        tmp = i[0].split('.')
        tmp = [j for j in tmp if j != '']
        if len(tmp) > 1:
            emb = get_emb(tmp)
            euclidean = cal_metrics('euclidean', emb)
            euclidean_sum = cal_metrics('euclidean_sum', emb)
            cosine = cal_metrics('cosine', emb)
            cosine_sum = cal_metrics('cosine_sum', emb)

            out_data.append([i[0], i[1], i[2], euclidean[0], euclidean[1],
                            euclidean_sum, cosine[0], cosine[1], cosine_sum])
        else:
            out_data.append(
                [i[0], i[1], i[2], None, None, None, None, None, None])
