import numpy as np
import os
from preprocessing import getSentenceData
from rnn import Model
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import sys
def get_train_data(
    filename: str,
    add_start_stop_states: bool = True,
) -> Tuple[List[List[str]], List[List[int]]]:
    filename = Path(filename)
    assert filename.is_file()

    with filename.open() as f:
        lines = f.readlines()
        f.close()

    lines = [line.strip().split() for line in lines]

    x: List[List[str]] = []  # list of list of tokens
    y: List[List[str]] = []  # list of list of labels

    tmp_x, tmp_y = [], []  # temporary list of tokens and labels

    for line in lines:
        if len(line) == 2:
            tmp_x.append(line[0])
            tmp_y.append(line[1])
        elif len(line) == 3:
            # Special token: ". ..."
            tmp_x.append(f"{line[0]} {line[1]}")
            tmp_y.append(line[2])
        elif len(line) == 0:
            x.append(tmp_x)

            if add_start_stop_states:
                tmp_y.insert(0, "<START>")
                tmp_y.append("<END>")

            y.append(tmp_y)
            tmp_x, tmp_y = [], []
        else:
            raise ValueError("Unexpected line format")

    return x, y

def get_test_data(filename: str) -> List[List[str]]:
    filename = Path(filename)
    assert filename.is_file()

    with filename.open() as f:
        lines = f.readlines()
        f.close()

    lines = [line.strip() for line in lines]
    x: List[List[str]] = []  # list of list of tokens
    tmp_x = []  # temporary list of tokens

    for line in lines:
        if len(line) == 0 and len(tmp_x) > 0:
            x.append(tmp_x)
            tmp_x = []
        else:
            tmp_x.append(line)

    if len(tmp_x) > 0:
        x.append(tmp_x)

    return x

def build_vocab(data: List[List[str]], use_unk: bool = False, ratio=1.0) -> Dict[str, int]:
    """
    Build vocabulary from training data.

    Args:
        data: list of list of tokens
        use_unk: should we add UNK token to the vocab
        ratio: (1-ratio) will be the the ratio of UNK tokens in the data
    return:
        dict of token to index
    """
    counter = Counter()
    for i in data:
        counter.update(i)
    vocab = [t for t, _ in counter.most_common()]
    totol_size = len(vocab)
    print('total number of different word: %d'%totol_size)
    vocab = [t for t, _ in counter.most_common(int(totol_size*ratio))]
    print('we use the top %d most common words as dictionary'%(len(vocab)))
    if use_unk:
        vocab.insert(0, "<UNK>")
    vocab = {t: i for i, t in enumerate(vocab)}
    return vocab

def token2id(x, vocab):
    if x in vocab: return vocab[x]
    else: return vocab["<UNK>"] #<UNK>

def save_prediction(path, X_str, Y_str):
    with open(path, 'w') as f:
        for i in range(len(X_str)):
            for j in range(len(X_str[i])):
                f.write(X_str[i][j] + " " + Y_str[i][j]+"\n")
            f.write("\n")

def train(model, X, Y, learning_rate, nepoch, evaluate_loss_after, X_test_str, X_test_id,runname):
    print(X.shape)
    print(Y.shape)
    num_examples_seen = 0
    losses = []
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_total_loss(X, Y)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("Training loss after num_examples_seen=%d epoch=%d: %f" % (num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            Y_pred_id = [model.predict(sentence) for sentence in X_test_id]
            Y_pred_str = [[output_id2token[id] for id in sentence]  for sentence in Y_pred_id]
            os.makedirs('../RU/result/%s/'%(runname), exist_ok=True)
            save_prediction('../RU/result/%s/dev_epoch%d.out'%(runname, epoch),X_test_str, Y_pred_str )
            model.save('../RU/result/%s/model_epoch%d/'%(runname, epoch))
        for i in range(len(Y)):
            model.sgd_step(X[i], Y[i], learning_rate)
            num_examples_seen += 1
            if (i % 100 ==0): print('epoch %d, iteration %d'%(epoch, i))
    return losses



runname = sys.argv[1]
lr = float(sys.argv[2])
hidden_dim = int(sys.argv[3])
truncate = int(sys.argv[4])
ratio = float(sys.argv[5])


X_train_str, Y_train_str = get_train_data('../RU/train', False)
input_token2id = build_vocab(X_train_str, True, ratio)
output_token2id = build_vocab(Y_train_str, False)
output_id2token = { id:token for token, id in output_token2id.items()}
X_train_id = [[ token2id(word, input_token2id)for word in sentence] for sentence in X_train_str]
Y_train_id =  [[ output_token2id[word]  for word in sentence] for sentence in Y_train_str]
X_train_id = np.asarray(X_train_id)
Y_train_id = np.asarray(Y_train_id)


#print(X_train.shape)
#print(y_train.shape)
#assert False
np.random.seed(10)
model = Model(len(input_token2id), 7, hidden_dim, truncate)
X_test_str = get_test_data('../RU/dev.in')
X_test_id = [[ token2id(word, input_token2id)for word in sentence] for sentence in X_test_str]
losses = train(model, X_train_id, Y_train_id, learning_rate=lr, nepoch=10, evaluate_loss_after=1, X_test_str=X_test_str, X_test_id=X_test_id, runname=runname )


