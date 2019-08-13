import numpy as np
import sys, argparse, json

import keras
from keras.models import load_model
import keras.backend as K

def precision(ytrue, yhat):
    true_pos = K.sum(K.round(K.clip(ytrue * yhat, 0, 1)))
    predicted_pos = K.sum(K.round(K.clip(yhat, 0, 1)))
    precision = true_pos / (predicted_pos + K.epsilon())
    return precision

def recall(ytrue, yhat):
    true_pos = K.sum(K.round(K.clip(ytrue * yhat, 0, 1)))
    possible_pos = K.sum(K.round(K.clip(ytrue, 0, 1)))
    recall = true_pos / (possible_pos + K.epsilon())
    return recall

def fbeta_score(ytrue, yhat, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(ytrue, 0, 1))) == 0:
        return 0

    p = precision(ytrue, yhat)
    r = recall(ytrue, yhat)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(ytrue, yhat):
    # also known as f1 measure
    return fbeta_score(ytrue, yhat, beta=1)

def prepare_data(query_elmo, passage_elmo, query_para_dict):
    x = []
    query_para_list = []
    for q in query_para_dict.keys():
        if q in query_elmo[()].keys():
            qvec = query_elmo[()][q]
        else:
            print('query id: '+q+' not present in elmo dict')
            sys.exit(1)
        for p in query_para_dict[q]:
            if p in passage_elmo[()].keys():
                pvec = passage_elmo[()][p]
                query_para_list.append(q+'_'+p)
            else:
                print('psg id: ' + p + ' not present in elmo dict')
                sys.exit(1)
            x.append(np.hstack((qvec, pvec)))
    return np.array(x), query_para_list

def get_query_para_dl_sim(model, vec_len, Xdata, q_para_list):
    yhat = model.predict([Xdata[:, :vec_len], Xdata[:, vec_len:]], verbose=0)
    q_para_dist = dict()
    for i in range(len(q_para_list)):
        q_para_dist[q_para_list[i]] = float(yhat[i][0])
    return q_para_dist

def main():
    parser = argparse.ArgumentParser(description='Produce query-passage pair distance dict using pretrained Dense-Siamese model')
    parser.add_argument('-r', '--run', help='Path to input run file')
    parser.add_argument('-m', '--model', help='Path to pretrained model file')
    parser.add_argument('-pe', '--psg_emb', help='Path to elmo embedding for passage')
    parser.add_argument('-qe', '--query_emb', help='Path to elmo embedding for query')
    parser.add_argument('-k', '--top_k', type=int, help='Top k ranked passages for which dist will be calculated')
    parser.add_argument('-vl', '--vec_len', type=int, help='Length of elmo vector')
    parser.add_argument('-o', '--out', help='Path to output file')
    args = vars(parser.parse_args())
    runfile = args['run']
    model_file = args['model']
    psg_emb_file = args['psg_emb']
    query_emb_file = args['query_emb']
    k = args['top_k']
    vec_len = args['vec_len']
    outfile = args['out']

    run_dict = dict()
    with open(runfile, 'r') as run:
        for l in run:
            q = l.split(' ')[0]
            p = l.split(' ')[2]
            if q in run_dict.keys():
                run_dict[q].append(p)
            else:
                run_dict[q] = [p]
    topk_run_dict = dict()
    for q in run_dict.keys():
        topk_run_dict[q] = run_dict[q][:k]
    psg_emb = np.load(psg_emb_file, allow_pickle=True)
    query_emb = np.load(query_emb_file, allow_pickle=True)
    Xdata, query_para_list = prepare_data(query_emb, psg_emb, topk_run_dict)
    model = load_model(model_file, custom_objects={'precision': precision, 'recall': recall,
                                                   'fmeasure': fmeasure, 'fbeta_score': fbeta_score})
    query_para_score = get_query_para_dl_sim(model, vec_len, Xdata, query_para_list)
    with open(outfile, 'w') as out:
        json.dump(query_para_score, out)