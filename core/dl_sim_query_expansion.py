import numpy as np
import sys, argparse, json
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

def prepare_data(query_elmo_embed, psg_elmo_embed):
    x = []
    psg_id_list = list(psg_elmo_embed[()].keys())
    query_id_list = list(query_elmo_embed[()].keys())
    for q in query_id_list:
        qvec = query_elmo_embed[()][q]
        for p in psg_id_list:
            pvec = psg_elmo_embed[()][p]
            x.append(np.hstack((qvec, pvec)))
    return np.array(x), query_id_list, psg_id_list

def get_query_para_dl_scores(model, query_elmo, psg_elmo, vec_len):
    Xdata, query_list, psg_list = prepare_data(query_elmo, psg_elmo)
    yhat = model.predict([Xdata[:, :vec_len], Xdata[:, vec_len:]], verbose=0)
    i = 0
    query_psg_score_dict = dict()
    for q in query_list:
        query_psg_score_dict[q] = dict()
        for p in psg_list:
            query_psg_score_dict[q][p] = float(yhat[i][0])
            i += 1
    return query_psg_score_dict

def expand_query(query_psg_scores, query_text_dict, psg_text_dict):
    expanded_query_dict = dict()
    for q in query_psg_scores.keys():
        top_psg = ''
        top_score = 0
        for p in query_psg_scores[q].keys():
            if query_psg_scores[q][p] > top_score:
                top_psg = p
                top_score = query_psg_scores[q][p]
        expanded_query_dict[q] = query_text_dict[str(q)] + ". " + psg_text_dict[top_psg]
    return expanded_query_dict

def main():
    parser = argparse.ArgumentParser(description='Produce query expansion dict using pretrained Dense-Siamese model')
    parser.add_argument('-q', '--query_tsv', required=True, help='Path to query tsv file')
    parser.add_argument('-p', '--psg_text', required=True, help='Path to psg text dict')
    parser.add_argument('-qe', '--query_elmo', required=True, help='Path to query elmo embed dict')
    parser.add_argument('-pe', '--psg_elmo', required=True, help='Path to psg elmo embed dict')
    parser.add_argument('-vl', '--vec_len', type=int, required=True, help='Length of elmo vector')
    parser.add_argument('-m', '--model', required=True, help='Path to pretrained model')
    parser.add_argument('-o', '--out', required=True, help='Path to output file')
    args = vars(parser.parse_args())
    query_text_file = args['query_tsv']
    psg_text_file = args['psg_text']
    query_elmo_file = args['query_elmo']
    psg_elmo_file = args['psg_elmo']
    vec_len = args['vec_len']
    model_file = args['model']
    outfile = args['out']

    query_text_dict = dict()
    with open(query_text_file, 'r') as qf:
        for l in qf:
            query_text_dict[l.split('\t')[0]] = l.split('\t')[1]
    with open(psg_text_file, 'r') as pf:
        psg_text_dict = json.load(pf)
    query_elmo = np.load(query_elmo_file, allow_pickle=True)
    psg_elmo = np.load(psg_elmo_file, allow_pickle=True)
    print("Data loaded")
    model = load_model(model_file, custom_objects={'precision': precision, 'recall': recall,
                                                   'fmeasure': fmeasure, 'fbeta_score': fbeta_score})
    print("Model loaded, going to get query-para scores")

    query_psg_score_dict = get_query_para_dl_scores(model, query_elmo, psg_elmo, vec_len)
    print("Going to expand query")
    expanded_query_dict = expand_query(query_psg_score_dict, query_text_dict, psg_text_dict)
    with open(outfile, 'w') as out:
        for q in expanded_query_dict.keys():
            out.write(str(q).replace('\n', ' ')+'\t'+expanded_query_dict[q].replace('\n', ' ')+'\n')
    print("Done")

if __name__ == '__main__':
    main()