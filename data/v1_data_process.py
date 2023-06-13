import nltk
import json
import spacy
import numpy
import pickle
from spacy.tokens import Doc

sent_dic = {0: 'NEU', 1: 'POS', 2: 'NEG'}


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def dependency_adj_matrix(text):
    tokens = nlp(text)
    words = text.split()
    matrix = numpy.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix


def POS_generate(text):
    tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    # pos = [tup[1] for tup in tags]
    pos = [0] * len(pos_tags)
    for i, tags in enumerate(pos_tags):
        word, tag = tags
        if tag.startswith('NN'):  # 名词
            pos[i] = 1
        elif tag.startswith('VB'):  # 动词
            pos[i] = 2
        elif tag.startswith('JJ'):  # 形容词
            pos[i] = 3
        elif tag.startswith('RB'):
            pos[i] = 4
    return pos


def process(filename):
    with open(filename[:11] + filename[5:10] + '_pair/' + filename[11:-4] + '_pair.pkl', 'rb') as f:
        pair = pickle.load(f)
    with open(filename, encoding='utf-8') as f:
        text = f.readlines()
    assert len(pair) == len(text)
    all_data = []
    count = 0
    for sentence_pack in text:
        data = {'ID': count}
        sentence = sentence_pack.strip().split('####')[0]
        sentence = sentence.strip()
        triplets = pair[count]
        data['sentence'] = sentence
        data['tokens'] = str(sentence.split())
        pos = POS_generate(sentence)
        data['pos_tags'] = str(pos)
        data['pairs'] = []
        for triplet in triplets:
            data['pairs'].append([triplet[0][0], triplet[0][-1] + 1, triplet[1][0], triplet[1][-1] + 1, sent_dic[triplet[2]]])
        entities = []
        for entity in data['pairs']:
            tar = [sentence.split()[i] for i in range(entity[0], entity[1])]
            opn = [sentence.split()[i] for i in range(entity[2], entity[3])]
            entities.append(["target", entity[0], entity[1], str(tar), ' '.join(tar)])
            entities.append(["opinion", entity[2], entity[3], str(opn), ' '.join(opn)])
        data['entities'] = entities
        adj_matrix = dependency_adj_matrix(sentence)
        adj_matrix = adj_matrix.tolist()
        data['adj'] = adj_matrix
        all_data.append(data)
        # print(count)
        assert len(adj_matrix) == len(sentence.split())
        assert len(pos) == len(sentence.split())
        count += 1

    with open(filename[:-4] + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(all_data, ensure_ascii=False, indent=1))
        # json.dump(all_data, f)
    print(filename[:-4] + ' is ok')


if __name__ == '__main__':
    process('./V1/14lap/train.txt')
    process('./V1/14lap/dev.txt')
    process('./V1/14lap/test.txt')

    process('./V1/14res/train.txt')
    process('./V1/14res/dev.txt')
    process('./V1/14res/test.txt')

    process('./V1/15res/train.txt')
    process('./V1/15res/dev.txt')
    process('./V1/15res/test.txt')

    process('./V1/16res/train.txt')
    process('./V1/16res/dev.txt')
    process('./V1/16res/test.txt')
