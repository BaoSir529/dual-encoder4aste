import json
import tqdm
import numpy as np


#
# def word2idx(word):
#
#     return 0


def glove_embed(word2idx):
    f = open(r'..\embedding\glove.840B.300d.txt', 'rb')
    embeddings_dict = {}
    for line in f:
        values = line.split()
        word = values[0].strip().decode('utf-8')
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

    glove_embedding = []
    unk_embedding = embeddings_dict.get('<unk>')
    unk_count = 0
    for word in word2idx:
        embedding = embeddings_dict.get(word)
        if embedding is not None:
            glove_embedding.append(embedding)
        else:
            glove_embedding.append(unk_embedding)
            unk_count += 1

    glove_embedding = np.array(glove_embedding)

    f.close()
    del embeddings_dict
    print('GloVe is OK !')

    return glove_embedding


def review_embed(word2idx):
    f = open(r'C:\Embedding\amazon_product_review_corpus.particle_verbs.cbow.w5.d500.txt', 'rb')
    embeddings_dict = {}
    for line in f:
        values = line.split()
        word = values[0].strip().decode('utf-8')
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

    review_embedding = []
    unk_embedding = embeddings_dict.get('</s>')
    unk_count = 0
    for word in word2idx:
        embedding = embeddings_dict.get(word)
        if embedding is not None:
            review_embedding.append(embedding)
        else:
            review_embedding.append(unk_embedding)
            unk_count += 1

    review_embedding = np.array(review_embedding)

    f.close()
    del embeddings_dict
    print('Amazon is OK!')

    return review_embedding


def process(path):
    with open(path+r'\train_triplets.txt', encoding='utf-8') as f:
        train = f.readlines()
    with open(path+r'\dev_triplets.txt', encoding='utf-8') as f:
        dev = f.readlines()
    with open(path+r'\test_triplets.txt', encoding='utf-8') as f:
        test = f.readlines()

    text = train + dev + test
    word2idx = {'<unk>': 0}
    idx2word = {0: '<unk>'}
    index = 1
    for sentence_pack in text:
        sentence, _ = sentence_pack.strip().split("####")
        for word in sentence.strip().split():
            if word.lower() not in word2idx:
                word2idx[word.lower()] = index
                idx2word[index] = word.lower()
                index += 1

    with open(path+r'\word2idx.json', 'w') as f:
        json.dump(word2idx, f)
    with open(path+r'\idx2word.json', 'w') as f:
        json.dump(idx2word, f)

    glove_embedding = glove_embed(word2idx)
    np.save(path+r'\glove.npy', glove_embedding)
    del glove_embedding

    amazon_embedding = review_embed(word2idx)
    np.save(path+r'\amazon.npy', amazon_embedding)
    del amazon_embedding

    return 0


if __name__ == '__main__':
    process(r'.\V2\14lap')
    # process(r'.\V2\14res')
    # process(r'.\V2\15res')
    # process(r'.\V2\16res')

