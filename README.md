# dual-encoder4aste
Code for our paper:
Baoxing Jiang, Shehui Liang, Peiyu Liu, Kaifang Dong, Hongye Li, A semantically enhanced dual encoder for aspect sentiment triplet extraction, Neurocomputing, Volume 562, 2023, 126917, [https://doi.org/10.1016/j.neucom.2023.126917](https://doi.org/10.1016/j.neucom.2023.126917)

## Abstract
Aspect sentiment triplet extraction (ASTE) is a crucial subtask of aspect-based sentiment analysis (ABSA) that aims to comprehensively identify sentiment triplets. Previous research has focused on enhancing ASTE through innovative table-filling strategies. However, these approaches often overlook the multi-perspective nature of language expressions, resulting in a loss of valuable interaction information between aspects and opinions. To address this limitation, we propose a framework that leverages both a basic encoder, primarily based on BERT, and a particular encoder comprising a Bi-LSTM network and graph convolutional network (GCN). The basic encoder captures the surface-level semantics of linguistic expressions, while the particular encoder extracts deeper semantics, including syntactic and lexical information. By modeling the dependency tree of comments and considering the part-of-speech and positional information of words, we aim to capture semantics that are more relevant to the underlying intentions of the sentences. An interaction strategy combines the semantics learned by the two encoders, enabling the fusion of multiple perspectives and facilitating a more comprehensive understanding of aspect–opinion relationships. Experiments conducted on benchmark datasets demonstrate the state-of-the-art performance of our proposed framework.

## Requirements
* Python 3.9
* PyTorch 1.12
* SpaCy 3.3.1
* numpy 1.21.5
* pytorch-lightning =1.3.5
* argparse 1.4.0
* scikit-learn 1.0.2
* einops =0.4.0
* transformers =4.15.0
* torchmetrics =0.7.0

## Data pre-processing stage (optional)

* Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and unzip `glove.840B.300d.txt` into `./embedding/GloVe/`.
* Download pretrained "Word Embedding of Amazon Product Review Corpus" with this [link](https://zenodo.org/record/3370051) and unzip into `./embedding/Particular/`.
* Prepare data for models, run the code [v1_data_process.py](./data/v1_data_process.py) and [v1_word2ids.py](./data/v1_word2ids.py), same for the v2 dataset.
* NOTE: For convenience, we have put with the processed data files in the data folder, so you can start the training phase directly.
```bash
python ./data/v1_data_process.py
python ./data/v1_word2ids.py
```
## Train stage
* You can train the model using the corresponding .sh file [./code/bash/V1/](./code/bash/V1/) or [./code/bash/V2/](./code/bash/V2/).
* For example:
```bash
bash aste_14lap.sh
```

## Citation

If our work has been helpful to you, please mark references to our work in your research and thank you for your support.

## Note
* Code of this repo heavily relies on [BDTF-ABSA](https://github.com/HITSZ-HLT/BDTF-ABSA).
