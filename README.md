# dual-encoder4aste
Code to support the paper “Jiang Baoxing, Liang Shehui, Liu Peiyu, Dong Kaifang & Li Hongye, A semantically enhanced dual encoder for aspect sentiment triplet extraction, Neurocomputing, 2023”

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

## Usage

* Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and unzip `glove.840B.300d.txt` into `./embedding/GloVe/`.
* Download pretrained "Word Embedding of Amazon Product Review Corpus" with this [link](https://zenodo.org/record/3370051) and unzip into `./embedding/Particular/`.
* Prepare data for models, run the code [v1_data_process.py](./data/v1_data_process.py) and [v1_word2ids.py](./data/v1_word2ids.py), same for the v2 dataset.
```bash
python ./data/v1_data_process.py
python ./data/v1_word2ids.py
```
* You can train the model using the corresponding .sh file [./code/bash/V1/](./code/bash/V1/) or [./code/bash/V2/](./code/bash/V2/).
* For example:
```bash
python ./code/bash/V1/aste_14lap.sh
```

## Citation

If you use the code in your paper, please kindly star this repo and cite our paper.

## Note
* Code of this repo heavily relies on [BDTF-ABSA](https://github.com/HITSZ-HLT/BDTF-ABSA).
