# Retrieval-Augmented Primitive Representations for Compositional Zero-Shot Learning

* C. Jing, Y. Li, H. Chen, C. Shen, *Retrieval-Augmented Primitive Representations for Compositional Zero-Shot Learning*. in AAAI 2024 ([PDF](https://ojs.aaai.org/index.php/AAAI/article/view/28043/28096))


## Setup
```bash
conda create --name rapr python=3.8
conda activate rapr
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

## Download Dataset
We conduct experiments on commonly used MIT-States, UT-Zappos, and C-GQA. The datasets can be downloaded via 

```bash
sh download_data/download_data.sh
```
The databases for each dataset are in the directory of "data/database".

## Training

```sh
python train.py --dataset mit-states
```

## Evaluation

The evaluations are conducted in two settings: closed-world and open-world. 


```sh
python test.py --dataset mit-states
python test.py --dataset mit-states --open_world True
```

## Citation

```
@inproceedings{jing2024retrieval,
  title={Retrieval-Augmented Primitive Representations for Compositional Zero-Shot Learning},
  author={Jing, Chenchen and Li, Yukun and Chen, Hao and Shen, Chunhua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={3},
  pages={2652--2660},
  year={2024}
}
```

## Acknowledgement

The implementation of our method is partly based on the following codebases, [DFSP](https://github.com/Forest-art/DFSP) and [CZSL](https://github.com/ExplainableML/czsl). We gratefully thank the authors for their wonderful works.

