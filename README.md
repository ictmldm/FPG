# FPG : Fact-Preserved Personalized News Headline Generation
The PyTorch open source implementation for the ICDM 2023 paper "Fact-Preserved Personalized News Headline Generation".

[arXiv link](https://arxiv.org/abs/2501.11828)
[IEEE Xplore link](https://ieeexplore.ieee.org/document/10415680)

![intro-1](https://github.com/ictmldm/FPG/assets/152248858/f039a607-cbc0-43a0-a5bb-3c84594e39dc)

Please reach us via emails or via github issues for any enquiries!

## Citation

Please cite our work if you find it useful for your research and work.

```
@inproceedings{yang2023fact,
  title = {Fact-Preserved Personalized News Headline Generation},
  author = {Zhao Yang and Junhong Lian and Xiang Ao},
  booktitle = {2023 IEEE International Conference on Data Mining (ICDM)},
  pages = {1493--1498},
  year = {2023},
  organization = {IEEE}
}
```

## Abstract
Personalized news headline generation, aiming at generating user-specific headlines based on readers' preferences, burgeons a recent flourishing research direction. Existing studies generally inject a user interest embedding into an encoder-decoder headline generator to make the output personalized, while the factual consistency of headlines is inadequate to be verified. In this paper, we propose a framework **F**act-Preserved **P**ersonalized News Headline **G**eneration (short for FPG), to prompt a tradeoff between personalization and consistency. In FPG, the similarity between the candidate news to be exposed and the historical clicked news is used to give different levels of attention to key facts in the candidate news, and the similarity scores help to learn a fact-aware global user embedding. Besides, an additional training procedure based on contrastive learning is devised to further enhance the factual consistency of generated headlines. Extensive experiments conducted on a real-world benchmark [PENS](https://msnews.github.io/pens.html) validate the superiority of FPG, especially on the tradeoff between personalization and factual consistency.


## The Framework of FPG
![overall-new](https://github.com/ictmldm/FPG/assets/152248858/ab438a8f-a303-4f74-bbb9-c12d87cc1b49)

## Requirements
Install requirements (in the cloned repository):

```
pip3 install -r requirements.txt
```

## Update

[2025-05-26]  We are pleased to announce that we have released the model implementation and most of the essential code. The remaining code will be available soon.

[2025-05-21]  We would like to acknowledge that the comments in this project have been automatically regenerated using `Gemini-2.5-Flash`.