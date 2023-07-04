# StyleGAN
StyleGAN from scratch (pytorch)

> [Paper Review](https://inhopp.github.io/paper/Paper21/)

![error](https://github.com/inhopp/inhopp/assets/96368476/1024f773-9f2b-4e9a-aaf7-0e51f1e8514e){: width="60%" height="70%"}

- 학습 잘 하는 줄 알았는데 inference 결과가 에러... (fix 예정)

## Repository Directory 

``` python 
├── StyleGAN
        ├── datasets
        │    
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── option.py
        ├── model.py
        ├── train.py
        ├── inference.py
        └── README.md
```

- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get item
- `model.py` : Define block and construct Model
- `option.py` : Environment setting

<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/StyleGAN.git
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --lr {}(default: 0.0002) \
    --n_epoch {}(default: 30) \
    --num_workers {}(default: 4) \
```

### testset inference
``` python
python3 inference.py
    --device {}(defautl: cpu) \
    --num_workers {}(default: 4) \
```


<br>


#### Main Reference
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master