## MBHT
This is our implementation for our paper **Multi-Behavior Hypergraph-Enhanced Transformer for Next-Item Recommendation**, accepted by *KDD'22*.

## Requirements
The code is built on Pytorch and the [RecBole](https://github.com/RUCAIBox/RecBole) benchmark library. Run the following code to satisfy the requeiremnts by pip:

`pip install -r requirements.txt`


## Datasets
##### Download the three public datasets we use in the paper at:
https://drive.google.com/file/d/1OFT_5Xp_az-GSHIl7QEPB9zhulbooLzE/view?usp=sharing

##### Unzip the datasets and move them to *./dataset/*

## Run MBHT

`python run_MBHT.py --model=[MBHT] --dataset=[tmall_beh] --gpu_id=[0] --batch_size=[2048]`, where [value] means the default value.

## Tips
- Note that we modified the evaluation sampling setting in `recbole/sampler/sampler.py` to make it static.
- The model code is at `recbole/model/sequential_recommender/mbht.py`.
- Feel free to explore other baseline models provided by the RecBole library and directly run them to compare the performances.
