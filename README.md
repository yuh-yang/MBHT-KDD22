## MBHT
This is the official implementation for our paper **Multi-Behavior Hypergraph-Enhanced Transformer for Next-Item Recommendation**, accepted by **KDD'22**.

## Requirements
The code is built on Pytorch and the [RecBole](https://github.com/RUCAIBox/RecBole) benchmark library. Run the following code to satisfy the requeiremnts by pip:

`pip install -r requirements.txt`


## Datasets
- Download the three public datasets we use in the paper at:
  https://drive.google.com/file/d/1OFT_5Xp_az-GSHIl7QEPB9zhulbooLzE/view?usp=sharing

- Unzip the datasets and move them to **./dataset/**

- You may also refer to the raw data here :)

  [IJCAI](https://tianchi.aliyun.com/dataset/42)

  [Taobao](https://tianchi.aliyun.com/dataset/649)

  [Retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

## Run MBHT

`python run_MBHT.py --model=[MBHT] --dataset=[tmall_beh] --gpu_id=[0] --batch_size=[2048]`, where [value] means the default value.

## Tips
- Note that we modified the evaluation sampling setting in `recbole/sampler/sampler.py` to make it static.
- The model code is at `recbole/model/sequential_recommender/mbht.py`.
- Feel free to explore other baseline models provided by the RecBole library and directly run them to compare the performances.

## Citation
If you find our work helpful, please kindly cite our research paper:
```
@inproceedings{yang2022mbht,
  title={Multi-behavior hypergraph-enhanced transformer for sequential recommendation},
  author={Yang, Yuhao and Huang, Chao and Xia, Lianghao and Liang, Yuxuan and Yu, Yanwei and Li, Chenliang},
  booktitle={Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining},
  pages={2263--2274},
  year={2022}
}
```
