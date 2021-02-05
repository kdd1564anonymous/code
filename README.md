requirements:

```text
python >= 3.7
pytorch == 1.4.0
torch-geometric == 1.4.3
tensorboard
matplotlib
scipy
sklearn
```

[nasbench101](https://github.com/google-research/nasbench)

[nasbench201](https://github.com/D-X-Y/NAS-Bench-201)

## Standalone Surrogate Evaluation

Generate dataset for offline training

```shell
python -m data.collect_data_201.py <path/to/nasbench101.tfrecord>  # for NAS-Bench-101 data

python -m data.collect_data_101.py <path/to/NAS-Bench-201.pth>  # for NAS-Bench-201 data
```

Run training and validation:

```shell
python -m predictor.train_offline --samples 256 --dataset 201
```

## Search Evaluation

```shell
# search on NAS-Bench-101
python -m search.search_101 --nas_bench_path <path/to/nasbench101.tfrecord>
# search on NAS-Bench-201
python -m search.search_201 --nas_bench_path <path/to/NAS-Bench-201.pth>
```

Use cli arguments `--repeats` and `--workers` for multi process multi trail running (on single GPU).


## Weight-sharing Evaluation

Weight-sharing evaluation requires NAS-Bench-201 and CIFAR-10 datasets.

Trained super-nets are provided in `data/supernet-201`.

Run search:

```shell
python -m search.search_ws --nas_bench_path <path/to/NAS-Bench-201.pth> --data_path <path/to/cifar-10> --load_checkpoint data/supernet-201 --repeat 4
```

This will run search process for 4 trails with different seeds.

To prepare super-net from scratch, run:

```shell
python -m search.search_ws --nas_bench_path <path/to/NAS-Bench-201.pth> --data_path <path/to/cifar-10> --repeat <num to repeat>
```

Trained super-net will be stored in `logs/search-ws/<time-stamp>`.

The CIFAR-10 split config `data/cifar-split.txt` is provided by [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201).
