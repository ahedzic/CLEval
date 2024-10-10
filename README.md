# CLEval

After installing python requirement in requirements.txt and downloading data sets,

experiments for each data set can be reproduced by running:
./scripts/hyperparameters/cold_start/starcraft.sh
./scripts/hyperparameters/cold_start/flickr.sh
./scripts/hyperparameters/cold_start/yelp.sh
./scripts/hyperparameters/cold_start/reddit.sh

Individual model tests on specific datasets can be done by creating a separate script and copying the experiment you are interested in.

Our cold-start model implementations are in:
benchmarking/cold_start_model.py

WSLP model implementations are in:
benchmarking/baselines_models
benchmarking/gnn_model.py
benchmarking/get_heuristic.py

The original WSLP models from utilized from the HeaRT paper by Tang et al can be found here:
https://github.com/Juanhui28/HeaRT

Please cite their work if utilizing them:
@inproceedings{
  li2023evaluating,
  title={Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking},
  author={Li, Juanhui and Shomer, Harry and Mao, Haitao and Zeng, Shenglai and Ma, Yao and Shah, Neil and Tang, Jiliang and Yin, Dawei},
  booktitle={Neural Information Processing Systems {NeurIPS}, Datasets and Benchmarks Track},
  year={2023}
}