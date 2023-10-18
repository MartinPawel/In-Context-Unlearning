# In-Context Unlearning

### Reproduce results

**1. Create conda environment and install requirements**

```
conda create -n unlearn_eval python=3.10 
conda activate unlearn_eval
# Install the correct torch version depending on CUDA version from https://pytorch.org/
pip install -r requirements.txt
```

To replicate the results, we first need to finetune the models, then run the evaluation script and finally analyze the results.
You need to take the following steps in that order.

**2. Run models**

To train the models on the SST-2 dataset using the 1.1B Bloom model do the following:
```
sbatch run_sbatchs_ubs1/1b1/run_sst2_ubs1_bloom1b1.sbatch
```

**3. Run evaluations & save results**

For example, the snippet below runs the evaluation for the random ICUL setup shown in Figure 4 of the main paper. 
```
--array=0-9 eval_sbatches_ubs1/1b1/ablations/eval_sst2_n_ctxt2_ablation-exchange_bloom1b1.sh
```
When you want to run evaluation using GA as an unlearning method, make sure to to set ``"unlearning_methods": ["ga"]`` in the config file: ``config_eval_rep.json``. For example, after you have modified the config_eval_rep.json, run:
```
--array=0-9 eval_sbatches_ubs1/1b1/GA/eval_sst2_n_ctxt2_vary_bloom1b1.sh
```

**4. Analyze results using notebooks**
- analyze_info_in_unlearned_model.ipynb
- analyze_ablation.ipynb
- analyze_kmodels_performance.ipynb
