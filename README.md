# RWKV-Lite: Deeply Compressed RWKV for Resource-Constrained Devices

RWKV-Lite is a suite of compression techniques to reduce the system memory usage on runtime. 

## Training

### Set up environments for training
```
# Create a conda environment
conda create -n rwkv python=3.10

# Install python requirements
pip install -r requirements.txt
```

### SVD Training

Step 1.  create a workspace directory

```
mkdir -p RWKV-v5/out/04b-x58
```

Step 2. copy over all training scripts

```
cd RWKV-v5
cp template/*.sh out/04b-x58
```
Files to change: 

* `model-config.sh` to set model variant, # of layers, etc. 
* `run-train.sh` to change training hyperparams, e.g. learning rate, etc
* `run-eval.sh` to change evaluation hyperparams, etc

Step 3. prepare a dataset: See [Dataset](#dataset).

Step 4. initialize a model
```
cd {your-path}/RWKV-v5/out/04b-x58
bash prep.sh
```

Step 5. run a training script
```
# check the current dir
pwd 
{your-path}/RWKV-Lite/RWKV-v5/out/04b-x58

# run a script
bash run-train.sh
```

### Sparsity training
Step 1. Collect FFN data. See [Sparsity dataset](#sparsity-dataset).

Step 2. Change file paths
```bash
pwd
~/RWKV-LM-/RWKV-v5

vim src/train-ffn.py
# Change the following paths
# in_model_file=f"{RWKV_HOME}/out/04b-x58/04b-x58.pth"
# outpath=f"{RWKV_HOME}/out/04b-x58/"
# out_model_file=f"{RWKV_HOME}/out/04b-x58/04b-x58-ffn.pth"
```

Step 3. Run a script
```bash
python src/train-ffn.py
```

### Hierarchical head training
Step 1. Cluster vocabulary: In `svd.py`, `decompose_emb()` function has $K$ variable, which determines the number of clusters. You may want to change it. The default is 200.
```bash
pwd
~/RWKV-LM/RWKV-v5

python src/svd.py --decompose 2 --orig_model out/04b-x58/04b-x58
[ 642 5161  ... 203  503  128  177   40]

# Check your output file
ls out/04b-x58/
... 04b-x58-cls.npy ...
```

Step 2. Edit a training script and change environment variables
```bash
# out/04b-x58/model-config.sh
HEAD_K = 200 # The number is the same as we set $K$ in svd.py

# out/04b-x58/run-train.sh
# add the following flags at the last line of the script
 --head_K $HEAD_K \
 --load_token_cls "$PROJ_DIR/rwkv-cls.npy" \
 --load_partial 1
```

Step 3. Make symbolic links for the cluster and model
```bash
pwd
~/RWKV-LM/RWKV-v5/out/04b-x58/

ln -s 04b-x58-cls.npy rwkv-cls.npy
ln -s 04b-x58-mlp.pth rwkv-init.pth
```

Step 4. Run a script
```bash
pwd
~/RWKV-LM/RWKV-v5/out/04b-x58/

run-train.sh

# You will see such outputs
will train: head_l1.weight
will train: head_l1fc1.weight
will train: head_l1fc2.weight
...

  | Name       | Type          | Params
---------------------------------------------
0 | emb        | Embedding     | 67.1 M
1 | blocks     | ModuleList    | 233 M
2 | ln_out     | LayerNorm     | 2.0 K
3 | head       | Linear        | 67.1 M
4 | head_l1    | Linear        | 204 K
5 | head_l1fc1 | Linear        | 1.0 M
6 | head_l1fc2 | Linear        | 204 K
7 | head_l2    | ParameterList | 67.1 M
---------------------------------------------
1.5 M     Trainable params  # Notice that this is our trainable parameters for hiearchical head
434 M     Non-trainable params
435 M     Total params

```


## Inference
### Evaluation
Step 0. Create a symbolic link in `src`
```
pwd 
~/RWKV-Lite
ln -s $(pwd)/rwkv RWKV-v5/src/
```

Step 1. Install `lm_eval`

```
cd {your-path}/RWKV-Lite
bash scripts/install-lm-eval.sh
```

Step 2. Run a script
```
cd out/04b-x58
bash run-eval.sh
```

### Example: a simple ChatBot
```bash
pwd
~/RWKV-LM/RWKV-v5

export RWKV_HOME=$(pwd)
python src/test-rwkv-chat.py

Elon Musk has made a real case for the possibility of owning a Tesla, the company he founded in 2002 and co-founded with Elon Musk’s son, Elon Musk Sr. Tesla’s shares soared from $100 on April 4 to over $120 in the first three days of trading on Friday, a significant climb from its low point.
```

### Inference (Raspberry Pi 5)

### Turn on our feats: sparsity / hiearchical head / lazy embedding

```python
# Ensemble sparsity FFN variables
quant_bit = 1
quant_map = [0.95] * 24
mlp_map = [0.7] * 24

# Hiearchical head path
hh_on = True
hh_path = f"{RWKV_HOME}/out/04b-x58/04b-x58-cls.npy"

# Lazy embedding
emb_on = True

t0 = time.time()
model = RWKV(model=model_path,
             strategy=strategy,
             quant_bit=quant_bit,     # Sparse FFN
             quant_map=quant_map,     # Sparse FFN
             mlp_map=mlp_map,         # Sparse FFN
             load_token_cls=hh_path,  # Hiearchical head
             on_cluster_head=hh_on,   # Hiearchical head
             lazy_emb=emb_on,         # Lazy embedding
             verbose=True)
```

## Dataset
### A toy example
Step 1. Get `minipile` dataset
```bash
bash RWKV-v5/getdata.sh
ls RWKV-v5/data
minipile.bin    minipile.idx
```

### Sparsity dataset
Step 1. Set an environment variable
```bash
pwd
~/RWKV-LM/RWKV-v5

export RWKV_HOME=$(pwd)
```

Step 2. Run a collecting script
> Make sure you have the `rwkv` symbolic link in `src/`
```bash
pwd
~/RWKV-LM/RWKV-v5

# You need to set your input model and output path.
python src/collect-sparse-data.py
```

Step 3. Check your data
```bash
# Your output path
ls <your-output-path>
FFN.key-layer0-query.npy    # Random input datum
FFN.key-layer0-weights.npy  # FFN weights
...
FFN.key-layerN-query.npy    # Random input datum
FFN.key-layerN-weights.npy  # FFN weights
```


## Model information
| paramter size | # of layers | Embedding dim |
| ------------- | ----------- | ------------- |
| 0.1B          | 12          | 768           |
| 0.4B          | 24          | 1024          |
| 1.5B          | 24          | 2048          |
| 3B            | 32          | 2560          |
| 7B            | 32          | 4096          |

## Troubleshooting
### Issue: GPU
```bash
zation -std=c++17 -c /data/home/bfr4xr/RWKV-LM/RWKV-v5/src/rwkv/cuda/operators.cu -o operators.cuda.o
/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’:
  435 |         function(_Functor&& __f)
      |                                                                                                                                                 ^
/usr/include/c++/11/bits/std_function.h:435:145: note:         ‘_ArgTypes’
/usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:
  530 |         operator=(_Functor&& __f)
      |                                                                                                                                                  ^
/usr/include/c++/11/bits/std_function.h:530:146: note:         ‘_ArgTypes’
ninja: build stopped: subcommand failed.
```
This is because of CUDA. Please do `source env.sh` for setting `CUDA_HOME`


## TODO
- Training
    - [x] SVD training
    - [x] Sparsity FFN training
    - [x] Hierarchical head training

- Inference
  - [x] A toy example: inference e.g., chat
    - [x] Hierarchical head example
    - [x] Sparsity FFN example
    - [x] Embedding example
  - [ ] NEON instruction
  - [ ] RPI inference
    
- Evaluation
    - [x] run `lm-evaluation-harness` example

- Data collection
    - [x] A toy example: minipile
    - [x] Sparsity data collection
    - [ ] General dataset preparation e.g., pile

- ETC
  - [] Clutter unused codes or private comments