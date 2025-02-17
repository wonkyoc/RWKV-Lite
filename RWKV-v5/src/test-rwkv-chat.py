'''
test rwkv inference engine
cf: https://pypi.org/project/rwkv/
'''
import sys, os
import time

# Run chat app on the inference engine (rwkv), check for sanity 

if os.environ.get("RWKV_JIT_ON") != '0':
    os.environ["RWKV_JIT_ON"] = '1'

if os.environ.get('RWKV_CUDA_ON') != '0':
    os.environ["RWKV_CUDA_ON"] = '1' #default

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS, print_memory_usage
from rwkv.arm_plat import is_amd_cpu

import os

RWKV_HOME = os.environ.get("RWKV_HOME") # User specific. See env-amd.sh
model_path=f'{RWKV_HOME}/out/04b-x58/04b-x58-full'

print(f'Loading model - {model_path}')

# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32

if os.environ["RWKV_CUDA_ON"] == '1':
    strategy='cuda fp16'
else:
    if is_amd_cpu():
        strategy='cpu fp32'  # amd cpu lacks hard fp16...
    else:
         strategy='cpu fp16'

# use below to quantize model & save
if False: 
    strategy_token = strategy.split()[1]
    basename, extension = os.path.splitext(os.path.basename(model_path))
    save_path = os.path.join(os.path.dirname(model_path), f"{basename}_{strategy_token}{extension}")
    print(f'Save path: {save_path}')
    model = RWKV(model=model_path, strategy=strategy, verbose=True, convert_and_save_and_exit=save_path)
    sys.exit(0)

#print_memory_usage("before model build")

# Sparsity FFN variables
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
             quant_bit=quant_bit, 
             quant_map=quant_map, 
             mlp_map=mlp_map,
             load_token_cls=hh_path,
             on_cluster_head=hh_on,
             lazy_emb=emb_on,
             verbose=True)

#print_memory_usage("before pipeline build")

pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# ex prompt from paper: https://arxiv.org/pdf/2305.07759
ctx = "\nElon Musk has"
print(ctx, end='')

cnt=0

def my_print(s):
    global cnt
    cnt += 1
    print(s, end='', flush=True)
    #if cnt % 50 == 0:
        #print_memory_usage("\n")

t1 = time.time()

# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

#print_memory_usage("before generate")

TOKEN_CNT = 100 
pipeline.generate(ctx, token_count=TOKEN_CNT, args=args, callback=my_print)
print('\n')

t2 = time.time()

print(f"model build: {(t1-t0):.2f} sec, exec {TOKEN_CNT} tokens in {(t2-t1):.2f} sec, {TOKEN_CNT/(t2-t1):.2f} tok/sec")

if model.stat_runs != 0:
    print(f"stats: runs: {model.stat_runs} \
        cls/run {model.stat_loaded_cls/model.stat_runs:.2f} \
        tokens/run {model.stat_loaded_tokens/model.stat_runs/65535:.2f}")
