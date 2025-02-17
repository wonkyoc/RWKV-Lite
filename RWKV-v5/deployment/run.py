import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ["RWKV_JIT_ON"] = '1'
RWKV_HOME = os.environ.get('RWKV_HOME')

if os.environ.get('RWKV_CUDA_ON') != '0':
    os.environ["RWKV_CUDA_ON"] = '1' #default
os.environ["RWKV_CUDA_ON"] = '0' #default

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

models = [
        #'models/official-0.1b',
        #'models/official-0.4b',
        #'models/official-1.5b'
        'models/01b-x59',
        #'models/04b-x59'
        ]
cls_models = [
        'models/01b-x59-cls.npy',
        #'models/04b-x59-cls.npy',
        #'models/1b5-x59-cls.npy',
        ]

def my_print(s):
    pass
    #print(s, end='', flush=True)

token_limit = 200

if __name__ == "__main__":
    isverbose = True

    for model_path, cls_path in zip(models, cls_models):
        # 8/26/24: using fp16 will make some benchmarks (eg openai) nan... so use fp32
        if "official" in model_path:
            # official
            model = RWKV(model=model_path, strategy='cpu fp16', verbose=isverbose)
        else:
            # x59
            quant_bit = 1
            quant_map = [0.9] * 24
            mlp_map = [0.7] * 24
            model = RWKV(model=model_path, strategy='cpu fp16', verbose=isverbose,
                         quant_bit=quant_bit, quant_map=quant_map, mlp_map=mlp_map,
                         load_token_cls=cls_path)
        #print(f'Loading model - {model_path}')
        pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

        # borrow from RWKV_CHAT
        args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                             alpha_frequency = 0.25,
                             alpha_presence = 0.25,
                             alpha_decay = 0.996, # gradually decay the penalty
                             token_ban = [0], # ban the generation of some tokens
                             token_stop = [], # stop generation whenever you see any token here
                             chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

        total_fwd_t = 0
        total_att_t = 0
        total_ffn_t = 0
        total_cls_t = 0
        
        ctx = "\nAlice was so tired when she got back home so she went"
        iterations = 5
        for _ in range(iterations):
            pipeline.generate(ctx, token_count=token_limit, args=args, callback=my_print)
        total_fwd_t = (model.stat_time_fwd - model.stat_time_quant) / iterations
        total_att_t = model.stat_time_att / iterations
        total_ffn_t = (model.stat_time_ffn - model.stat_time_quant) / iterations
        total_cls_t = model.stat_time_cls / iterations

        if isverbose: 
            # last exec for now
            print(f"stats: runs: {model.stat_runs/iterations} \
            cls/run {model.stat_loaded_cls/model.stat_runs/iterations:.2f} \
            avg %loaded {model.stat_loaded_tokens/model.stat_runs/65535:.2f}")
            print(f"forward {total_fwd_t:.2f} (s)")
            print(f"att {total_att_t:.2f}")
            print(f"ffn {total_ffn_t:.2f}")
            print(f"\tmlp {model.stat_time_mlp/iterations:.2f}")
            print(f"\t(excluded) quant {model.stat_time_quant/iterations:.2f}")
            print(f"\tffn: rx @ rw {model.stat_time_ffn_rx_rw/iterations:.2f}")
            print(f"\tffn: kx @ kw {model.stat_time_ffn_kx_kw/iterations:.2f}")
            print(f"\tffn: vx @ vw {model.stat_time_ffn_vx_vw/iterations:.2f}")
            print(f"cls {total_cls_t:.2f}")
            print(f"{(model.stat_runs/iterations)/total_fwd_t:.2f} token/s")
        else:
            print(f"model={model_path} fwd_t={total_fwd_t} att_t={total_att_t} ffn_t={total_ffn_t} cls_t={total_cls_t} tps={token_limit/total_fwd_t:.2f}")

        print("\a")   # audiable alert when done -- works on linux & Mac terminals.
