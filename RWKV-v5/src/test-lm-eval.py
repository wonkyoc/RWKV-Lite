# to run: 
# python3.10 src/test-lm-eval.py

import os,sys
import run_lm_eval

# os.environ["RWKV_CUDA_ON"] = '0'   # for x58, x59, we dont have cuda custom ops

if __name__ == "__main__":
    if len(sys.argv)>1:
        path=sys.argv[1]
    res = run_lm_eval.do_eval(path, isverbose=False, benchmarks=["lambada_openai"])
    print(f"test-lm-eval {sys.argv[1]}")
    print(res)

'''
# test if res is cacahed, below 
run_lm_eval.clean_cache()

# cached???
res = run_lm_eval.do_eval(path)
print(res)

'''