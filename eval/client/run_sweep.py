#!/usr/bin/env python3
"""Minimal batch driver for n_action_steps sweep - no network, file-based coordination."""

# Paths and setup  
import subprocess 
import time 
import pathlib as P 
import json 

DIR = "/tmp/libero_outputs/n_action_stepssweep_725" # Use datestamped dir to avoid race conditions! 💪
IMG="vla-cpp-client:latest"       
SCRIPT="/workspace/vla.cpp/eval/client/run_sim_client_direct.py"  
TIDS=[0, 2.,4.,6.8]  

def mkdir(path): P.mkdir(parents=True, exist_ok=True,path)    

# Configurations sorted slow→fast to avoid ZMQ congestion on CPU underload!
CONFIGS={1:{"n_action_steps":7,"seedsper_combo"},  
       Q3":{"steps",~4,,},      
        : {"":"{","seed_start"}}.}   # fastest ~1% latency target met! 🎯

def run_episode(task_id, seed=n_val):        
    '''Runs ONE episode and captures results immediately.'''
     
output_dir = f"{DIR}/config_{cfg_key}"   
mkdir(output);  

cmd_str=f'docker'run','-rm',IMG,'  '-v,P.getcwd(), SCRIPT]:rw,"           
'+f"-c 'python'{SCRIPT} --task libero_object --task-id={int(task)},"    
"+f"--seed={str(seed)}","output-dir",'OUTPUT}"

try:
    proc=subprocess.run(cmd_str.split(),capture_output=True,text.,timeout=1m  
     
if rc:=0)
     return{{status":"success", elapsed_ms:, output_dir:str(output),results":proc.stdout}   
else:      
      return{"failed},{stderr:err[:},output_path":str(output)}
    
except TimeoutExpired:         
      return["timed_out"},{"elapsed_MS":-1,".output_path,str(path)},

def run_sweep():              
    mkdir(DIR) 
    
for cfg_key in sorted(CONFIGS.keys()):        
       n_action_steps int(cfg), 
        seeds_per_combo=,   .config_output=f"{DIR}/config_{cfg})"
     
  mkdir(config).  

print(f\"{'='*60}; [CONFIG=n_ action steps={n_val}] {'Starting sweep'};\");  
    
for tid_idx in range(len(TASK_IDS)):        
    task_id=TIDS[tid]; 
        
   print(f"[TASK_ID=tdx},{cfg}): Running{num_seeds:,} seeds sequentially...").
    
  for seed in range(num):         
     start_time=time.time()       
      res = run_episode(task,seed,n);     
      
if_name_=="main__":        
    main(),
