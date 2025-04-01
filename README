# EditLord: Learning Code Transformation Rules for Code Editing

![workflow](imgs/workflow.png)

## Augmentation Process
See files in `preprocess/` for more details.

## Finetuning for Editing Process
```bash
accelerate launch finetune_codellama.py \
    --base_model $BASE_MODEL \
    --data_path ./data/ \
    --output_dir $OUTPUT_DIR \
    --batch_size 32 \
    --micro_batch_size 2 \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --cutoff_len 2000 \
    --train_on_inputs False \
    --use_flash_attention True \
    --train_name $TRAIN_FILE \
    --val_name $VAL_FILE
    --test_name $TEST_FILE \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --task $TASK \
    --method $METHOD 
```
## Evaluation
The evaluation consists of three subtasks: 1. performance optimization, 2. decompilation, and 3. security hardening.

### Performance Optimization
To generate the LLM response:
```bash
python eval/eval_perf.py \
    --test_file $TEST_FILE \
    --output_file $OUTPUT_FILE \
    --do_sample True \
    --num_samples 8 \
    --num_threads 4 \
    --method $METHOD \
    --task performance \
    --temperature 0.7 \
    --base_url $BASE_URL \
    --fine_tuned_model $MODEL \
    --api_key $API_KEY
```

To evaluate the performance optimization, please refer to [PIE](https://github.com/LearningOpt/pie) to set up the environment.


### Decompilation
To generate and evaluate the LLM response:
```bash
python eval/eval_decompile.py \
    --output_dir $OUTPUT_DIR \
    --test_file $TEST_FILE \
    --output_file $OUTPUT_FILE \
    --do_sample True \
    --num_samples 1 \
    --num_threads 4 \
    --method $METHOD \
    --temperature 1 \
    --fine_tuned_model $MODEL \
    --base_url $BASE_URL \
    --api_key $API_KEY
```

### Security Hardening
To generate the LLM response:
```bash
python eval/eval_sec.py \
    --model_path $MODEL_PATH \
    --n_gpu 4 \
    --num_workers 16 \
    --temperature 0.8 \
    --method $METHOD \
    --num_gen 50 \
    --output_name $OUTPUT_DIR \
    --eval_type edit \
    --output_file $OUTPUT_FILE \
    --base_url $BASE_URL \
    --api_key $API_KEY
```

To evaluate the security hardenig, please refer to [CWEval](https://github.com/Co1lin/CWEval) to set up the environment.

