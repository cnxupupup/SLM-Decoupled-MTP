SAVEROOT="YOUR_SAVE_ROOT"
ckpt_step=54
mkdir -p $SAVEROOT
mkdir -p $SAVEROOT/logs
python infer.py \
--model-path "YOUR_MODEL_PATH" \
--save-dir "$SAVEROOT/${ckpt_step}k" \
--logger-path "$SAVEROOT/logs/${ckpt_step}k.log" \
--text-path "YOUR_DIR_OF_TEXT" \
--wav-ref-path "YOUR_WAV_REF_PATH" \
--task "tts"