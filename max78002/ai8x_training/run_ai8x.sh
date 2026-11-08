#!/bin/bash
# Runs the Analog Devices toolchain for MAX78002 in WSL/Linux (task 6.2):
#   bash run_ai8x.sh train | quantize | evaluate | synthesize | all
# Expects ai8x-training and ai8x-synthesis in ~/ai8x with their .venv folders (see README.md),
# and the data from export_data.py in ../../../ai8x_data.
set -eo pipefail

STEP=${1:-all}
EPOCHS=${EPOCHS:-60}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="$(cd "$REPO/../ai8x_data" && pwd)"
TRAINING=~/ai8x/ai8x-training
SYNTHESIS=~/ai8x/ai8x-synthesis
OUT="$REPO/max78002/synthesis_output"
QAT_CHECKPOINT=$SYNTHESIS/trained/ai87-ecg-rr-qat8.pth.tar
QUANTIZED=$SYNTHESIS/trained/ai87-ecg-rr-qat8-q.pth.tar
mkdir -p "$OUT" $SYNTHESIS/trained

if [ "$STEP" = train ] || [ "$STEP" = all ]; then
    echo "===== 1. Prepare and train with quantization-aware training"
    cp "$HERE/datasets/ecg_rr.py" $TRAINING/datasets/
    cp "$HERE/models/ai87net-ecg.py" $TRAINING/models/
    cp "$HERE/schedule_ecg.yaml" "$HERE/qat_policy_ecg.yaml" $TRAINING/policies/
    cd $TRAINING && . .venv/bin/activate
    python train.py --model ai87ecgnet --dataset ECG_RR --data "$DATA" --device MAX78002 --use-bias \
        --epochs $EPOCHS --optimizer Adam --lr 0.001 --wd 0 --deterministic --seed 42 \
        --compress policies/schedule_ecg.yaml --qat-policy policies/qat_policy_ecg.yaml \
        --confusion --name ecg_rr --out-dir logs --print-freq 100
    LOG_DIR=$(ls -td logs/ecg_rr___* | head -n 1)
    cp "$LOG_DIR"/ecg_rr_qat_best.pth.tar $QAT_CHECKPOINT
    cp "$LOG_DIR"/*.log "$OUT/train.log"
    deactivate
fi

if [ "$STEP" = quantize ] || [ "$STEP" = all ]; then
    echo "===== 2. Quantize"
    cd $SYNTHESIS && . .venv/bin/activate
    python quantize.py $QAT_CHECKPOINT $QUANTIZED --device MAX78002 -v 2>&1 | tee "$OUT/quantize.log"
    cp $QUANTIZED "$REPO/max78002/quantization_output/"
    deactivate
fi

if [ "$STEP" = evaluate ] || [ "$STEP" = all ]; then
    echo "===== 3. Evaluate the quantized model in 8-bit mode and save a sample input"
    cd $TRAINING && . .venv/bin/activate
    python train.py --model ai87ecgnet --dataset ECG_RR --data "$DATA" --device MAX78002 --use-bias \
        --confusion --evaluate --exp-load-weights-from $QUANTIZED -8 --save-sample 0 \
        --out-dir /tmp/ai8x_evaluate 2>&1 | tee "$OUT/evaluate.log"
    mv sample_ecg_rr.npy $SYNTHESIS/tests/
    deactivate
fi

if [ "$STEP" = synthesize ] || [ "$STEP" = all ]; then
    echo "===== 4. Synthesize C code for MAX78002"
    cd $SYNTHESIS && . .venv/bin/activate
    python ai8xize.py --test-dir synthed_net --prefix ecg_rr --checkpoint-file $QUANTIZED \
        --config-file "$HERE/ecg_rr.yaml" --sample-input tests/sample_ecg_rr.npy \
        --device MAX78002 --softmax --timer 0 --display-checkpoint --verbose --overwrite 2>&1 | tee "$OUT/synthesize.log"
    rm -rf "$OUT/ecg_rr"
    cp -r synthed_net/ecg_rr "$OUT/"
    deactivate
fi
