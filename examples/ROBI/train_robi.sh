export `<../../../.env`
set -x

# 
EXP=aug
EXP_DIR=./exps/${EXP}



function train() {
    # save the experiment settings
    if [ -d ${EXP_DIR} ]; then
        echo "Experiment directory ${EXP_DIR} already exists"
        exit 1
    else
        mkdir -p ${EXP_DIR}
        cp -r ./*.sh ${EXP_DIR}
        cp -r ./*.py ${EXP_DIR}
        cp -r ./*.yaml ${EXP_DIR}

        echo "settings_version: 0.0.4
datasets_dir: $(pwd)
weights_dir: ${EXP_DIR}/weights
runs_dir: ${EXP_DIR}/runs
uuid: 47dc55b504f844c507ea3a5235c3f239b398edb996961dce386951ca051e6205
sync: true
api_key: ''
openai_api_key: ''
clearml: true
comet: true
dvc: true
hub: true
mlflow: true
neptune: true
raytune: true
tensorboard: true
wandb: true" > ~/.config/Ultralytics/settings.yaml

        yolo detect train data=${EXP_DIR}/data.yaml model=yolov8m.pt epochs=100 imgsz=640 batch=48  copy_paste=0.5 mixup=0.5 degrees=180 hsv_s=1.0

    fi
}


train

# yolo detect val data=${EXP_DIR}/data.yaml model=./runs/detect/train/weights/best.pt imgsz=640 batch=1
# python predict.py --