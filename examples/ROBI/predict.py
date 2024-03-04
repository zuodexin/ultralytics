import os
from tqdm import tqdm
from ultralytics import YOLO
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument(
        "--model",
        type=str,
        default="./runs/detect/train/weights/best.pt",
        help="model.pt path(s)",
    )
    # img_dir
    parser.add_argument(
        "--img_dir",
        type=str,
        default="./robi_yolo/images/test/",
        help="directory of images",
    )

    # output_dir
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="directory of output",
    )
    # conf
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="object confidence threshold",
    )

    args = parser.parse_args()

    model = YOLO(args.model)

    results = model(args.img_dir, conf=args.conf)

    vis_dir = f"{args.output_dir}/vis"
    txt_dir = f"{args.output_dir}/txt"

    if not os.path.exists(f"{vis_dir}"):
        os.makedirs(f"{vis_dir}", exist_ok=True)

    if not os.path.exists(f"{txt_dir}"):
        os.makedirs(f"{txt_dir}", exist_ok=True)

    for i, r in enumerate(tqdm(results)):
        basename = os.path.basename(r.path).split(".")[0]
        r.save(f"{vis_dir}/{basename}.png")
        r.save_txt(f"{txt_dir}/{basename}.txt")
