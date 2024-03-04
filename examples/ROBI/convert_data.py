import os
import shutil
from ultralytics.data.converter import convert_coco

robi_syn_root = "../../../data/ROBI/bop/coco/robi_20000"
robi_real_root = "../../../data/ROBI/bop/coco/robi_10000"
save_dir = "robi_yolo"

convert_coco(
    f"{robi_syn_root}/annotations",
    save_dir="robi_20000",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)
convert_coco(
    f"{robi_real_root}/annotations",
    save_dir="robi_10000",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)

os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/labels", exist_ok=True)
shutil.move("robi_20000/labels/train", f"{save_dir}/labels")
shutil.move("robi_20000/labels/vali", f"{save_dir}/labels")
shutil.move("robi_10000/labels/test", f"{save_dir}/labels")

shutil.rmtree(f"robi_20000")
shutil.rmtree(f"robi_10000")


print("Copying images to the new directory")
shutil.copytree(f"{robi_syn_root}/train", f"{save_dir}/images/train")
shutil.copytree(f"{robi_syn_root}/vali", f"{save_dir}/images/vali")
shutil.copytree(f"{robi_real_root}/test", f"{save_dir}/images/test")

print("Done")
