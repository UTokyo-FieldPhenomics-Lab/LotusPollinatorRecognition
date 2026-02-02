# Standard imports
import os
from pathlib import Path


def create_symlinks(img_list, base_dir, class_remapping):
    os.makedirs(base_dir / "images", exist_ok=True)
    os.makedirs(base_dir / "labels", exist_ok=True)

    for img_path in img_list:
        img_path = Path(img_path)

        target_img_path = base_dir / "images" / img_path.name
        if target_img_path.is_symlink() or target_img_path.exists():
            target_img_path.unlink()
        os.symlink(img_path, target_img_path)

        label_path = Path(str(img_path.with_suffix('.txt')).replace("images", "labels"))

        if label_path.exists():
            target_label_path = base_dir / "labels" / label_path.name

            with open(label_path, 'r') as f_in, open(target_label_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    original_class_id = int(parts[0])
                    new_class_id = class_remapping.get(original_class_id)

                    if new_class_id is not None:
                        new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                        f_out.write(new_line)
