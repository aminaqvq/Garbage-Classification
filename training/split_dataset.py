import csv
import json
import random
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

from PIL import Image


# =========================================================
# 基础配置区：主要改这里
# =========================================================

# 原始采集数据目录
# 你的采集程序默认保存到 dataset，所以这里默认就是 dataset
INPUT_DATASET_DIR = "dataset"

# 划分后的数据集输出目录
OUTPUT_DATASET_DIR = "garbage_dataset"

# 固定四大类，顺序很重要，后续训练和推理也要保持这个顺序
CLASS_NAMES = ["可回收", "有害", "厨余", "其他"]

# 数据集划分比例
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

# 随机种子
# 固定后，每次划分结果一致，方便复现实验
RANDOM_SEED = 42

# 支持的图片格式
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# 默认复制，不移动
# copy：复制图片，安全，推荐
# move：移动图片，会改变原始数据，不推荐
FILE_OPERATION_MODE = "copy"

# 是否计算每张图片的 SHA1
# 如果数据量很大，计算哈希会稍慢
# 想排查重复图片时可以改成 True
ENABLE_FILE_HASH = False


# =========================================================
# 工具函数
# =========================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def calculate_sha1(path: Path) -> str:
    """
    计算文件 SHA1，用于识别完全重复的文件。
    默认关闭，因为数据多时会稍微慢一点。
    """
    sha1 = hashlib.sha1()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha1.update(chunk)

    return sha1.hexdigest()


def check_image_valid(path: Path):
    """
    检查图片是否可以正常打开。
    返回：
    valid, width, height, mode, reason
    """
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode

        return True, width, height, mode, ""

    except Exception as e:
        return False, None, None, None, str(e)


def prepare_output_dir(output_dir: Path) -> Path:
    """
    准备输出目录。
    如果目录已经存在，给用户选择：
    1. 删除后重建
    2. 自动创建带时间戳的新目录
    3. 退出
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    print(f"\n输出目录已存在：{output_dir.resolve()}")
    print("请选择处理方式：")
    print("1. 删除该目录后重新划分")
    print("2. 保留原目录，自动创建一个新的输出目录")
    print("3. 退出程序")

    while True:
        choice = input("\n请输入 1 / 2 / 3：").strip()

        if choice == "1":
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"已清空并重建目录：{output_dir.resolve()}")
            return output_dir

        elif choice == "2":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_output_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
            new_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"已创建新目录：{new_output_dir.resolve()}")
            return new_output_dir

        elif choice == "3":
            print("程序已退出。")
            raise SystemExit

        else:
            print("输入无效，请重新输入。")


def create_split_folders(output_dir: Path):
    """
    创建 train / val / test 目录结构。
    """
    for split_name in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            class_dir = output_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)


def get_unique_dest_path(dest_dir: Path, original_name: str) -> Path:
    """
    防止目标目录中出现重名文件。
    如果目标文件已存在，则自动追加 _dup001。
    """
    original_path = Path(original_name)
    stem = original_path.stem
    suffix = original_path.suffix

    dest_path = dest_dir / original_name

    if not dest_path.exists():
        return dest_path

    index = 1
    while True:
        new_name = f"{stem}_dup{index:03d}{suffix}"
        new_path = dest_dir / new_name

        if not new_path.exists():
            return new_path

        index += 1


def calculate_split_counts(total_count: int):
    """
    根据比例计算 train / val / test 数量。

    对小样本做了保护：
    - 1 张：全部放 train
    - 2 张：1 train，1 val
    - 3 张及以上：尽量保证 train / val / test 都有数据
    """
    if total_count <= 0:
        return 0, 0, 0

    if total_count == 1:
        return 1, 0, 0

    if total_count == 2:
        return 1, 1, 0

    train_count = int(total_count * SPLIT_RATIOS["train"])
    val_count = int(total_count * SPLIT_RATIOS["val"])
    test_count = total_count - train_count - val_count

    if train_count <= 0:
        train_count = 1

    if val_count <= 0:
        val_count = 1

    if test_count <= 0:
        test_count = 1

    # 如果因为修正导致总数超了，就优先从 train 里扣
    while train_count + val_count + test_count > total_count:
        if train_count > 1:
            train_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            test_count -= 1

    # 如果因为取整导致总数少了，就补给 train
    while train_count + val_count + test_count < total_count:
        train_count += 1

    return train_count, val_count, test_count


def collect_images(input_dir: Path):
    """
    收集并检查四大类图片。
    返回：
    valid_images_by_class: 每类有效图片
    invalid_records: 无效图片日志
    """
    valid_images_by_class = {class_name: [] for class_name in CLASS_NAMES}
    invalid_records = []

    for class_name in CLASS_NAMES:
        class_dir = input_dir / class_name

        if not class_dir.exists():
            invalid_records.append({
                "time": now_str(),
                "class_name": class_name,
                "file_path": str(class_dir),
                "file_name": "",
                "reason": "类别文件夹不存在"
            })
            print(f"警告：类别文件夹不存在：{class_dir}")
            continue

        image_paths = [
            path for path in class_dir.rglob("*")
            if is_image_file(path)
        ]

        print(f"\n正在检查类别：{class_name}，发现图片文件 {len(image_paths)} 张")

        for image_path in image_paths:
            valid, width, height, mode, reason = check_image_valid(image_path)

            if valid:
                record = {
                    "class_name": class_name,
                    "source_path": image_path,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "file_size_bytes": image_path.stat().st_size,
                    "sha1": calculate_sha1(image_path) if ENABLE_FILE_HASH else ""
                }

                valid_images_by_class[class_name].append(record)

            else:
                invalid_records.append({
                    "time": now_str(),
                    "class_name": class_name,
                    "file_path": str(image_path),
                    "file_name": image_path.name,
                    "reason": reason
                })

    return valid_images_by_class, invalid_records


def split_class_images(image_records):
    """
    对某一个类别的图片进行随机划分。
    """
    records = image_records[:]
    random.shuffle(records)

    total_count = len(records)
    train_count, val_count, test_count = calculate_split_counts(total_count)

    train_records = records[:train_count]
    val_records = records[train_count:train_count + val_count]
    test_records = records[train_count + val_count:]

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records
    }


def copy_or_move_file(source_path: Path, dest_path: Path):
    """
    根据配置复制或移动文件。
    """
    if FILE_OPERATION_MODE == "copy":
        shutil.copy2(source_path, dest_path)
    elif FILE_OPERATION_MODE == "move":
        shutil.move(str(source_path), str(dest_path))
    else:
        raise ValueError("FILE_OPERATION_MODE 只能是 copy 或 move")


def write_csv(path: Path, rows: list, fieldnames: list):
    """
    写 CSV 文件。
    """
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_class_names(output_dir: Path):
    """
    保存类别顺序。
    这个文件后面训练和推理都要用。
    """
    class_names_path = output_dir / "class_names.json"

    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=4)

    return class_names_path


def write_summary_json(path: Path, summary: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


def write_report_txt(path: Path, summary: dict):
    """
    保存人类可读的 txt 报告。
    """
    lines = []

    lines.append("========== 垃圾分类数据集划分报告 ==========")
    lines.append(f"生成时间：{summary['created_time']}")
    lines.append(f"原始数据目录：{summary['input_dataset_dir']}")
    lines.append(f"输出数据目录：{summary['output_dataset_dir']}")
    lines.append(f"文件操作模式：{summary['file_operation_mode']}")
    lines.append(f"随机种子：{summary['random_seed']}")
    lines.append("")
    lines.append("划分比例：")
    lines.append(f"train：{summary['split_ratios']['train']}")
    lines.append(f"val：{summary['split_ratios']['val']}")
    lines.append(f"test：{summary['split_ratios']['test']}")
    lines.append("")
    lines.append("类别顺序：")
    for index, class_name in enumerate(summary["class_names"]):
        lines.append(f"{index}: {class_name}")

    lines.append("")
    lines.append("每类划分数量：")

    for class_name, item in summary["class_summary"].items():
        lines.append(
            f"{class_name}："
            f"总数 {item['total']}，"
            f"train {item['train']}，"
            f"val {item['val']}，"
            f"test {item['test']}"
        )

    lines.append("")
    lines.append("整体数量：")
    lines.append(f"总有效图片：{summary['total_valid_images']}")
    lines.append(f"train 总数：{summary['split_total']['train']}")
    lines.append(f"val 总数：{summary['split_total']['val']}")
    lines.append(f"test 总数：{summary['split_total']['test']}")
    lines.append(f"无效图片数量：{summary['invalid_image_count']}")

    lines.append("")
    lines.append("日志文件：")
    for name, log_path in summary["log_files"].items():
        lines.append(f"{name}：{log_path}")

    lines.append("")
    lines.append("说明：")
    lines.append("1. split_files.csv 记录每张有效图片被分到哪里。")
    lines.append("2. invalid_images.csv 记录损坏图片或无法读取的图片。")
    lines.append("3. split_summary.csv 记录每个类别的 train / val / test 数量。")
    lines.append("4. class_names.json 保存类别顺序，训练和推理时必须保持一致。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# 主程序
# =========================================================

def main():
    print("========== 垃圾分类数据集划分程序 ==========")

    input_dir = Path(INPUT_DATASET_DIR)

    if not input_dir.exists():
        print(f"错误：原始数据目录不存在：{input_dir.resolve()}")
        print("请检查 INPUT_DATASET_DIR 是否设置正确。")
        return

    output_dir = prepare_output_dir(Path(OUTPUT_DATASET_DIR))
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    create_split_folders(output_dir)

    random.seed(RANDOM_SEED)

    print(f"\n原始数据目录：{input_dir.resolve()}")
    print(f"输出数据目录：{output_dir.resolve()}")
    print(f"划分比例：train={SPLIT_RATIOS['train']}，val={SPLIT_RATIOS['val']}，test={SPLIT_RATIOS['test']}")
    print(f"操作模式：{FILE_OPERATION_MODE}")

    valid_images_by_class, invalid_records = collect_images(input_dir)

    file_log_rows = []
    summary_rows = []
    class_summary = {}

    split_total = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    total_valid_images = 0

    print("\n开始划分并复制图片...")

    for class_name in CLASS_NAMES:
        image_records = valid_images_by_class[class_name]
        total_count = len(image_records)
        total_valid_images += total_count

        split_result = split_class_images(image_records)

        train_count = len(split_result["train"])
        val_count = len(split_result["val"])
        test_count = len(split_result["test"])

        class_summary[class_name] = {
            "total": total_count,
            "train": train_count,
            "val": val_count,
            "test": test_count
        }

        summary_rows.append({
            "class_name": class_name,
            "total": total_count,
            "train": train_count,
            "val": val_count,
            "test": test_count
        })

        print(
            f"{class_name}："
            f"总数 {total_count}，"
            f"train {train_count}，"
            f"val {val_count}，"
            f"test {test_count}"
        )

        for split_name, records in split_result.items():
            split_total[split_name] += len(records)

            for record in records:
                source_path = record["source_path"]
                dest_class_dir = output_dir / split_name / class_name
                dest_path = get_unique_dest_path(dest_class_dir, source_path.name)

                status = "success"
                reason = ""

                try:
                    copy_or_move_file(source_path, dest_path)

                except Exception as e:
                    status = "failed"
                    reason = str(e)

                file_log_rows.append({
                    "time": now_str(),
                    "class_name": class_name,
                    "split": split_name,
                    "source_path": str(source_path),
                    "dest_path": str(dest_path),
                    "source_file_name": source_path.name,
                    "dest_file_name": dest_path.name,
                    "width": record["width"],
                    "height": record["height"],
                    "mode": record["mode"],
                    "file_size_bytes": record["file_size_bytes"],
                    "sha1": record["sha1"],
                    "status": status,
                    "reason": reason
                })

    class_names_path = save_class_names(output_dir)

    split_files_csv = logs_dir / "split_files.csv"
    split_summary_csv = logs_dir / "split_summary.csv"
    invalid_images_csv = logs_dir / "invalid_images.csv"
    split_summary_json = logs_dir / "split_summary.json"
    split_report_txt = logs_dir / "split_report.txt"

    write_csv(
        split_files_csv,
        file_log_rows,
        [
            "time",
            "class_name",
            "split",
            "source_path",
            "dest_path",
            "source_file_name",
            "dest_file_name",
            "width",
            "height",
            "mode",
            "file_size_bytes",
            "sha1",
            "status",
            "reason"
        ]
    )

    write_csv(
        split_summary_csv,
        summary_rows,
        [
            "class_name",
            "total",
            "train",
            "val",
            "test"
        ]
    )

    write_csv(
        invalid_images_csv,
        invalid_records,
        [
            "time",
            "class_name",
            "file_path",
            "file_name",
            "reason"
        ]
    )

    summary = {
        "created_time": now_str(),
        "input_dataset_dir": str(input_dir.resolve()),
        "output_dataset_dir": str(output_dir.resolve()),
        "file_operation_mode": FILE_OPERATION_MODE,
        "random_seed": RANDOM_SEED,
        "split_ratios": SPLIT_RATIOS,
        "class_names": CLASS_NAMES,
        "class_names_json": str(class_names_path.resolve()),
        "class_summary": class_summary,
        "split_total": split_total,
        "total_valid_images": total_valid_images,
        "invalid_image_count": len(invalid_records),
        "log_files": {
            "split_files_csv": str(split_files_csv.resolve()),
            "split_summary_csv": str(split_summary_csv.resolve()),
            "invalid_images_csv": str(invalid_images_csv.resolve()),
            "split_summary_json": str(split_summary_json.resolve()),
            "split_report_txt": str(split_report_txt.resolve())
        }
    }

    write_summary_json(split_summary_json, summary)
    write_report_txt(split_report_txt, summary)

    print("\n========== 数据集划分完成 ==========")
    print(f"输出数据集目录：{output_dir.resolve()}")
    print(f"class_names.json：{class_names_path.resolve()}")

    print("\n整体数量：")
    print(f"有效图片总数：{total_valid_images}")
    print(f"train：{split_total['train']}")
    print(f"val：{split_total['val']}")
    print(f"test：{split_total['test']}")
    print(f"无效图片：{len(invalid_records)}")

    print("\n日志文件：")
    print(f"每张图片去向日志：{split_files_csv.resolve()}")
    print(f"划分数量汇总 CSV：{split_summary_csv.resolve()}")
    print(f"划分数量汇总 JSON：{split_summary_json.resolve()}")
    print(f"文字报告：{split_report_txt.resolve()}")
    print(f"无效图片日志：{invalid_images_csv.resolve()}")

if __name__ == "__main__":
    main()