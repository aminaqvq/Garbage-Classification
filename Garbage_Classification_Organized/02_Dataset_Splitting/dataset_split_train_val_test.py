import csv
import json
import random
import shutil
import sys
import hashlib
import argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from datetime import datetime

from PIL import Image


# =========================================================
# 五分类配置读取
# =========================================================

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "09_Vision_Trigger_5Class_System"
    / "config"
    / "class_mapping_5class.json"
)


def load_class_mapping(config_path=None):
    """从 class_mapping_5class.json 读取五分类配置，返回 (class_names, class_to_idx, full_config)。"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    class_to_idx = data.get("class_to_idx")
    if not isinstance(class_to_idx, dict):
        raise ValueError(f"缺少有效的 class_to_idx：{config_path}")
    sorted_pairs = sorted(class_to_idx.items(), key=lambda kv: kv[1])
    class_names = [name for name, _ in sorted_pairs]
    expected = {"待分拣", "其他", "厨余", "可回收", "有害"}
    if set(class_names) != expected:
        raise ValueError(f"类别不匹配：需要 {expected}，实际 {class_names}")
    if len(class_names) != 5:
        raise ValueError(f"类别数应为 5，实际 {len(class_names)}")
    return class_names, class_to_idx, data


try:
    CLASS_NAMES, _CLASS_TO_IDX, _FULL_CONFIG = load_class_mapping()
except Exception:
    CLASS_NAMES = None
    _CLASS_TO_IDX = None
    _FULL_CONFIG = None


# =========================================================
# 基础配置区
# =========================================================

INPUT_DATASET_DIR = str(PROJECT_ROOT / "dataset")
OUTPUT_DATASET_DIR = str(PROJECT_ROOT / "garbage_dataset")
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
FILE_OPERATION_MODE = "copy"
ENABLE_FILE_HASH = False


# =========================================================
# 工具函数
# =========================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def calculate_sha1(path: Path) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def check_image_valid(path: Path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
        return True, width, height, mode, ""
    except Exception as e:
        return False, None, None, None, str(e)


def prepare_output_dir(output_dir: Path, clean_output: bool = False) -> Path:
    if clean_output:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"已清空并重建目录：{output_dir.resolve()}")
        return output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    # 输出目录已存在且未指定 --clean-output：自动创建带时间戳的新目录，避免数据覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_output_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
    new_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录已存在，自动创建新目录：{new_output_dir.resolve()}")
    print(f"（如需覆盖原目录，请使用 --clean-output）")
    return new_output_dir


def create_split_folders(output_dir: Path, class_names):
    for split_name in ["train", "val", "test"]:
        for class_name in class_names:
            class_dir = output_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)


def get_unique_dest_path(dest_dir: Path, original_name: str) -> Path:
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


def calculate_split_counts(total_count: int, ratios: dict):
    if total_count <= 0:
        return 0, 0, 0
    if total_count == 1:
        return 1, 0, 0
    if total_count == 2:
        return 1, 1, 0
    train_count = int(total_count * ratios["train"])
    val_count = int(total_count * ratios["val"])
    test_count = total_count - train_count - val_count
    if train_count <= 0:
        train_count = 1
    if val_count <= 0:
        val_count = 1
    if test_count <= 0:
        test_count = 1
    while train_count + val_count + test_count > total_count:
        if train_count > 1:
            train_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            test_count -= 1
    while train_count + val_count + test_count < total_count:
        train_count += 1
    return train_count, val_count, test_count


def collect_images(input_dir: Path, class_names):
    valid_images_by_class = {class_name: [] for class_name in class_names}
    invalid_records = []
    for class_name in class_names:
        class_dir = input_dir / class_name
        if not class_dir.exists():
            invalid_records.append({
                "time": now_str(), "class_name": class_name,
                "file_path": str(class_dir), "file_name": "",
                "reason": "类别文件夹不存在"
            })
            print(f"警告：类别文件夹不存在：{class_dir}")
            continue
        image_paths = [path for path in class_dir.rglob("*") if is_image_file(path)]
        print(f"\n正在检查类别：{class_name}，发现图片文件 {len(image_paths)} 张")
        for image_path in image_paths:
            valid, width, height, mode, reason = check_image_valid(image_path)
            if valid:
                valid_images_by_class[class_name].append({
                    "class_name": class_name, "source_path": image_path,
                    "file_name": image_path.name, "width": width,
                    "height": height, "mode": mode,
                    "file_size_bytes": image_path.stat().st_size,
                    "sha1": calculate_sha1(image_path) if ENABLE_FILE_HASH else ""
                })
            else:
                invalid_records.append({
                    "time": now_str(), "class_name": class_name,
                    "file_path": str(image_path), "file_name": image_path.name,
                    "reason": reason
                })
    return valid_images_by_class, invalid_records


def split_class_images(image_records):
    records = image_records[:]
    random.shuffle(records)
    total_count = len(records)
    train_count, val_count, test_count = calculate_split_counts(total_count, SPLIT_RATIOS)
    return {
        "train": records[:train_count],
        "val": records[train_count:train_count + val_count],
        "test": records[train_count + val_count:]
    }


def copy_or_move_file(source_path: Path, dest_path: Path):
    if FILE_OPERATION_MODE == "copy":
        shutil.copy2(source_path, dest_path)
    elif FILE_OPERATION_MODE == "move":
        shutil.move(str(source_path), str(dest_path))
    else:
        raise ValueError("FILE_OPERATION_MODE 只能是 copy 或 move")


def write_csv(path: Path, rows: list, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_class_names(output_dir: Path, class_names):
    path = output_dir / "class_names.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    return path


def save_class_mapping_json(output_dir: Path, full_config: dict):
    """复制 class_mapping_5class.json 完整内容（含 serial_protocol、notes）到输出目录。"""
    path = output_dir / "class_mapping.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, ensure_ascii=False, indent=2)
    return path


def save_split_summary_json(path: Path, summary: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


def write_report_txt(path: Path, summary: dict):
    lines = [
        "========== 垃圾分类数据集划分报告 ==========",
        f"生成时间：{summary['created_time']}",
        f"原始数据目录：{summary['input_dataset_dir']}",
        f"输出数据目录：{summary['output_dataset_dir']}",
        f"文件操作模式：{summary['file_operation_mode']}",
        f"随机种子：{summary['random_seed']}",
        "",
        "划分比例：",
        f"train：{summary['split_ratios']['train']}",
        f"val：{summary['split_ratios']['val']}",
        f"test：{summary['split_ratios']['test']}",
        "",
        "类别顺序：",
    ]
    for index, class_name in enumerate(summary["class_names"]):
        lines.append(f"{index}: {class_name}")
    lines.append("")
    lines.append("每类划分数量：")
    for class_name, item in summary["class_summary"].items():
        lines.append(f"{class_name}：总数 {item['total']}，train {item['train']}，val {item['val']}，test {item['test']}")
    lines.append("")
    lines.append("整体数量：")
    lines.append(f"总有效图片：{summary['total_valid_images']}")
    lines.append(f"train 总数：{summary['split_total']['train']}")
    lines.append(f"val 总数：{summary['split_total']['val']}")
    lines.append(f"test 总数：{summary['split_total']['test']}")
    lines.append(f"无效图片数量：{summary['invalid_image_count']}")
    if summary.get("warnings"):
        lines.append("")
        lines.append("警告：")
        for w in summary["warnings"]:
            lines.append(f"  - {w}")
    lines.append("")
    lines.append("日志文件：")
    for name, log_path in summary["log_files"].items():
        lines.append(f"{name}：{log_path}")
    lines.append("")
    lines.append("说明：")
    lines.append("1. split_files.csv 记录每张有效图片被分到哪里。")
    lines.append("2. invalid_images.csv 记录损坏图片或无法读取的图片。")
    lines.append("3. split_summary.csv 记录每个类别的 train/val/test 数量。")
    lines.append("4. class_names.json 保存类别顺序，训练和推理时必须保持一致。")
    lines.append("5. class_mapping.json 是完整的五分类映射配置副本（含 serial_protocol、notes）。")
    lines.append("6. split_summary.json 是完整的划分元数据。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# 命令行参数
# =========================================================

def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        print(f"错误：划分比例之和必须接近 1.0，当前 train={train_ratio} + val={val_ratio} + test={test_ratio} = {total:.4f}")
        sys.exit(1)
    if train_ratio <= 0:
        print(f"错误：train 比例必须大于 0，当前为 {train_ratio}")
        sys.exit(1)
    if val_ratio <= 0:
        print(f"错误：val 比例必须大于 0，当前为 {val_ratio}")
        sys.exit(1)
    if test_ratio <= 0:
        print(f"错误：test 比例必须大于 0，当前为 {test_ratio}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="垃圾分类数据集划分程序（五分类视觉触发版）")
    parser.add_argument("--config", default=None, help=f"五分类配置文件路径")
    parser.add_argument("--source-dir", default=INPUT_DATASET_DIR, help="原始数据目录")
    parser.add_argument("--output-dir", default=OUTPUT_DATASET_DIR, help="划分后数据目录")
    parser.add_argument("--train-ratio", type=float, default=SPLIT_RATIOS["train"], help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=SPLIT_RATIOS["val"], help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=SPLIT_RATIOS["test"], help="测试集比例")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--copy", action="store_true", default=True, help="复制文件（默认）")
    mode_group.add_argument("--move", action="store_true", help="移动文件（谨慎）")
    parser.add_argument("--clean-output", action="store_true", help="清空输出目录（不交互）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写文件")
    parser.add_argument("--strict", action="store_true", help="任何类别为空则报错退出")
    args = parser.parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    return args


# =========================================================
# 主程序
# =========================================================

def main():
    if CLASS_NAMES is None:
        print("错误：无法加载五分类配置文件。")
        print(f"请确认文件存在：{DEFAULT_CONFIG_PATH}")
        sys.exit(1)

    class_names = CLASS_NAMES
    full_config = _FULL_CONFIG

    args = parse_args()

    global INPUT_DATASET_DIR, OUTPUT_DATASET_DIR, SPLIT_RATIOS, RANDOM_SEED, FILE_OPERATION_MODE
    INPUT_DATASET_DIR = args.source_dir
    OUTPUT_DATASET_DIR = args.output_dir
    SPLIT_RATIOS = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    RANDOM_SEED = args.seed
    FILE_OPERATION_MODE = "move" if args.move else "copy"

    if args.config:
        try:
            class_names, _, full_config = load_class_mapping(args.config)
        except Exception as e:
            print(f"错误：加载指定配置文件失败：{e}")
            sys.exit(1)

    print("========== 垃圾分类数据集划分程序（五分类视觉触发版）==========")
    print(f"当前五分类类别：{class_names}")

    input_dir = Path(INPUT_DATASET_DIR)
    output_dir = Path(OUTPUT_DATASET_DIR)

    if args.dry_run:
        print(f"\n[DRY-RUN] 不会创建目录、不写文件、不复制/移动文件。")
        print(f"[DRY-RUN] 源目录：{input_dir.resolve()}")
        print(f"[DRY-RUN] 输出目录：{output_dir.resolve()}")
        print(f"[DRY-RUN] 比例：train={SPLIT_RATIOS['train']} val={SPLIT_RATIOS['val']} test={SPLIT_RATIOS['test']}")
        print(f"[DRY-RUN] 模式：{FILE_OPERATION_MODE} 种子：{RANDOM_SEED}")
        if not input_dir.exists():
            print(f"\n[Dry-Run] 警告：源目录不存在：{input_dir.resolve()}")
            print(f"[Dry-Run] 参数解析正常，验证通过。")
            sys.exit(0)
        warnings_list = []
        print()
        for class_name in class_names:
            class_dir = input_dir / class_name
            if not class_dir.exists():
                msg = f"类别 '{class_name}' 目录不存在"
                print(f"[Dry-Run] 警告：{msg}")
                warnings_list.append(msg)
                continue
            count = len([p for p in class_dir.rglob("*") if is_image_file(p)])
            tc, vc, tec = calculate_split_counts(count, SPLIT_RATIOS)
            print(f"[Dry-Run] {class_name}：{count} 张 → train={tc} val={vc} test={tec}")
            if count == 0:
                warnings_list.append(f"类别 '{class_name}' 图片数为 0")
            elif count < 10:
                warnings_list.append(f"类别 '{class_name}' 仅 {count} 张，建议 >=300")
        print(f"\n[Dry-Run] 输出将包含：class_names.json / class_mapping.json / split_summary.json")
        print(f"[Dry-Run] 警告数：{len(warnings_list)}")
        print(f"[Dry-Run] 验证通过。")
        sys.exit(0)

    if not input_dir.exists():
        print(f"错误：原始数据目录不存在：{input_dir.resolve()}")
        sys.exit(1)

    output_dir = prepare_output_dir(output_dir, clean_output=args.clean_output)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    create_split_folders(output_dir, class_names)
    random.seed(RANDOM_SEED)

    print(f"\n原始：{input_dir.resolve()}  →  输出：{output_dir.resolve()}")
    print(f"比例：train={SPLIT_RATIOS['train']} val={SPLIT_RATIOS['val']} test={SPLIT_RATIOS['test']}  模式：{FILE_OPERATION_MODE}")

    valid_images_by_class, invalid_records = collect_images(input_dir, class_names)

    strict_error = False
    for class_name in class_names:
        if len(valid_images_by_class[class_name]) == 0:
            if args.strict:
                print(f"错误（strict）：类别 '{class_name}' 图片数为 0。")
                strict_error = True
            else:
                print(f"警告：类别 '{class_name}' 图片数为 0。")
    if strict_error:
        sys.exit(1)

    file_log_rows, summary_rows, class_summary, warnings_list = [], [], {}, []
    split_total = {"train": 0, "val": 0, "test": 0}
    total_valid_images = 0

    print("\n开始划分...")
    for class_name in class_names:
        image_records = valid_images_by_class[class_name]
        total_count = len(image_records)
        total_valid_images += total_count
        if total_count == 0:
            warnings_list.append(f"类别 '{class_name}' 图片数为 0")
        if 0 < total_count < 10:
            warnings_list.append(f"类别 '{class_name}' 仅 {total_count} 张（建议 >=300）")
        split_result = split_class_images(image_records)
        train_count, val_count, test_count = len(split_result["train"]), len(split_result["val"]), len(split_result["test"])
        class_summary[class_name] = {"total": total_count, "train": train_count, "val": val_count, "test": test_count}
        summary_rows.append({"class_name": class_name, "total": total_count, "train": train_count, "val": val_count, "test": test_count})
        print(f"{class_name}：总数 {total_count}，train {train_count}，val {val_count}，test {test_count}")
        for split_name, records in split_result.items():
            split_total[split_name] += len(records)
            for record in records:
                source_path = record["source_path"]
                dest_path = get_unique_dest_path(output_dir / split_name / class_name, source_path.name)
                try:
                    copy_or_move_file(source_path, dest_path)
                    status, reason = "success", ""
                except Exception as e:
                    status, reason = "failed", str(e)
                file_log_rows.append({
                    "time": now_str(), "class_name": class_name, "split": split_name,
                    "source_path": str(source_path), "dest_path": str(dest_path),
                    "source_file_name": source_path.name, "dest_file_name": dest_path.name,
                    "width": record["width"], "height": record["height"], "mode": record["mode"],
                    "file_size_bytes": record["file_size_bytes"], "sha1": record["sha1"],
                    "status": status, "reason": reason
                })

    class_names_path = save_class_names(output_dir, class_names)
    class_mapping_path = save_class_mapping_json(output_dir, full_config)

    split_files_csv = logs_dir / "split_files.csv"
    split_summary_csv = logs_dir / "split_summary.csv"
    invalid_images_csv = logs_dir / "invalid_images.csv"
    split_summary_json = logs_dir / "split_summary.json"
    split_report_txt = logs_dir / "split_report.txt"

    write_csv(split_files_csv, file_log_rows,
              ["time", "class_name", "split", "source_path", "dest_path", "source_file_name",
               "dest_file_name", "width", "height", "mode", "file_size_bytes", "sha1", "status", "reason"])
    write_csv(split_summary_csv, summary_rows, ["class_name", "total", "train", "val", "test"])
    write_csv(invalid_images_csv, invalid_records, ["time", "class_name", "file_path", "file_name", "reason"])

    summary = {
        "source_dir": str(input_dir.resolve()), "output_dir": str(output_dir.resolve()),
        "train_ratio": SPLIT_RATIOS["train"], "val_ratio": SPLIT_RATIOS["val"], "test_ratio": SPLIT_RATIOS["test"],
        "seed": RANDOM_SEED,
        "class_counts": {n: class_summary[n]["total"] for n in class_names},
        "split_counts": {n: {"train": class_summary[n]["train"], "val": class_summary[n]["val"], "test": class_summary[n]["test"]} for n in class_names},
        "total_images": total_valid_images, "generated_at": now_str(), "warnings": warnings_list,
        "created_time": now_str(), "input_dataset_dir": str(input_dir.resolve()),
        "output_dataset_dir": str(output_dir.resolve()), "file_operation_mode": FILE_OPERATION_MODE,
        "random_seed": RANDOM_SEED, "split_ratios": SPLIT_RATIOS,
        "class_names": class_names, "class_names_json": str(class_names_path.resolve()),
        "class_mapping_json": str(class_mapping_path.resolve()),
        "class_summary": class_summary, "split_total": split_total,
        "total_valid_images": total_valid_images, "invalid_image_count": len(invalid_records),
        "log_files": {
            "split_files_csv": str(split_files_csv.resolve()),
            "split_summary_csv": str(split_summary_csv.resolve()),
            "invalid_images_csv": str(invalid_images_csv.resolve()),
            "split_summary_json": str(split_summary_json.resolve()),
            "split_report_txt": str(split_report_txt.resolve()),
            "class_mapping_json": str(class_mapping_path.resolve()),
        }
    }

    save_split_summary_json(split_summary_json, summary)
    write_report_txt(split_report_txt, summary)

    print(f"\n========== 划分完成 ==========")
    print(f"class_names.json：{class_names_path.resolve()}")
    print(f"class_mapping.json：{class_mapping_path.resolve()}")
    print(f"split_summary.json：{split_summary_json.resolve()}")
    print(f"总有效图片：{total_valid_images}  train={split_total['train']} val={split_total['val']} test={split_total['test']}")
    if warnings_list:
        print(f"\n警告 ({len(warnings_list)} 条)：")
        for w in warnings_list:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
