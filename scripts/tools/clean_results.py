from pathlib import Path
import argparse


def is_safe_path(base_dir: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def collect_files(out_dir: Path, patterns: list[str]) -> list[Path]:
    files = []
    for pattern in patterns:
        files.extend(out_dir.rglob(pattern))
    return sorted(set(files))


def clean_outputs(base_dir: Path, subdir: str, prefix: str, dry_run: bool = False) -> None:
    out_dir = base_dir / "outputs" / subdir
    patterns = [f"{prefix}_*.csv", f"{prefix}_*.txt", f"{prefix}_*.png"]

    if not out_dir.exists():
        print(f"[SKIP] 目录不存在: {out_dir}")
        return

    files = collect_files(out_dir, patterns)
    if not files:
        print(f"[SKIP] 没有匹配文件: {out_dir}")
        return

    print(f"[INFO] {subdir} 匹配到 {len(files)} 个文件")
    for file_path in files:
        print(f" - {file_path}")

    if dry_run:
        print("[DRY-RUN] 仅预览，不执行删除")
        return

    deleted = 0
    for file_path in files:
        if is_safe_path(base_dir, file_path):
            file_path.unlink(missing_ok=True)
            deleted += 1
        else:
            print(f"[WARN] 跳过不安全路径: {file_path}")

    print(f"[DONE] {subdir} 已删除 {deleted} 个文件")


def main() -> None:
    parser = argparse.ArgumentParser(description="清理建模结果文件")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将删除的文件，不真正删除",
    )
    parser.add_argument(
        "--target",
        choices=["q1", "q2", "q3", "all"],
        default="all",
        help="选择清理范围: q1/q2/q3/all(默认)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    if args.target in ("q1", "all"):
        clean_outputs(base_dir=base_dir, subdir="q1", prefix="q1", dry_run=args.dry_run)
    if args.target in ("q2", "all"):
        clean_outputs(base_dir=base_dir, subdir="q2", prefix="q2", dry_run=args.dry_run)
    if args.target in ("q3", "all"):
        clean_outputs(base_dir=base_dir, subdir="q3", prefix="q3", dry_run=args.dry_run)


if __name__ == "__main__":
    main()
