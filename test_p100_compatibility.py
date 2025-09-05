"""
Tesla P100 Compatibility Check

檢查要點：
- CUDA capability 應為 (6, 0)
- 禁用 torch.compile / torch._dynamo（環境變數）
- 可見 GPU 數量
"""

import os
import torch


def print_sep(title: str) -> None:
    print("=== " + title)


def main() -> None:
    print_sep("P100 Compatibility Check")

    # CUDA與設備
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")

    if has_cuda:
        num = torch.cuda.device_count()
        print(f"CUDA device count: {num}")
        for i in range(num):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            print(f"GPU[{i}] name: {name}")
            print(f"GPU[{i}] capability: {cap}")
    else:
        print("No CUDA devices detected.")

    # torch.compile / dynamo 設定
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "<unset>")
    dynamo_disable = os.environ.get("TORCHDYNAMO_DISABLE", "<unset>")
    print(f"TORCH_COMPILE_BACKEND: {backend}")
    print(f"TORCHDYNAMO_DISABLE: {dynamo_disable}")

    # 建議
    print_sep("Recommendations")
    print("- Ensure TORCH_COMPILE_BACKEND=eager for P100.")
    print("- Ensure TORCHDYNAMO_DISABLE=1 to fully bypass torch.compile.")
    print("- Use eager mode; avoid Triton kernels on CC 6.0.")


if __name__ == "__main__":
    main()

