#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 编码检测脚本（支持多种常见编码）
用法：python csv_encoding_checker.py [目标目录]
"""

import os
import sys

# 常见编码探测顺序（优先级从高到低）
ENCODINGS = [
    'utf-8-sig',   # 带 BOM 的 UTF-8
    'utf-16',      # 带 BOM 的 UTF-16 (LE/BE 自动识别)
    'gb18030',     # 中文最全
    'gbk',
    'big5',        # 繁体中文
    'latin-1',     # 西欧，基本不会解码失败（兜底）
]

def detect_encoding(file_path, sample_size=1024*1024):
    """检测文件编码，返回编码名或 'unknown'"""
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
        if not raw:
            return 'empty'

        for enc in ENCODINGS:
            try:
                # 对于 utf-16，Python 会自动处理 BOM
                raw.decode(enc)
                # 特殊标记去掉 BOM 的 UTF-8
                if enc == 'utf-8-sig' and raw.startswith(b'\xef\xbb\xbf'):
                    return 'utf-8'
                return enc
            except UnicodeDecodeError:
                continue
        return 'unknown'

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    if not os.path.isdir(target_dir):
        print(f"错误：'{target_dir}' 不是有效目录。")
        sys.exit(1)

    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                encoding = detect_encoding(full_path)
                print(f"{full_path} : {encoding}")

if __name__ == '__main__':
    main()