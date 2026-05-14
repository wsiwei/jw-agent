#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 批量转 UTF-8 编码脚本（无乱码）
用法：python csv_to_utf8.py [目标目录] [--backup]
参数：
  target_dir : 要扫描的根目录（默认当前目录）
  --backup   : 转换前在原文件同目录下创建 .bak 备份
"""

import os
import sys
import chardet

# 是否添加 UTF-8 BOM（建议 False，兼容性更好）
ADD_BOM = False

def detect_encoding(file_path, sample_size=1024*1024):
    """使用 chardet 检测编码，返回编码名（小写）"""
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
        if not raw:
            return 'empty'
        result = chardet.detect(raw)
        encoding = result.get('encoding')
        confidence = result.get('confidence', 0)
        # 置信度过低时回退到常见编码循环检测（可选）
        if confidence < 0.5 or encoding is None:
            # 回退尝试常见编码列表
            for enc in ['utf-8', 'gb18030', 'gbk', 'big5', 'utf-16']:
                try:
                    raw.decode(enc)
                    return enc
                except:
                    continue
            return 'unknown'
        return encoding.lower()

def convert_to_utf8(file_path, backup=False):
    """将文件转换为 UTF-8 编码（原地替换），返回 True 表示已转换"""
    # 1. 检测源编码
    src_enc = detect_encoding(file_path)
    if src_enc in ['empty', 'unknown']:
        print(f"[跳过] {file_path} : 无法检测编码 ({src_enc})")
        return False

    # 2. 如果是 UTF-8 且没有 BOM 要求，跳过
    if src_enc in ['utf-8', 'utf-8-sig']:
        # 如果已经是 UTF-8 但可能带 BOM，可以去掉 BOM（可选）
        if ADD_BOM:
            # 需要确保是带 BOM 的 UTF-8，这里简单处理：读取然后重写带 BOM
            with open(file_path, 'rb') as f:
                raw = f.read()
            if raw.startswith(b'\xef\xbb\xbf'):
                # 已经有 BOM，跳过
                print(f"[跳过] {file_path} : 已是 UTF-8 带 BOM")
                return False
            else:
                # 写入带 BOM 的 UTF-8
                with open(file_path, 'rb') as f:
                    content = f.read()
                with open(file_path, 'wb') as f:
                    f.write(b'\xef\xbb\xbf' + content)
                print(f"[转换] {file_path} : 添加 UTF-8 BOM")
                return True
        else:
            # 已经是 UTF-8 且不需要 BOM，跳过
            print(f"[跳过] {file_path} : 已是 UTF-8")
            return False

    # 3. 备份（如果需要）
    if backup:
        bak_path = file_path + '.bak'
        if not os.path.exists(bak_path):
            import shutil
            shutil.copy2(file_path, bak_path)
            print(f"[备份] {bak_path}")

    # 4. 读取原文（按检测到的编码）
    try:
        with open(file_path, 'r', encoding=src_enc, errors='strict') as f:
            content = f.read()
    except Exception as e:
        print(f"[错误] 读取 {file_path} 失败（编码 {src_enc}）：{e}")
        return False

    # 5. 写入 UTF-8
    mode = 'wb' if ADD_BOM else 'w'
    try:
        if ADD_BOM:
            with open(file_path, 'wb') as f:
                f.write(content.encode('utf-8'))
        else:
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(content)
        print(f"[转换] {file_path} : {src_enc} -> UTF-8")
        return True
    except Exception as e:
        print(f"[错误] 写入 {file_path} 失败：{e}")
        return False

def main():
    # 解析命令行参数
    args = sys.argv[1:]
    target_dir = '.'
    backup = False
    for arg in args:
        if arg == '--backup':
            backup = True
        elif not arg.startswith('-'):
            target_dir = arg

    if not os.path.isdir(target_dir):
        print(f"错误：'{target_dir}' 不是有效目录。")
        sys.exit(1)

    print(f"扫描目录：{target_dir}")
    print(f"备份模式：{'开启' if backup else '关闭'}")
    print("开始处理...\n")

    converted = 0
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                if convert_to_utf8(full_path, backup):
                    converted += 1

    print(f"\n完成！共转换 {converted} 个文件。")

if __name__ == '__main__':
    main()