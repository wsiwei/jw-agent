"""
从达梦数据库查询字段注释，补充 db_columns.json 中注释为空的字段。
运行：conda activate ReActAgentsTest && python scripts/fill_col_comments.py
"""
import json
import re
import sys
from pathlib import Path

import dmPython

# ── 配置 ──────────────────────────────────────────────────────────────
DB_HOST     = "10.249.84.22"
DB_PORT     = 5236
DB_USER     = "SYSDBA"
DB_PASSWORD = "SYSDBA001"

BASE        = Path(__file__).parent.parent
COL_MAP_PATH = BASE / "db_columns.json"
# ──────────────────────────────────────────────────────────────────────


def connect():
    return dmPython.connect(
        user=DB_USER, password=DB_PASSWORD,
        server=DB_HOST, port=DB_PORT
    )


def fetch_col_comments(conn, owner: str, table: str) -> dict[str, str]:
    """查 DBA_COL_COMMENTS，返回 {列名: 注释}"""
    sql = (
        "SELECT COLUMN_NAME, COMMENTS "
        "FROM DBA_COL_COMMENTS "
        "WHERE OWNER = ? AND TABLE_NAME = ?"
    )
    cur = conn.cursor()
    cur.execute(sql, (owner.upper(), table.upper()))
    rows = cur.fetchall()
    cur.close()
    return {row[0].upper(): (row[1] or "").strip() for row in rows}


def parse_col_line(line: str):
    """
    解析一行字段描述，如 '  DEPT VARCHAR -- 所属部门'
    返回 (col_name, type_part, comment)
    """
    m = re.match(r'\s*(\w+)\s+(.*?)\s*--\s*(.*)', line)
    if m:
        return m.group(1).upper(), m.group(2).strip(), m.group(3).strip()
    return None, None, None


def has_empty_comments(cols: list[str]) -> bool:
    """判断该表是否有任何字段注释为空"""
    for line in cols:
        col_name, _, comment = parse_col_line(line)
        if col_name and not comment:
            return True
    return False


def fill_comments(cols: list[str], db_comments: dict[str, str]) -> tuple[list[str], int]:
    """用数据库注释填充空注释，返回 (新列表, 填充数量)"""
    filled = 0
    result = []
    for line in cols:
        col_name, type_part, comment = parse_col_line(line)
        if col_name and not comment:
            db_cmt = db_comments.get(col_name, "")
            if db_cmt:
                result.append(f"  {col_name} {type_part} -- {db_cmt}")
                filled += 1
                continue
        result.append(line)
    return result, filled


def main():
    print("加载 db_columns.json ...")
    col_map: dict = json.loads(COL_MAP_PATH.read_text(encoding="utf-8"))

    # 找出有空注释的表
    empty_tables = [
        tbl for tbl, cols in col_map.items()
        if has_empty_comments(cols)
    ]
    print(f"发现 {len(empty_tables)} 张表有空注释字段，开始查询数据库...\n")

    if not empty_tables:
        print("无需更新。")
        return

    conn = connect()
    total_filled = 0
    updated_tables = 0

    for tbl in empty_tables:
        parts = tbl.split(".", 1)
        if len(parts) != 2:
            print(f"  [跳过] 无法解析 schema.table: {tbl}")
            continue
        owner, table = parts

        db_comments = fetch_col_comments(conn, owner, table)
        new_cols, filled = fill_comments(col_map[tbl], db_comments)

        if filled > 0:
            col_map[tbl] = new_cols
            total_filled += filled
            updated_tables += 1
            print(f"  [OK] {tbl}: 补充 {filled} 个注释")
        else:
            print(f"  [--] {tbl}: 数据库也无注释，跳过")

    conn.close()

    if total_filled > 0:
        COL_MAP_PATH.write_text(
            json.dumps(col_map, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\n完成：更新 {updated_tables} 张表，共补充 {total_filled} 个字段注释。")
        print(f"已写入 {COL_MAP_PATH}")
    else:
        print("\n数据库中也没有注释，db_columns.json 未修改。")


if __name__ == "__main__":
    main()
