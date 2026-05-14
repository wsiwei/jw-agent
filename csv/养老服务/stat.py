import csv
from collections import defaultdict

def process_coverage(input_file, output_file):
    # 存储统计数据：每个县（市、区）一个字典
    stats = defaultdict(lambda: {
        'towns': set(),           # 所有镇（街道）集合
        'towns_with_center': set(), # 有综合养老服务中心的镇集合
        'village_count': 0,       # 村（社区）总数
        'village_with_center': 0  # 有居家养老服务中心的村（社区）数
    })

    # 自动检测编码（优先处理UTF-8 BOM）
    encodings = ['utf-8-sig', 'utf-8', 'gbk']
    file_encoding = None
    for enc in encodings:
        try:
            with open(input_file, 'r', encoding=enc) as f:
                f.read(1024)
            file_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if file_encoding is None:
        print("无法识别文件编码，使用默认 utf-8")
        file_encoding = 'utf-8'

    print(f"使用编码: {file_encoding}")

    with open(input_file, 'r', encoding=file_encoding) as infile:
        reader = csv.DictReader(infile)
        # 检查必要的列是否存在
        required_cols = ['所在县（市、区）', '所在镇（街道）', '镇（街道）综合养老服务中心', '所在村（社区）', '村（社区）居家养老服务中心']
        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError(f"文件中缺少必要的列: {col}")

        for row in reader:
            county = row['所在县（市、区）'].strip()
            town = row['所在镇（街道）'].strip()
            town_center = row['镇（街道）综合养老服务中心'].strip()
            village = row['所在村（社区）'].strip()
            village_center = row['村（社区）居家养老服务中心'].strip()

            if not county or not town:
                continue  # 跳过关键字段为空的行

            # 统计镇（街道）
            stats[county]['towns'].add(town)
            # 统计有综合养老服务中心的镇（只要该行有非空值即认为该镇有中心）
            if town_center:
                stats[county]['towns_with_center'].add(town)

            # 统计村（社区），只要村名非空就算一个村
            if village:
                stats[county]['village_count'] += 1
                # 统计有居家养老服务中心的村
                if village_center:
                    stats[county]['village_with_center'] += 1

    # 生成输出结果
    results = []
    for idx, (county, data) in enumerate(stats.items(), start=1):
        town_total = len(data['towns'])
        town_with_center = len(data['towns_with_center'])
        town_coverage = (town_with_center / town_total * 100) if town_total > 0 else 0

        village_total = data['village_count']
        village_with_center = data['village_with_center']
        village_coverage = (village_with_center / village_total * 100) if village_total > 0 else 0

        results.append({
            '序号': idx,
            '所在县（市、区）': county,
            '镇（街道）数量': town_total,
            '镇（街道）综合养老服务中心数量': town_with_center,
            '镇养老服务中心覆盖率': f"{town_coverage:.2f}%",
            '村（社区）数量': village_total,
            '村（社区）综合养老服务中心数量': village_with_center,
            '村养老服务中心覆盖率': f"{village_coverage:.2f}%"
        })

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as outfile:
        fieldnames = ['序号', '所在县（市、区）', '镇（街道）数量', '镇（街道）综合养老服务中心数量',
                      '镇养老服务中心覆盖率', '村（社区）数量', '村（社区）综合养老服务中心数量',
                      '村养老服务中心覆盖率']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"处理完成！结果已保存至: {output_file}")
    # 打印前几行预览
    print("\n预览结果:")
    for r in results[:5]:
        print(r)

if __name__ == '__main__':
    input_path = r"C:\Users\LXY\Desktop\4+3全部数据整合\养老服务\覆盖率.csv"
    output_path = r"C:\Users\LXY\Desktop\4+3全部数据整合\养老服务\覆盖率_统计结果.csv"
    process_coverage(input_path, output_path)