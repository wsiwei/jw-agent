FROM python:3.11-slim

WORKDIR /app

# 换阿里云 apt 镜像源（python:3.11-slim 基于 Debian Bookworm）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# 复制达梦客户端动态库（仅 .so 文件）
RUN mkdir -p /opt/dmdbms/bin
COPY dm_so/*.so /opt/dmdbms/bin/

# 设置动态库搜索路径
ENV LD_LIBRARY_PATH=/opt/dmdbms/bin:$LD_LIBRARY_PATH

# 先装依赖（利用 Docker 层缓存，代码变动时不重新装包）
# 换阿里云 pip 镜像源
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    -r requirements.txt

# 复制应用代码
COPY app/        ./app/
COPY prompt/     ./prompt/
COPY csv/        ./csv/
COPY main.py     .

# 复制数据库 schema 文件
COPY db_table_list.txt    ./db_table_list.txt
COPY db_columns.json      ./db_columns.json
COPY db_schema_compact.txt ./db_schema_compact.txt

# 创建上传目录和 config 目录（config.json 运行时挂载）
RUN mkdir -p upload config

EXPOSE 9527

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9527"]
