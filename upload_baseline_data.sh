#!/bin/bash

# StackOverflow 运维文本聚类 - 数据上传脚本
# 使用方法: ./upload_baseline_data.sh

# ==================== 配置参数 ====================
MASTER_NODE="10.176.62.230"
HDFS_DATA_DIR="/user/root/data"
LOCAL_DATA_FILE="./StackOverFlow_Oracle_Database/oracle_database_questions.json"
HDFS_DATA_FILE="${HDFS_DATA_DIR}/oracle_database_questions.json"

# ==================== 打印配置 ====================
echo "============================================================"
echo "StackOverflow 数据上传脚本"
echo "============================================================"
echo "本地数据: ${LOCAL_DATA_FILE}"
echo "HDFS路径: hdfs://${MASTER_NODE}:9000${HDFS_DATA_FILE}"
echo "============================================================"

# ==================== 检查本地文件 ====================
echo ""
echo "[检查] 验证本地数据文件..."

if [ ! -f "${LOCAL_DATA_FILE}" ]; then
    echo "错误: 本地数据文件不存在: ${LOCAL_DATA_FILE}"
    exit 1
fi

FILE_SIZE=$(du -h "${LOCAL_DATA_FILE}" | awk '{print $1}')
echo "本地文件大小: ${FILE_SIZE}"

# ==================== 创建HDFS目录 ====================
echo ""
echo "[创建] 创建HDFS数据目录..."

hdfs dfs -test -d ${HDFS_DATA_DIR}
if [ $? -ne 0 ]; then
    hdfs dfs -mkdir -p ${HDFS_DATA_DIR}
    echo "已创建目录: ${HDFS_DATA_DIR}"
else
    echo "目录已存在: ${HDFS_DATA_DIR}"
fi

# ==================== 检查HDFS上是否已有数据 ====================
echo ""
echo "[检查] 检查HDFS上是否已有数据..."

hdfs dfs -test -e ${HDFS_DATA_FILE}
if [ $? -eq 0 ]; then
    echo "警告: HDFS上已存在该文件"
    echo ""
    read -p "是否覆盖? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消上传"
        exit 0
    fi
    hdfs dfs -rm ${HDFS_DATA_FILE}
    echo "已删除旧文件"
fi

# ==================== 上传数据 ====================
echo ""
echo "[上传] 上传数据到HDFS..."
echo "这可能需要几分钟，请耐心等待..."
echo "============================================================"

hdfs dfs -put "${LOCAL_DATA_FILE}" "${HDFS_DATA_FILE}"

if [ $? -eq 0 ]; then
    echo "============================================================"
    echo "✓ 数据上传成功!"
    echo "============================================================"
else
    echo "============================================================"
    echo "✗ 数据上传失败!"
    echo "============================================================"
    exit 1
fi

# ==================== 验证上传 ====================
echo ""
echo "[验证] 验证上传结果..."

hdfs dfs -test -e ${HDFS_DATA_FILE}
if [ $? -eq 0 ]; then
    echo "✓ 文件存在于HDFS"
    
    HDFS_SIZE=$(hdfs dfs -du -h ${HDFS_DATA_FILE} | awk '{print $1}')
    echo "HDFS文件大小: ${HDFS_SIZE}"
    
    LINE_COUNT=$(hdfs dfs -cat ${HDFS_DATA_FILE} | wc -l)
    echo "文件行数: ${LINE_COUNT}"
    
    echo ""
    echo "文件前3行预览:"
    hdfs dfs -cat ${HDFS_DATA_FILE} | head -3
fi

echo ""
echo "============================================================"
echo "数据上传完成!"
echo "============================================================"
echo "HDFS路径: hdfs://${MASTER_NODE}:9000${HDFS_DATA_FILE}"
echo ""
echo "下一步: 运行基准方案"
echo "  ./run_baseline.sh"
echo "============================================================"
