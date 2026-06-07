#!/bin/bash
# upload_data.sh - Upload StackOverflow data to HDFS

HDFS_INPUT="/stackoverflow_oracle/input"
LOCAL_JSON="./StackOverFlow_Oracle_Database/oracle_database_questions.json"

echo "============================================"
echo "Uploading StackOverflow data to HDFS"
echo "============================================"

echo "[1] Checking HDFS status..."
hdfs dfsadmin -report | head -5

echo ""
echo "[2] Creating HDFS directory..."
hdfs dfs -mkdir -p ${HDFS_INPUT}

echo ""
echo "[3] Uploading JSON data..."
hdfs dfs -put -f ${LOCAL_JSON} ${HDFS_INPUT}/

echo ""
echo "[4] Verifying upload..."
hdfs dfs -ls ${HDFS_INPUT}/
hdfs dfs -cat ${HDFS_INPUT}/oracle_database_questions.json | wc -l

echo ""
echo "Upload complete! Data is at: ${HDFS_INPUT}/oracle_database_questions.json"
