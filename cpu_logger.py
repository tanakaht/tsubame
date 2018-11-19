#!/usr/bin/env python
# -*- coding: utf-8 -*-
import psutil
import time
import sys

# 標準入力からのファイル名の受け取り
argvs = sys.argv
argc = len(argvs)
if argc == 2:
    saveFileName = argvs[1]
else:
    saveFileName = './NoFileName.md'

intervalSec = 10

# ヘッダ部分の作成
# cpu memoryの使用率　pname pid time
with open(saveFileName, 'a') as f:
    f.write('|time|pname|pid|cpu%|memory%|\n|:--|:--|:--|:--|:--|\n')

# CPUメモリの情報記録
while True:
    process = filter(lambda p: p.name().startswith("python"), psutil.process_iter())

    with open(saveFileName, 'a') as f:

        # 各pythonプロセスごとの情報取得 書き出し
        for i in process:
            retval = [str(i) for i in [time.time(), i.name, i.pid, i.cpu_percent(interval=0), i.memory_percent()]]
            f.write('|' + '|'.join(retval) + '|\n')

    # 休止
    time.sleep(intervalSec)
