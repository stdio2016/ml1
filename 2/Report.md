# 報告

這是個用 k-NN classifier 做機器學習的程式，我用 Kd-Tree 來實做

## 輸入格式

在命令列輸入 `./run.sh <訓練資料檔> <測試資料檔>` 可以用 k-NN 演算法分析資料

訓練資料是 csv 檔，有 12 個欄位，其中第 1 個欄位是編號，第 2 個欄位是 ecoli 的名字，最後一個欄位是 class of ecoli，有 8 個 class：cp、im、pp、imU、om、omL、inL、imS，作為 classifier 的輸出。其他欄位是 attribute ，作為 classifier 的輸入

訓練資料共 300 筆紀錄，而檔案第 1 行是欄位名稱，不是資料

測試資料和訓練資料的格式相同，不過有 36 筆紀錄

### csv 欄位說明
1. 編號
2. ecoli 的名字
3. mcg
4. gvh
5. lip
6. chg
7. aac
8. alm1
9. alm2
10. 助教製造的混淆數據
11. 助教製造的混淆數據
12. class of ecoli

我不是很懂這些專有名詞，所以也不敢解釋這些欄位的意思，如果有興趣，可以到 <https://archive.ics.uci.edu/ml/datasets/Ecoli> 查詢

## 執行環境
* Ubuntu 17.10
* Python 3.6.3

## 使用函式庫和語言

我用 Python 3 來寫，用了 numpy 套件，還有內建套件：csv、sys、collections、math、heapq

## 程式如何運行
主程式是 `main.py`，主程式會讀取 csv 檔案，建立 KD tree，然後用這個 k-D tree 進行 KNN classification。

KNN classification 會從訓練資料中，找 k 筆距離測試資料最近的點，這些點各自有一個分類，輸出就是用這些點進行投票，有最多票的分類就是 classification 的輸出。如果有平手的情況，則在平手的點裡面選一個距離測試資料最近的點，它的分類當作輸出

我把 KD tree 和 K-NN 的實作放在 `kdtree.py` 裡面。KD tree 的建立方法是，把資料按照第一個維度的值排列，取出中位數作為中間節點，並把資料分成兩半。這兩半繼續遞迴製作 KD tree，不過每往下一層，排列的維度就要換成下一個，讓每一層都用不同維度排列，直到維度用完就回到第一個維度，或者沒有資料，就變成空節點。

取得 K-NN 的方法是
1. 設 bests 是目前找到的點， bests = 空集合
2. 把 KD-tree 假裝成二分樹，在樹上遞迴找目標點
3. 到達空節點後往上走，並做以下步驟：
  1. 如果 bests 的大小比 k 小，則把目前所在的節點加到 bests，否則，設 P = bests 中距離目標最遠的點，如果目前的節點到目標點比 P 到目標更近，則把目前所在的節點加到 bests，並從 bests 移除 P
  2. 接下來檢查有沒有可能從節點的另一邊找到 K-NN。如果 bests 的大小比 k 小，則另一邊有可能包含 K-NN，否則計算 mindist = 目前節點的切割面到目標點的距離，設 P = bests 中距離目標最遠的點，如果 P 到目標的距離 > mindist，則另一邊有可能包含 K-NN，否則不可能
  3. 如果可能從節點的另一邊找到 K-NN的話，則對節點的另一邊的節點遞迴做第 2~3 步

實作使用 heapq 內建套件，提供 heap，可以用來取得 bests 中距離目標最遠的點。因為 heapq 裡的 heap 只能取得 heap 裡最小的值，所以在實作中，我把距離取負號，取負號後最小的值就是取負號前最大的值

PCA 的計算也放在 `kdtree.py` 裡。我有做 normalize ，把資料的平均值變成 0，標準差變成 1，因為我覺得這樣比較準

## 要用什麼 k 值

## 參考資料

`train.csv` 資料集來自 <https://archive.ics.uci.edu/ml/datasets/Ecoli>，助教取出其中 300 筆，並且加上 2 個無用欄位作為混淆

