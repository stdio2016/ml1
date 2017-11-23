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

## 參考資料

`train.csv` 資料集來自 <https://archive.ics.uci.edu/ml/datasets/Ecoli>，助教取出其中 300 筆，並且加上 2 個無用欄位作為混淆

