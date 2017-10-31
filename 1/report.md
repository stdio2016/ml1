# 報告

輸入 `./run.sh` 可以使用 ID3 演算法
還沒有 random forest

## 運作結果
使用 ID3 演算法測試了 10 次，得到的平均值

* total accuracy: 0.94
* precision of class 0: 1
* recall of class 0: 1
* precision of class 1: 0.908
* recall of class 1: 0.912
* precision of class 2: 0.916
* recall of class 2: 0.909

## 執行環境
* Ubuntu 17.10
* gcc 7.2.0

## 使用函式庫和語言

我用 C 語言來寫，而且是 C99 。 本來打算用 pthread ，後來還是移除掉它了

## 程式怎麼執行

這個程式可分為3個部份：main、id3、validate。其中 main 用來把資料存進陣列，id3 建立決策樹，validate 則會計算成效

在 main 讀完資料後，會把資料隨機排列，然後把資料切成5等份。k-fold 每次驗證取出1份作為 test data，其他4份作為訓練資料，建立決策樹

對連續型特徵使用 ID3 演算法需要排序，然而我不想每分支一次就排序一次，所以在建立決策樹之前，程式就會對特徵做排序，每個特徵有各自的排序。每次分支，就會把排序表分成兩部份，給下一層的樹使用

在計算 information gain 的部份，我原本有發現很神奇的方法可以計算 entropy，可是老師說不要用奇怪的方法計算，所以就只好用課本的方法。還有我注意到，有的時候會有兩筆資料，特徵的值一樣，決策樹就不能在這裡切割

這個程式要求特徵必須是連續的。我想這個程式沒辦法用在別的地方

程式使用了幾種結構：
1. `struct decision_tree`：就是決策樹！
2. `struct id3_state`：用來紀錄製造決策樹時的內部狀態（不要在程式外部使用）
3. `struct id3_performance_t`：儲存決策樹的 performance ，只有在 validate 部份使用到
4. `float Features[FEATURECOUNT][DATASIZE]`：每一筆資料的特徵的值
5. `int Target[DATASIZE]`：每一筆資料的分類
