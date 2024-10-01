# 主程式

`myfunc.py` 裡面放的是各種演算法還有產生資料的函式。

`Q9to12.py` 則是使用 `myfunc.py` 裡面的演算法執行出結果後，將結果存在 `exp_result.txt` 裡面。

最後再由 `Q9to12plot.py` 讀取 `exp_result.txt`，將圖畫出來。

# 額外測試

12 題中我特別抓出了兩筆資料，抓取的方法在 `test.ipynb` 裡面的一個 Cell(有文字標記)，並且額外存了這八筆資料的內容到 `x_train_data`,`x_test_data`...等的 txt 檔裡面，然後再用 `test2.ipynb` 做一些運算和畫圖。

`outlier.py` 跟 `outlierplot.py` 也是用來驗證 12 題的想法