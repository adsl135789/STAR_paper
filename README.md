我現在在寫一個ACM short paper，頁數含reference在四頁以內，我發現QGpT這篇論文的兩個缺點，我針對這兩點去改進，成功改進其效果
請你跟我一起討論我的論文內容，並作為reviewer提出質疑與抨擊，一起完成完整的論文
並請先參考 @STAR_ACM_PAPER/star_ZH.tex中的初稿，先幫我寫成中文


# QGpT缺點及我改進的方法
1. QGpT在取partial table時太過heuristic，直接擷取前k個instances與table header作為partial table
- 可能問題：可能會擷取到類似或是不重要的instance
- 解決想法： 使用Kmeans clustering來對table內部分群，將table拆分成k群，並取k個centroid row作為partial table，這樣能夠選出k個row足夠代表這個table的row
    - Header-aware kmeans clustering作法：使用embedding model將header與每一個row各自embedding，得$E_{header}$與$E_{row}$，將其加權$\alpha*E_{header}+(1-\alpha)*E_{row}$後作Kmeans clustering
    - Synthetic Query Generation: 原本的作法是直接將partial table丟給LLM生成能用這個table生成的問題，這些問題要作為user query與table之間的橋樑，我的方法會改成輸入為kmeans分群後的k個clustered tables，每一個clustered table生成一個問題，最終生成k個問題，進而確保synthetic queries能夠很好的涵蓋我的Partial Table，能夠真正與user query建立語意橋樑，我稱為Clustering-guided query Generation (CGQG)

2.  QGpT在處理partial table + synthetic queries時太粗糙，是直接將partial table與synthetic queries串接後送進encoder，作為最終的table representation
- 可能問題： 缺乏更細緻的語義整合
- 解決想法：
    - Weighted Fusion: 將partial table與將synthetic queries concatenate後送進embedding model，得到$e_{t}$與$e_{q}$，並將這2個embedding加權組合成最終的table representation
    - $E = W_t*E_{t}+W_q*E_{q}$
    - 方法分成2種
        - 1. Fixed Weight Fusion (FWF): 給予table固定權重 $\alpha$, 剩下權重給$e_{q}$
            - 請參考 @dense_embedding.py 中 **concat** mode的內容
        - 2. Dynamic Weight Fusion (DWF): 
            - 計算table與query之間的相似度，利用softmax注意力機制，計算出權重
            - 請參考 @dense_embedding.py 中 **dynamic_concat** mode的內容
            - $W_{q}=\beta*cos(E_t,E_q)$, ($\alpha$應該是用來控制query embedding的權重範圍，請幫我合理補充)



# 論文章節 
1. Introduction
2. Related Work (請幫我找出適合的related work並於reference中附上)
3. Methodology
    會有一個流程圖，請幫我保留空間
    3.1 Sementic Clustering and Query Generation (SCQG)
    3.2 Weight Fusion (WF)
        3.2.1 Fixed Weight Fusion
        3.2.2 Dynamic Weight Fusion
4. Experiments
    4.1 Experimental Settings
        4.1.1 Datasets
        4.1.2 Baselines
        4.1.3 Implementation (metric, hyperparamenter, model)
5. Conclusion and Future work
6. Reference


# experiments
- 依照常規論文數據的標記方式，實驗數據中的top1用粗體，top2用底線
## 實驗setup
- 參照QGpT設定
    - baseline: QGpT
    - metric: Recall@K ($K \in \{1, 5, 10\}$)
    - dataset: Mimo_ch, Mimo_en, feta, ottqa, e2ewtq
    - 使用BGE-M3作為embedding model
    - 使用LLama 3.1 8b-instruct作為synthetic querie generation的model 
    - k的設置為10，因此生成10個queries
    - 在header-aware kmeans中 $\alpha=0.2$
    - 在Fixed Weight Fusion中 $W_{q}=0.3$
    - 在Dynamic Weight Fusion中 $\beta=0.3$

## 主要實驗
比較以下在各實驗集Recall@1, Recall@5, Recall@10的表現
- QGpT
- STAR
    - w/ FWF
    - w/ DWF
- Main Results
    - 將其5個資料集數據結合成一個圖表
    - Mimo (ch)

        |Method|R@1|R@5|R@10|
        |---|---|---|---|
        |QGpT|		49.81|	71.06|	77.23|
        |STAR w/ FWF (0.7)	|	51.36|	72.16|	78.08|
        |STAR w/ DWF |		51.58|	72.15|	77.99|

    - Mimo (en)

        |Method|R@1|R@5|R@10|
        |---|---|---|---|
        |QGpT|50.66|72.35|80.8|
        |STAR w/ FWF (0.7)	|58.34|	76.98|82.5|
        |STAR w/ DWF |	58.89|	77.72|82.89|


    - OTTQA

        |Method|R@1|R@5|R@10|
        |---|---|---|---|
        |QGpT|54.45|	78.14|	86.68|
        |STAR w/ FWF (0.7)	|53.84|	80.17|	88.17|
        |STAR w/ DWF |		54.07|	79.99|	88.08|

    - FetaQA

        |Method|R@1|R@5|R@10|
        |---|---|---|---|
        |QGpT|33.95|	50.87|57.86|
        |STAR w/ FWF (0.7)	|36|54.92|62.21|
        |STAR w/ DWF |	36.25|54.77|62.21|
        
    - E2E-WTQ

        |Method|R@1|R@5|R@10|
        |---|---|---|---|
        |QGpT|41.49|	65.98|72.61|
        |STAR w/ FWF (0.7)	|58.51|	85.89|	90.04|
        |STAR w/ DWF |		58.51|	85.06|	90.06|


## analysis
### Ablation Study
1. Ablation Study on STAR
- 比較我的方法中各種module是否有用
- 使用Avg.(所有dataset的平均) Recall@1, Recall@5, Recall@10
- STAR (full): STAR w/ DWF
- w/o SCDG
- w/o WF
- QGpT baseline

|Method|R@1|R@5|R@10|
|---|---|---|---|
|STAR (full)|51.86|73.94|80.25|
|w/o SCDG|47.07|68.43|75.88|
|w/o WF|49.08|69.69|77.02|
|QGpT|45.47|67.68|75.04|

2. ablation 2:
- 不同權重的影響 (FWF vs DWF)
- 使用Avg.(所有dataset的平均) Recall@1, Recall@5, Recall@10
- STAR w/ FWF (table weight = 0.1)
- STAR w/ FWF (0.3)
- STAR w/ FWF (0.5)
- STAR w/ FWF (0.7)
- STAR w/ FWF (0.9)
- STAR w/ DWF

|Method|R@1|R@5|R@10|
|---|---|---|---|
|STAR w/ FWF (0.1)|46.61|	68.76|	76.22|
|STAR w/ FWF (0.3)|49.25|	71.66|	78.65|
|STAR w/ FWF (0.5)|51.33|	73.27|	79.81|
|STAR w/ FWF (0.7)|	51.62|	74.02|	80.2|
|STAR w/ FWF (0.9)|	48.61|	72.22|	79.28|
|STAR w/ DWF| 	51.86|	73.94|	80.25|

&rarr; 證明 dynamic、語義 aware 的權重比手動固定更好
&rarr; 自然引到「learned weighting」作為 future work