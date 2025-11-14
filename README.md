我現在在寫一個ACM short paper，頁數含reference在四頁以內，我發現QGpT這篇論文的兩個缺點，我針對這兩點去改進，成功改進其效果
請你跟我一起討論我的論文內容，並作為reviewer提出質疑與抨擊，一起完成完整的論文
並請先參考 @STAR_ACM_PAPER/star_ZH.tex中的初稿，先幫我寫成中文

# QGpT缺點及我改進的方法
1. QGpT在取partial table時太過heuristic，直接擷取前k個instances與table header作為partial table
- 可能問題：可能會擷取到類似或是不重要的instance
- 解決想法： 使用Kmeans clustering來對instances 分群，分出k群取k個centroid instance作為partial table，這樣能夠選出k個instance足夠代表這個table的instance
    - kmeans 作法：對直接對table instance的embedding分出k群，並在後續用於選出centroid instances與生成query
    - Synthetic Query Generation: 原本的作法是直接將partial table丟給LLM生成能用這個table生成的問題，這些問題要作為user query與table之間的橋樑，我的方法會改成輸入為kmeans分群後的k個clustered tables，每一個clustered table生成一個問題，最終生成k個問題，進而確保synthetic queries能夠很好的涵蓋我的Partial Table，能夠真正與user query建立語意橋樑，我稱為Clustering-guided query Generation (CGQG)

2.  QGpT在處理partial table + synthetic queries時太粗糙，是直接將partial table與synthetic queries串接後送進encoder，作為最終的table representation
- 可能問題： 缺乏更細緻的語義整合
- 解決想法：Weighted Fusion: 將partial table與每一個synthetic query各別送進encoder，並將這K+1個embedding加權組合成最終的table representation
    - 方法分成三種
        - 1. Fixed Weight: 給予table固定權重 $\alpha$, 剩下權重平均分給synthetic queries
        - 2. Dynamic Fusion: 
            - 使用consine-based semantic attention機制，根據每個synthetic query與table的語義相似度分配權重
            - 計算table與query之間的相似度，利用softmax注意力機制，計算出權重
        - 3. Diverse Fusion: 
            - 使用Query Diversity Weighting機制，獎勵語義多樣性高的questions，懲罰重複或相似的questions，目標讓不同角度的questions獲得更高權重
            - 對每個query $q_i$，計算與其他queries的平均距離 (距離: $1-cos(q_i,q_j)$)，利用softmax計算多樣性權重 

# 論文章節
1. Introduction
2. Related Work (請幫我找出適合的related work並於reference中附上)
3. Methodology
4. Experiments
5. Conclusion and Future work
6. Reference


# experiments
## baseline
- BGE-M3, QGpT
## 實驗setup
- 參照QGpT設定
- 使用BGE-M3作為embedding model
- 使用LLama 3.1 8b-instruct作為synthetic querie generation的model 
- metric: Recall@1, Recall@5, Recall@10
- dataset: Mimo_ch, Mimo_en, feta, ottqa, e2ewtq
- k的設置為10
## 主要實驗
比較以下在各實驗集Recall@1, Recall@5, Recall@10的表現
- BGE-M3
- QGpT
- STAR
    - Fixed Weight $\alpha$(table_weight) = [0.5,0.6,0.7,0.8,0.9]
    - Dynamic Fusion (依照table與相似度給予權重)
    - Diversity Fusion (依照synthetic query之間的相異性給予權重)

## analysis
### Ablation Study
1. Ablation Study on STAR
比較我的方法中各種module是否有用
比較以下在各實驗集Recall@1, Recall@5, Recall@10的表現
- STAR
- w/o KMeans
- w/o CGQG
- w/o Weighted Fusion


### Visualize analysis
使用降維工具來視覺化embedding（t-sne或其他的，建議我哪種技術比較適合，並更合理且更好說服reviewer）
2. 分析kmeans table是否能夠好代表original table
    抽取某個dataset中的幾筆資料來對original table, top-k partial table, kmeans clustered partial table這個embedding來降至2d，分析他們三者之間的距離
3. Visualize the Effected of STAR
    抽取某個dataset中的幾筆資料來對user query(綠色), STAR(紅色), positive table(藍色), negative tables(灰色點)，對這些embedding降維到2D，分析他們STAR的效果
