# STAR Paper - Reference Usage Documentation

本文檔說明論文中使用的所有參考文獻、選擇理由以及使用位置。

## 核心參考文獻（必要引用）

### 1. QGpT - 主要對比基準
**Reference:** `liang-etal-2025-improving-table`
```bibtex
@inproceedings{liang-etal-2025-improving-table,
    title = "Improving Table Retrieval with Question Generation from Partial Tables",
    author = "Liang, Hsing-Ping and Chang, Che-Wei and Fan, Yao-Chung",
    booktitle = "Proceedings of the 4th Table Representation Learning Workshop",
    year = "2025",
    pages = "217--228"
}
```

**使用位置：**
- **Introduction (第 122 行)**: 介紹 QGpT 方法及其缺陷
- **Related Works - 查詢生成與資料增強 (第 140 行)**: 說明 QGpT 的 synthetic question generation
- **Experiments - Baselines (第 232 行)**: 定義主要對比基準

**選擇理由：**
- STAR 的核心動機就是解決 QGpT 的兩個缺陷
- 這是我們直接對比和改進的方法
- 必須引用以建立研究脈絡

---

## Introduction 部分（建立背景與挑戰）

### 2. Dense Passage Retrieval (DPR)
**Reference:** `karpukhin2020dense`
```bibtex
@inproceedings{karpukhin2020dense,
  title={Dense passage retrieval for open-domain question answering},
  author={Karpukhin, Vladimir and others},
  booktitle={EMNLP},
  year={2020}
}
```

**使用位置：**
- **Introduction (第 120 行)**: 說明傳統檢索器的 token 限制問題
- **Related Works - 表格檢索 (第 137 行)**: 密集檢索方法的應用

**選擇理由：**
- 代表性的密集檢索方法
- 說明為何完整表格難以在 token 限制內編碼
- 建立表格檢索面臨的挑戰背景

### 3. CLTR - 表格分段方法
**Reference:** `pan2021table`
```bibtex
@inproceedings{pan2021table,
  title={Cltr: An end-to-end, transformer-based system for cell level table retrieval},
  author={Pan, Feifei and Canim, Mustafa and Glass, Michael and others},
  booktitle={NAACL-HLT},
  year={2021}
}
```

**使用位置：**
- **Introduction (第 120 行)**: 基於表格分段的方法（關鍵詞匹配）

**選擇理由：**
- 代表基於表格分段的檢索方法
- 說明現有方法依賴關鍵詞匹配而非語義表示
- 凸顯 STAR 使用 LLM 生成 semantic queries 的優勢

---

## Related Works 部分（相關技術背景）

### 4. BM25 - 稀疏檢索基準
**Reference:** `robertson2009probabilistic`
```bibtex
@article{robertson2009probabilistic,
  title={The probabilistic relevance framework: BM25 and beyond},
  author={Robertson, Stephen and Zaragoza, Hugo},
  journal={Foundations and Trends in Information Retrieval},
  year={2009}
}
```

**使用位置：**
- **Related Works - 表格檢索 (第 137 行)**: 早期稀疏檢索方法

**選擇理由：**
- 經典的稀疏檢索基準
- 對比說明從稀疏到密集檢索的演進

### 5. TaBERT - 結構感知編碼
**Reference:** `yin2020tabert`
```bibtex
@inproceedings{yin2020tabert,
  title={TaBERT: Pretraining for joint understanding of textual and tabular data},
  author={Yin, Pengcheng and Neubig, Graham and others},
  booktitle={ACL},
  year={2020}
}
```

**使用位置：**
- **Related Works - 表格檢索 (第 137 行)**: 結構感知的表格編碼方法

**選擇理由：**
- 代表性的表格預訓練模型
- 說明表格檢索中對結構建模的重要性

### 6. GPL - 偽查詢生成
**Reference:** `wang2022gpl`
```bibtex
@inproceedings{wang2022gpl,
  title={GPL: Generative pseudo labeling for unsupervised domain adaptation},
  author={Wang, Kexin and Thakur, Nandan and others},
  booktitle={NAACL-HLT},
  year={2022}
}
```

**使用位置：**
- **Related Works - 查詢生成與資料增強 (第 140 行)**: 查詢生成技術

**選擇理由：**
- 查詢生成在文本檢索中的經典方法
- 說明 query generation 作為資料增強技術的有效性

### 7. InPars - LLM 查詢生成
**Reference:** `bonifacio2022inpars`
```bibtex
@inproceedings{bonifacio2022inpars,
  title={InPars: Data augmentation for information retrieval using large language models},
  author={Bonifacio, Luiz and Abonizio, Hugo and others},
  booktitle={SIGIR},
  year={2022}
}
```

**使用位置：**
- **Related Works - 查詢生成與資料增強 (第 140 行)**: 使用 LLM 進行查詢生成

**選擇理由：**
- 探索 LLM 用於查詢生成的可行性
- 為 QGpT 和 STAR 使用 LLM 生成 queries 提供背景

### 8. 聚類在資訊檢索中的應用
**Reference:** `liu2004cluster`
```bibtex
@inproceedings{liu2004cluster,
  title={Cluster-based retrieval using language models},
  author={Liu, Xiaoyong and Croft, W Bruce},
  booktitle={SIGIR},
  year={2004}
}
```

**使用位置：**
- **Related Works - 聚類在資訊檢索中的應用 (第 143 行)**: 聚類技術在檢索中的應用

**選擇理由：**
- 說明聚類技術在資訊檢索中的有效性
- 支持 STAR 使用聚類進行實例選擇的設計

---

## Experiments 部分（資料集與模型）

### 9. Mimo - 多尺度表格基準
**Reference:** `li2025mimotable`
```bibtex
@inproceedings{li2025mimotable,
  title={MiMoTable: A Multi-scale Spreadsheet Benchmark},
  author={Li, Zheng and Du, Yang and others},
  booktitle={COLING},
  year={2025}
}
```

**使用位置：**
- **Experiments - 資料集 (第 227 行)**: Mimo (ch/en) 資料集

**選擇理由：**
- 實驗使用的主要資料集之一
- 多語言（中文/英文）評估

### 10. OTTQA - 開放域表格問答
**Reference:** `chenopen`
```bibtex
@inproceedings{chenopen,
  title={Open Question Answering over Tables and Text},
  author={Chen, Wenhu and Chang, Ming-Wei and others},
  booktitle={ICLR},
  year={2021}
}
```

**使用位置：**
- **Experiments - 資料集 (第 228 行)**: OTTQA 資料集

**選擇理由：**
- 實驗使用的資料集
- 開放域表格問答基準

### 11. FetaQA - 基於表格的問答
**Reference:** `nan2022fetaqa`
```bibtex
@article{nan2022fetaqa,
  title={FeTaQA: Free-form table question answering},
  author={Nan, Linyong and Hsieh, Chiachun and others},
  journal={TACL},
  year={2022}
}
```

**使用位置：**
- **Experiments - 資料集 (第 229 行)**: FetaQA 資料集

**選擇理由：**
- 實驗使用的資料集
- 自由形式的表格問答

### 12. E2E-WTQ - 端到端表格問答
**Reference:** `pasupat2015compositional`
```bibtex
@inproceedings{pasupat2015compositional,
  title={Compositional semantic parsing on semi-structured tables},
  author={Pasupat, Panupong and Liang, Percy},
  booktitle={ACL-IJCNLP},
  year={2015}
}
```

**使用位置：**
- **Experiments - 資料集 (第 230 行)**: E2E-WTQ 資料集

**選擇理由：**
- 實驗使用的資料集
- 經典的表格語義解析資料集

### 13. BGE-M3 - Embedding 模型
**Reference:** `xiao2023cpack`
```bibtex
@article{xiao2023cpack,
  title={C-Pack: Packaged resources to advance general Chinese embedding},
  author={Xiao, Shitao and Liu, Zheng and others},
  journal={arXiv preprint arXiv:2309.07597},
  year={2023}
}
```

**使用位置：**
- **Experiments - 實現細節 (第 234 行)**: BGE-M3 作為 embedding 模型

**選擇理由：**
- 實驗使用的具體模型
- 支持多語言的先進 embedding 模型

---

## 未使用但存在於 references.bib 的文獻

以下文獻已添加到 references.bib 但目前論文中未引用，建議移除以保持 4 頁 short paper 的精簡：

### 可移除的文獻：
- `zhong2017seq2sql` - Seq2SQL（未使用）
- `chen2017reading` - Reading Wikipedia（未使用）
- `kwiatkowski2019natural` - Natural Questions（未使用）
- `glass2021capturing` - RCI model（未使用）
- `wu2025mimosa` - MMQA（未使用）
- `chen2024tablerag` - TableRAG（未使用）
- `khattab2020colbert` - ColBERT（未使用）
- `lin2023durepa` - DUREPA（未使用）
- 以及其他大量未引用的文獻

---

## 總結

### 當前使用的參考文獻數量：13 個

**分類統計：**
- 核心對比方法：1 個 (QGpT)
- 背景方法：3 個 (DPR, CLTR, BM25)
- 相關技術：4 個 (TaBERT, GPL, InPars, Clustering)
- 資料集：4 個 (Mimo, OTTQA, FetaQA, E2E-WTQ)
- 模型：1 個 (BGE-M3)

### 建議：
對於 4 頁 short paper，當前 13 個引用已經非常精簡且聚焦。每個引用都有明確的作用：
1. 建立問題背景（DPR, CLTR, BM25）
2. 說明相關技術（TaBERT, GPL, InPars, Clustering）
3. 定義對比基準（QGpT）
4. 描述實驗設置（資料集與模型）

所有引用都直接支持論文的核心論點，沒有無關文獻。
