# 編譯繁體中文版本的說明

## 重要提示

`star_zh.tex` 檔案包含繁體中文內容，必須使用 **XeLaTeX** 編譯器來編譯（而非傳統的 pdflatex）。

## 編譯指令

### 使用 XeLaTeX（推薦）

```bash
xelatex star_zh.tex
bibtex star_zh
xelatex star_zh.tex
xelatex star_zh.tex
```

### 使用 latexmk（自動化編譯）

```bash
latexmk -xelatex star_zh.tex
```

## 字型需求

文檔使用 `xeCJK` 套件來支援中文字符，預設字型為 `AR PL UMing TW`。

如果系統中沒有此字型，可以修改 `star_zh.tex` 中的第10行：

```latex
\setCJKmainfont{AR PL UMing TW}  % 修改為你系統中的中文字型
```

### 常見中文字型選項：

- **Linux**: `AR PL UMing TW`, `Noto Sans CJK TC`, `WenQuanYi Micro Hei`
- **macOS**: `PingFang TC`, `Heiti TC`, `STSong`
- **Windows**: `Microsoft JhengHei`, `MingLiU`, `SimSun`

## 檢查可用字型

### Linux
```bash
fc-list :lang=zh-tw
```

### macOS
```bash
fc-list :lang=zh-hant
```

## 故障排除

如果編譯時出現 "Font not found" 錯誤：

1. 檢查系統是否安裝了中文字型
2. 修改 `\setCJKmainfont{}` 為系統中存在的字型名稱
3. 確保使用 XeLaTeX 而非 pdflatex 編譯

## 範例：使用 Overleaf

如果在 Overleaf 上編譯：

1. 點擊左上角的 "Menu"
2. 將 "Compiler" 從 "pdfLaTeX" 改為 "XeLaTeX"
3. 點擊 "Recompile"
