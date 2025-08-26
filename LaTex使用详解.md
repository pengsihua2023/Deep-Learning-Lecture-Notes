# 📘 LaTeX 使用详解

## 1. 文档基本结构

一个 LaTeX 文档通常由三部分构成：

```latex
\documentclass[选项]{类型}  % 文档类型: article, report, book, beamer等

% 预导言区 (导言区)：加载宏包、定义命令等
\usepackage{amsmath, amssymb, graphicx, hyperref}

\begin{document}   % 文档开始

正文内容...

\end{document}     % 文档结束
```

常用文档类型：

* `article`（文章，期刊论文常用）
* `report`（报告）
* `book`（书籍）
* `beamer`（幻灯片）

---

## 2. 文本排版

### 字体样式

```latex
\textbf{粗体}, \textit{斜体}, \underline{下划线}
\texttt{等宽字体}, \emph{强调}
```

### 段落与换行

* 空一行 = 新段落
* `\\` 或 `\newline` = 强制换行
* `\par` = 新段落

---

## 3. 数学公式（核心优势）

### 行内公式

```latex
这是行内公式 $E=mc^2$。
```

### 独立公式

```latex
\begin{equation}
   \int_a^b f(x)\, dx = F(b) - F(a)
\end{equation}
```

### 常用符号

* 上标/下标：`x^2`, `a_{ij}`
* 分数：`\frac{a}{b}`
* 根号：`\sqrt{x}`, `\sqrt[n]{x}`
* 求和/积分：

  ```latex
  \sum_{i=1}^n i^2, \quad \int_0^\infty e^{-x} dx
  ```
* 矩阵：

  ```latex
  \begin{bmatrix}
     1 & 2 \\
     3 & 4
  \end{bmatrix}
  ```

### 多行对齐

```latex
\begin{align}
   a &= b + c \\
   x &= y - z
\end{align}
```

---

## 4. 列表与结构

### 无序列表

```latex
\begin{itemize}
   \item 第一项
   \item 第二项
\end{itemize}
```

### 有序列表

```latex
\begin{enumerate}
   \item 第一项
   \item 第二项
\end{enumerate}
```

### 描述列表

```latex
\begin{description}
   \item[猫] 喜欢睡觉
   \item[狗] 喜欢玩耍
\end{description}
```

---

## 5. 图表插入

### 插图

```latex
\usepackage{graphicx}

\begin{figure}[htbp]
   \centering
   \includegraphics[width=0.6\textwidth]{example.png}
   \caption{示例图片}
   \label{fig:example}
\end{figure}
```

### 表格

```latex
\begin{table}[htbp]
   \centering
   \begin{tabular}{|c|c|c|}
      \hline
      姓名 & 年龄 & 分数 \\
      \hline
      张三 & 20 & 95 \\
      李四 & 21 & 88 \\
      \hline
   \end{tabular}
   \caption{成绩表}
   \label{tab:scores}
\end{table}
```

---

## 6. 交叉引用与目录

```latex
\label{eq:einstein}     % 给公式打标签
如公式 \ref{eq:einstein} 所示...

\tableofcontents   % 自动生成目录
```

---

## 7. 参考文献

### 手动引用

```latex
\begin{thebibliography}{99}
   \bibitem{einstein} Einstein, A. (1905). On the Electrodynamics of Moving Bodies.
   \bibitem{latex} Lamport, L. (1994). LaTeX: A Document Preparation System.
\end{thebibliography}
```

### BibTeX 自动管理

```latex
\bibliographystyle{plain}
\bibliography{references}
```

在 `references.bib` 文件里写条目：

```bibtex
@article{einstein1905,
   author = {Einstein, A.},
   title = {On the Electrodynamics of Moving Bodies},
   year = {1905},
   journal = {Annalen der Physik}
}
```

---

## 8. 定理、证明、定义（数学写作）

```latex
\usepackage{amsthm}

\newtheorem{theorem}{定理}
\newtheorem{definition}{定义}

\begin{theorem}
   若 $a > b$ 且 $b > c$，则 $a > c$。
\end{theorem}

\begin{proof}
   由不等式传递性可得。$\qed$
\end{proof}
```

---

## 9. 超链接

```latex
\usepackage{hyperref}
\href{https://www.overleaf.com}{Overleaf}
```

---

## 10. 常用宏包推荐

* **数学**：`amsmath, amssymb, amsthm`
* **图形**：`graphicx`
* **参考文献**：`biblatex, natbib`
* **超链接**：`hyperref`
* **页面设置**：`geometry`
* **颜色支持**：`xcolor`
* **算法伪代码**：`algorithm, algorithmic`

---

# ✅ 总结

LaTeX 使用流程：

1. **文档类型**：`\documentclass{}`
2. **导言区加载宏包**
3. **正文环境**：文本 + 数学 + 图表
4. **结构化工具**：交叉引用、目录、参考文献
5. **扩展宏包**：根据需求灵活扩展

---


