# Cerebellum ADHD Research

基于小脑形态学特征的 ADHD 分类研究代码，包含传统机器学习比较、多视图融合、解剖特征工程、对照组常模偏离建模与五模型消融实验。

Research code for ADHD classification using cerebellar morphometry, multi-view fusion and control-defined normative deviations. Participant data and trained models are not distributed. See [reproducibility notes](docs/REPRODUCIBILITY.md) for the validation limits.

**项目状态：研究代码；许可证待项目权利人确定。** 当前结果属于探索性、选择后的内部验证，不能视为独立队列或临床验证。仓库不包含真实受试者数据、逐人预测、论文草稿或预训练模型。

## 快速开始

参考环境为 **Python 3.12**。在仓库根目录创建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source .venv/bin/activate
```

安装依赖并运行测试，不需要真实数据：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

生成完全合成的演示输入，运行五分类器 × 三种特征视图的简化实验：

```bash
python examples/generate_synthetic_data.py --output outputs/synthetic/cerebellum.csv
python cerebnorm_five_model_ablation.py --data outputs/synthetic/cerebellum.csv --mode smoke --n-jobs 1 --output outputs/demo_five_model
```

演示数据从随机数独立生成，列名遵循输入模式；其中的数值、类别效应和评分只用于检查软件。**合成数据上的 AUC 不代表真实研究性能。** `smoke` 仍会运行多个模型，需要一定计算时间。

## 脚本与运行方式

所有命令均在仓库根目录执行。使用真实数据时，将以下命令的 `--data` 替换为你已获授权的本地 CSV 路径。

| 脚本 | 用途 | 简化模式 |
| --- | --- | --- |
| `ADHD_c.py` | 原始论文分析：五模型筛选、特征消融、图表和敏感性分析 | 无；默认完整论文流程 |
| `cereb_adhd_screen.py` | 多视图分类、固定特征基线和融合比较 | `--mode fast` |
| `cereb_adhd_optimization.py` | 有限候选特征、XGBoost 参数与融合权重开发 | `--mode smoke` |
| `cereb_adhd_v2.py` | 解剖比例特征、稳定性选择与多模型融合 | `--mode smoke` |
| `cerebnorm_adhd_screen.py` | 训练折内对照组常模偏离建模与融合 | `--mode smoke` |
| `cerebnorm_five_model_ablation.py` | SVM、随机森林、Elastic Net、XGBoost、CatBoost 的特征视图消融 | `--mode smoke` |

```bash
python cereb_adhd_screen.py --data /path/to/cerebellum.csv --mode fast --output outputs/screen
python cereb_adhd_optimization.py --data /path/to/cerebellum.csv --mode smoke --output outputs/optimization
python cereb_adhd_v2.py --data /path/to/cerebellum.csv --mode smoke --output outputs/v2
python cerebnorm_adhd_screen.py --data /path/to/cerebellum.csv --mode smoke --output outputs/normative
```

以上五个新实验入口均支持 `--mode full`，会增加交叉验证、搜索或 bootstrap 计算量。各脚本参数可用 `python <script>.py --help` 查看。实验默认使用 CPU；五模型消融的 `--n-jobs 1` 适合先检查运行流程，`-1` 使用可用 CPU 核。

原始论文分析还需要 SHAP：

```bash
python -m pip install -r requirements-paper.txt
python ADHD_c.py --data /path/to/cerebellum.csv --output outputs/paper
```

旧版入口默认开启重复交叉验证、1000 次置换和 SHAP，运行开销较大。`PUBLICATION_MODE` 及敏感性分析开关保留在脚本顶部；本次发布整理没有更改科学参数。

## 输入与输出

输入 CSV 每行对应一位受试者。`DX` 使用 `0=对照、1=ADHD`；年龄、性别编码、ICV 和形态学字段必须符合脚本约定。完整列名、ICV 归一化公式及各脚本要求见 [数据说明](docs/DATA.md)。代码消费已提取的特征表，不包含 MRI 原始影像预处理。

新实验通常输出 `model_comparison.csv`、`fold_metrics.csv`、`oof_predictions.csv`、`bootstrap_comparisons.csv`、`run_config.json` 和 `summary.txt`。五模型实验还输出 `feature_view_auc.csv` 与 `selected_hyperparameters.csv`；其他开发实验输出特征或调参过程。旧版输出论文图表和 `research_summary.txt`。

`oof_predictions.csv` 包含个体诊断、人口学资料或预测值，仍属于受试者级派生数据，默认不纳入版本控制。AUC 的汇总口径、选择偏差和常模拟合范围见 [复现与方法边界](docs/REPRODUCIBILITY.md)。

## 项目结构

```text
.
├── ADHD_c.py                         # 原始论文分析入口
├── cereb_adhd_screen.py              # 共享变换器、基线与多视图实验
├── cereb_adhd_optimization.py        # 有限搜索实验
├── cereb_adhd_v2.py                  # 解剖特征实验
├── cerebnorm_adhd_screen.py          # 对照组常模实验
├── cerebnorm_five_model_ablation.py  # 五模型特征消融
├── tests/                           # 合成数据与测试
├── examples/                        # 合成 CSV 生成入口
├── docs/                            # 数据、复现与发布说明
├── scripts/prepare_release.py        # 按白名单导出干净源码
├── requirements.txt                 # 固定直接依赖版本
├── requirements-paper.txt           # 旧版完整分析额外依赖
└── .github/workflows/tests.yml      # GitHub Actions
```

## 发布与贡献

按 [GitHub 发布说明](docs/RELEASING.md) 操作。执行 `python scripts/prepare_release.py` 可生成仅含公开文件的目录和 ZIP；原始本地数据及输出保留在原处。提交问题时请提供脚本、参数、依赖版本和可复现的合成输入，避免粘贴受试者资料。

**许可证尚未选定，当前没有授予开源复用许可。** 发布前由项目权利人确认代码权属并添加许可证。依赖包按各自许可证分发。论文标题、作者和 DOI 尚未在代码中确认，因此本仓库暂不提供虚构的引用条目。
