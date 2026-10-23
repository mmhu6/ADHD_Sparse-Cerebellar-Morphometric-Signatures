# 数据格式

本仓库不分发真实 `cerebellum.csv`，也不分发其逐人预测或训练模型。使用者需自行准备已获授权、符合下述模式的本地 CSV；上游影像处理与特征提取不属于本项目。

## 元数据

| 列名 | 代码使用方式 |
| --- | --- |
| `Sub` | 受试者标识，不作为分类器输入；部分输出保留此列 |
| `DX` | 二分类目标：0=对照，1=ADHD；不得直接传入模型作为预测特征 |
| `Age` | 数值年龄；合成示例按年生成；真实输入需使用一致单位 |
| `Gender` | 数值二元性别编码；合成示例用 0/1，真实编码对应关系需由数据提供方说明 |
| `ICVcm3` | 颅内容积，cm³；归一化计算要求正值 |
| `ADHDIndex`, `Inattentive`, `HyperImpulsive` | 症状指标；代码将其排除出预测候选特征 |
| `MedStatus` | 用药状态；代码将其排除出预测候选特征 |

为兼容所有输出和旧版分析，建议提供全部 9 个元数据字段。各脚本的缺失值处理不同：旧版先删除缺少 DX/Age/Gender/ICV 的行，新版部分变换器在训练折内插补数值；所有输入仍应提前完成标签和字段质控。代码不会自动理解任意额外元数据列：基础脚本会将未列入排除表且不含 `norm` 的列作为脑特征候选，新增 ID、站点或症状列前应明确排除。

## 形态学字段

参考模式为 282 列：9 个元数据字段，以及 273 个脑形态学字段。其中原始厚度 39 列、ICV 归一化厚度 39 列、其他形态学 195 列。基础候选集为原始厚度加其他形态学，共 234 列；字段命名含 `norm` 与否是现有代码分组规则，不能据此推断其他字段的生物学单位。

区域前缀为 `Cerebellum`、`III`、`IV`、`V`、`VI`、`CrusI`、`CrusII`、`VIIB`、`VIIIA`、`VIIIB`、`IX`、`X`，另有一组以 `_A` 结尾的 `III` 区域字段。例如：`IIIleftcorticalthickness_A`。

每组区域含：

- 3 个厚度字段：`meancorticalthickness`、`rightcorticalthickness`、`leftcorticalthickness`。
- 3 个归一化厚度字段：以上字段后加 `norm`。全小脑平均厚度的归一化列名例外，为 `Cerebellumcorticalthicknessnorm`，对应原始列 `Cerebellummeancorticalthickness`。
- 15 个其他字段：`totalcm3`、`rightcm3`、`leftcm3`、`total`、`right`、`left`、`asymmetry`、`corticalthicknessasymmetry`、`greymattercm3`、`rightgreymattercm3`、`leftgreymattercm3`、`greymatter`、`rightgreymatter`、`leftgreymatter`、`greymatterasymmetry`。

归一化厚度的参考公式为：

```text
normalized_thickness = raw_thickness × (1000 / ICVcm3)^(1/3)
```

这类确定性 ICV 缩放与 `cerebnorm_adhd_screen.py` 的对照组常模 z 偏离是两种不同的表示。

## 各入口要求

`cereb_adhd_screen.py` 按列名构造多视图，并比较以下 5 个既有锁定脑特征；`cerebnorm_adhd_screen.py` 和五模型消融也使用这 5 列：

```text
VIleftcorticalthickness
VIIIBrightcorticalthickness
Xleft
VIIIBtotalcm3
VIIIBleftcorticalthickness
```

五模型消融需要以上 5 列、`Age/Gender/ICVcm3` 和 `DX`；`Sub` 可选，存在时会写入逐人输出，缺失时使用行号。常模方法要求**每个实际拟合训练分区至少有 20 位对照**，不能只检查整张表的人数。

有限搜索脚本另使用 `cereb_adhd_optimization.CANDIDATE_FEATURES` 中的 12 列；v2 还使用 `AnatomicalFeatureEngineer.required_source_columns()` 返回的区域体积、灰质体积和厚度源列。为了运行全部实验，可使用完整参考模式；合成生成器会生成该模式的全部字段。

## 无真实数据的示例

```bash
python examples/generate_synthetic_data.py --rows 240 --seed 42 --output outputs/synthetic/cerebellum.csv
```

所有值由随机数生成，`Sub` 以 `SYNTHETIC_` 开头。生成器不读取本地真实 CSV、不估计真实数据分布，也不生成任何可用于科学结论的真实效应。为支持嵌套训练分区中的对照人数要求，命令行示例要求至少 160 行，并拒绝覆盖已有文件。
