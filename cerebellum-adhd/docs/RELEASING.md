# 发布到 GitHub

## 当前发布内容

发布包包含六个分析脚本、合成测试与示例、依赖、公开文档、忽略规则和 GitHub Actions。真实数据、实验输出、论文草稿、内部计划和虚拟环境保留在原目录，不会被打包。

许可证按项目负责人的选择暂不添加。**要正式作为开源项目发布，还需由代码权利人确认权属、选择许可证并填入真实版权信息。** 选定后在项目根目录添加 `LICENSE`，更新 README；重新打包时该文件会自动包含。论文标题、作者和 DOI 确定后再添加 `CITATION.cff`，不使用推测的信息。

## 生成干净发布包

在原项目根目录运行：

```bash
python -m unittest discover -s tests -v
python scripts/prepare_release.py
```

生成：

```text
dist/cerebellum-adhd/          # 可用作 GitHub 仓库根目录
dist/cerebellum-adhd.zip       # 对应 ZIP，包含顶层 cerebellum-adhd/ 文件夹
```

目录与 ZIP 中均有 `MANIFEST.sha256`，记录实际公开文件的 SHA-256。导出器只复制明确列出的文件；新增源文件后，需要更新 `scripts/prepare_release.py` 中的白名单。它不会读取或复制真实数据记录，也不会删除原项目中的文件。若目标目录或 ZIP 已存在，导出器会停止；可把上一版移至备份位置后再次生成。

推荐从 `dist/cerebellum-adhd/` 上传。直接拖拽整个原项目文件夹上传 GitHub 网页时，`.gitignore` 不会帮你过滤被手动选中的文件。

## Git 命令上传

先在 GitHub 新建空仓库，名称可用 `cerebellum-adhd`；为避免首次推送冲突，不要让 GitHub 预建 README、许可证或 .gitignore。以下命令由发布者在**干净发布目录**执行，将占位地址替换为实际仓库地址：

```bash
cd dist/cerebellum-adhd
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare cerebellum ADHD research code"
git remote add origin https://github.com/YOUR_ACCOUNT/cerebellum-adhd.git
git push -u origin main
```

上述命令尚未由本次整理执行。发布包没有 Git 历史；如果之后在已有仓库里使用本项目，新增 `.gitignore` 不会撤回已跟踪或历史中已经提交的数据。

不使用 Git 时，可将发布目录内文件上传至 GitHub 网页；请确认以点开头的 `.github/`、`.gitignore`、`.gitattributes` 也包含在内。将 ZIP 作为附件上传本身不会在仓库中展开源码。

## 发布时确认

- 在干净发布目录运行测试；不需要也不应复制真实 `cerebellum.csv`。
- 检查 Git 暂存文件或网页上传列表，确认没有真实数据、逐人预测和论文草稿。
- 确定许可证，补齐可确认的项目作者、论文引用和数据获取说明。
- 推送后查看 Actions 的实际结果，再决定是否建立版本标签或 Release。
