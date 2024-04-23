# 在VSCODE中配置LaTeX编译环境

## 前言

LaTeX 作为一种强大的排版系统，对于理工科，特别是公式比较多的数(tu)学(tou)专业，其重要性自不必多说，不在本文探讨范围之内。

而选择一个比较好的编译器是很重要的，至少对笔者而言是如此。笔者前期使用的TeXstudio进行文档的编译的，但是其编译速度比较慢，并且页面不是很美观。最让人头疼的是，当公式比较长的时候，使用的括号就比较多，但 Texstudio 的代码高亮功能实在是....（它对于括号根本就没有高亮，头秃）

VSCODE不仅能够对代码高亮，不同级别括号用不同颜色标注了，颜值也很高。vscode 最突出的特点就是其强大的插件功能，每个使用者都能够根据自己的需求和想法下载相应的插件，从而将之配置为高度个性化的编辑器。可以这么说，每个使用者的 vscode 都不一样，为其专属定制编辑器。

## 1 TeX Live 下载与安装

笔者选用的 Tex 系统是 TeX Live ，如果您想了解 TeX Live 和 MiKTeX 的区别，可以查看此篇文章：[在windows上使用TeX：TeX Live与MikTex的对比](http://www.cnblogs.com/liuliang1999/p/12656706.html)

接下来是 TeX Live 的下载与安装说明：

通过网址 ：[进入 ISO 下载页面](http://tug.org/texlive/acquire-iso.html)，进入清华大学镜像网站，点击链接进行 TeX Live 下载。

找到下载好的压缩包，右键，在打开方式中选择“Windows 资源管理器"打开；找到 "install-tl-windows" 文件，为了后面不必要的麻烦，右键以管理员身份运行。

安装过程中，需要进行路径的更改；由于 TeX Live 自带的 TeXworks 不怎么好用，并且此文主要将 vscode 作为 LaTeX 的编辑器，故而取消 安装 TeXworks 前端的选项，再点击安装。

检查安装是否正常： 按win + R 打开运行，输入cmd，打开命令行窗口；然后输入命令
```
pdflatex -v
```

## 2 vscode下载

[VSCODE官网下载](http://code.visualstudio.com/)

一定要选上"添加到PATH”这个选项，能省很多麻烦。其余如图所示，自行选择。

## 3 LaTeX的支持插件 LaTeX Workshop安装

- 点击拓展图标，打开拓展；
- 输入"latex workshop"，选择第一个LaTeX Workshop插件；
- 点击"install"进行安装，等待安装完成；

## 4 打开LaTeX环境设置页面

- 点击设置图标
- 点击设置
- 转到 UI 设置页面

![UI 设置页面](https://pic3.zhimg.com/80/v2-9f7cd81aeac1a5565d3f0e39c902ab02_720w.webp)

- 点击下图 1 处打开 json 文件，进入代码设置页面

![UI 设置页面](https://pic3.zhimg.com/80/v2-8a214dc687d81cf4ecb25c5e5e9ea69e_720w.webp)

UI 设置页面和JSON设置页面均为设置页面，其功能是一样的。不同的是，UI 设置页面交互能力较强，但一些设置需要去寻找，比较麻烦；而JSON设置页面虽然相对 UI 而言不那么直观，但却可以对自己想要的功能直接进行代码编写，且代码设置可以直接克隆别人的代码到自己的编辑器中，从而直接完成相应设置，比较便捷。

- 可以直接按***Ctrl + ，***进入设置页面。

## 5 LaTeX环境的代码配置

![LaTeX配置代码展示](https://pic2.zhimg.com/80/v2-175b476a0cf425370065156bf807e205_720w.webp)

```json
{
    "latex-workshop.latex.autoBuild.run": "never",
    "latex-workshop.showContextMenu": true,
    "latex-workshop.intellisense.package.enabled": true,
    "latex-workshop.message.error.show": false,
    "latex-workshop.message.warning.show": false,
    "latex-workshop.latex.tools": [
        {
            "name": "xelatex",
            "command": "xelatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOCFILE%"
            ]
        },
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOCFILE%"
            ]
        },
        {
            "name": "latexmk",
            "command": "latexmk",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-pdf",
                "-outdir=%OUTDIR%",
                "%DOCFILE%"
            ]
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": [
                "%DOCFILE%"
            ]
        }
    ],
    "latex-workshop.latex.recipes": [
        {
            "name": "XeLaTeX",
            "tools": [
                "xelatex"
            ]
        },
        {
            "name": "PDFLaTeX",
            "tools": [
                "pdflatex"
            ]
        },
        {
            "name": "BibTeX",
            "tools": [
                "bibtex"
            ]
        },
        {
            "name": "LaTeXmk",
            "tools": [
                "latexmk"
            ]
        },
        {
            "name": "xelatex -> bibtex -> xelatex*2",
            "tools": [
                "xelatex",
                "bibtex",
                "xelatex",
                "xelatex"
            ]
        },
        {
            "name": "pdflatex -> bibtex -> pdflatex*2",
            "tools": [
                "pdflatex",
                "bibtex",
                "pdflatex",
                "pdflatex"
            ]
        },
    ],
    "latex-workshop.latex.clean.fileTypes": [
        "*.aux",
        "*.bbl",
        "*.blg",
        "*.idx",
        "*.ind",
        "*.lof",
        "*.lot",
        "*.out",
        "*.toc",
        "*.acn",
        "*.acr",
        "*.alg",
        "*.glg",
        "*.glo",
        "*.gls",
        "*.ist",
        "*.fls",
        "*.log",
        "*.fdb_latexmk"
    ],
    "latex-workshop.latex.autoClean.run": "onFailed",
    "latex-workshop.latex.recipe.default": "lastUsed",
    "latex-workshop.view.pdf.internal.synctex.keybinding": "double-click"
}
```

## 6 LaTeX配置代码解读

```json
"latex-workshop.latex.autoBuild.run": "never"
```

设置何时使用默认的(第一个)编译链自动构建 LaTeX 项目，即什么时候自动进行代码的编译。有三个选项：

- 1. onFileChange：在检测任何依赖项中的文件更改(甚至被其他应用程序修改)时构建项目，即当检测到代码被更改时就自动编译tex文件；
- 2. onSave : 当代码被保存时自动编译文件；
- 3. never: 从不自动编译，即需编写者手动编译文档

```json
"latex-workshop.showContextMenu": true
```

启用上下文LaTeX菜单。此菜单默认状态下停用，即变量设置为false，因为它可以通过新的 LaTeX 标记使用（新的 LaTeX 标记能够编译文档，将在下文提及）。只需将此变量设置为true即可恢复菜单。即此命令设置是否将编译文档的选项出现在鼠标右键的菜单中。

下两图展示两者区别，第一幅图为设置false情况，第二幅图为设置true情况。可以看到的是，设置为true时，菜单中多了两个选项，其中多出来的第一个选项为进行tex文件的编译，而第二个选项为进行正向同步，即从代码定位到编译出来的 pdf 文件相应位置，下文会进行提及。
![](https://pic2.zhimg.com/80/v2-d614a4ecc4b7da18624926b5a49f144d_720w.webp)
![](https://pic3.zhimg.com/80/v2-c66892ca1a278e961fe9fdb124896126_720w.webp)

```json
"latex-workshop.intellisense.package.enabled": true
```
设置为true，则该拓展能够从使用的宏包中自动提取命令和环境，从而补全正在编写的代码。

```json
"latex-workshop.message.error.show"  : false,
"latex-workshop.message.warning.show": false
```
这两个命令是设置当文档编译错误时是否弹出显示出错和警告的弹窗。因为这些错误和警告信息能够从终端中获取，且弹窗弹出比较烦人，故而笔者设置均设置为false。

```json
"latex-workshop.latex.tools": [
        {
            "name": "xelatex",
            "command": "xelatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOCFILE%"
            ]
        },
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOCFILE%"
            ]
        },
        {
            "name": "latexmk",
            "command": "latexmk",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-pdf",
                "-outdir=%OUTDIR%",
                "%DOCFILE%"
            ]
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": [
                "%DOCFILE%"
            ]
        }
    ]
```

这些代码是定义在下文 recipes 编译链中被使用的编译命令，此处为默认配置，不需要进行更改。其中的name为这些命令的标签，用作下文 recipes 的引用；而command为在该拓展中的编译方式。

可以更改的代码为，将编译方式: pdflatex 、 xelatex 和 latexmk 中的%DOCFILE更改为%DOC。%DOCFILE表明编译器访问没有扩展名的根文件名，而%DOC表明编译器访问的是没有扩展名的根文件完整路径。这就意味着，使用%DOCFILE可以将文件所在路径设置为中文，但笔者不建议这么做，因为毕竟涉及到代码，当其余编译器引用时该 tex 文件仍需要根文件完整路径，且需要为英文路径。笔者此处设置为%DOCFILE仅是因为之前使用 TeXstudio，导致路径已经是中文了。

更多详情可以访问 [github 中 LaTeX-Workshop 的 Wiki](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile%23placeholders)

```json
"latex-workshop.latex.recipes": [
        {
            "name": "XeLaTeX",
            "tools": [
                "xelatex"
            ]
        },
        {
            "name": "PDFLaTeX",
            "tools": [
                "pdflatex"
            ]
        },
        {
            "name": "BibTeX",
            "tools": [
                "bibtex"
            ]
        },
        {
            "name": "LaTeXmk",
            "tools": [
                "latexmk"
            ]
        },
        {
            "name": "xelatex -> bibtex -> xelatex*2",
            "tools": [
                "xelatex",
                "bibtex",
                "xelatex",
                "xelatex"
            ]
        },
        {
            "name": "pdflatex -> bibtex -> pdflatex*2",
            "tools": [
                "pdflatex",
                "bibtex",
                "pdflatex",
                "pdflatex"
            ]
        }
    ]
```

此串代码是对编译链进行定义，其中name是标签，也就是出现在工具栏中的链名称；tool是name标签所对应的编译顺序，其内部编译命令来自上文latex-workshop.latex.recipes中内容。

定义完成后，能够在 vscode 编译器中能够看到的编译顺序，具体看下图：
![](https://pic2.zhimg.com/80/v2-42b419223d07a18fd406ecd54f674fd1_720w.webp)


编译链的存在是为了更方便编译，因为如果涉及到.bib文件，就需要进行多次不同命令的转换编译，比较麻烦，而编译链就解决了这个问题。

```json
"latex-workshop.latex.clean.fileTypes": [
        "*.aux",
        "*.bbl",
        "*.blg",
        "*.idx",
        "*.ind",
        "*.lof",
        "*.lot",
        "*.out",
        "*.toc",
        "*.acn",
        "*.acr",
        "*.alg",
        "*.glg",
        "*.glo",
        "*.gls",
        "*.ist",
        "*.fls",
        "*.log",
        "*.fdb_latexmk"
    ]
```
这串命令则是设置编译完成后要清除掉的辅助文件类型，若无特殊需求，无需进行更改。

```json
"latex-workshop.latex.autoClean.run": "onFailed"
```

这条命令是设置什么时候对上文设置的辅助文件进行清除。其变量有：

- 1. onBuilt : 无论是否编译成功，都选择清除辅助文件；
- 2. onFailed : 当编译失败时，清除辅助文件；
- 3. never : 无论何时，都不清除辅助文件。

由于 tex 文档编译有时需要用到辅助文件，比如编译目录和编译参考文献时，如果使用onBuilt命令，则会导致编译不出完整结果甚至编译失败；

而有时候将 tex 文件修改后进行编译时，可能会导致 pdf 文件没有正常更新的情况，这个时候可能就是由于辅助文件没有进行及时更新的缘故，需要清除辅助文件了，而never命令做不到这一点；

故而笔者使用了onFailed，同时解决了上述两个问题。

```json
"latex-workshop.latex.recipe.default": "lastUsed"
```
该命令的作用为设置 vscode 编译 tex 文档时的默认编译链。有两个变量： 
- 1. first : 使用latex-workshop.latex.recipes中的第一条编译链，故而您可以根据自己的需要更改编译链顺序； 
- 2. lastUsed : 使用最近一次编译所用的编译链。

```json
"latex-workshop.view.pdf.internal.synctex.keybinding": "double-click"
```
用于反向同步（即从编译出的 pdf 文件指定位置跳转到 tex 文件中相应代码所在位置）的内部查看器的快捷键绑定。变量有两种：

- 1. ctrl-click ： 为默认选项，使用Ctrl/cmd+鼠标左键单击
- 2. double-click : 使用鼠标左键双击

## 7 PDFLaTeX 编译模式与 XeLaTeX 区别如下：

- 1. PDFLaTeX 使用的是TeX的标准字体，所以生成PDF时，会将所有的非 TeX 标准字体进行替换，其生成的 PDF 文件默认嵌入所有字体；而使用 XeLaTeX 编译，如果说论文中有很多图片或者其他元素没有嵌入字体的话，生成的 PDF 文件也会有些字体没有嵌入。
- 2. XeLaTeX 对应的 XeTeX 对字体的支持更好，允许用户使用操作系统字体来代替 TeX 的标准字体，而且对非拉丁字体的支持更好。
- 3. PDFLaTeX 进行编译的速度比 XeLaTeX 速度快。

[引用自 蛐蛐蛐先生的博客 ](https://blog.csdn.net/qysh123/article/details/11833649%3Futm_source%3Dtuicool)