---
completed: true
---
Welcome to Anaconda! This topic can help you decide which Anaconda installer to download and whether to use [conda (CLI)](https://docs.conda.io/en/latest/) or [Anaconda Navigator (GUI)](https://www.anaconda.com/docs/tools/anaconda-navigator/main) for managing environments and packages. You’ll also find several resources to build your understanding of Anaconda’s most popular offerings.
>  CLI: conda
>  GUI: Anaconda Navigator

## Should I use Anaconda Distribution or Miniconda?
Both the Anaconda Distribution and Miniconda installers include the conda package and environment manager, but how you plan to use the software will determine which installer you want to choose.
>  conda (the open source package and environment manager) 在 Anaconda Distribution 和 Miniconda installer 中都有包含
>  二者的具体差异如下

|                                                                                       | Anaconda Distribution | Miniconda |
| ------------------------------------------------------------------------------------- | --------------------- | --------- |
| Created and published by Anaconda                                                     | Yes                   | Yes       |
| Has conda                                                                             | Yes                   | Yes       |
| Has [Anaconda Navigator](https://www.anaconda.com/docs/tools/anaconda-navigator/main) | Yes                   | No        |
| # of packages                                                                         | 300+                  | < 70      |
| Install space required                                                                | ~4.4 GB               | ~480 MB   |

### What’s your experience level?
I'm just starting out and don't know what packages I should use:
Install Anaconda Distribution! It includes over [300 standard data science and machine learning packages](https://www.anaconda.com/docs/getting-started/anaconda/main), which will give you a kickstart in your development journey.

I don't have much experience with the command line:
Install Anaconda Distribution! The install includes Anaconda Navigator, a desktop application that is built on top of conda. You can use Navigator’s graphical user interface (GUI) to create environments, install packages, and launch development applications like Jupyter Notebooks and Spyder. For more information on Navigator, see [Getting started with Navigator](https://www.anaconda.com/docs/tools/anaconda-navigator/getting-started).

I know exactly what packages I want to use and I don't want a large download:
Install Miniconda! Miniconda is a minimal installer that only includes a very small installation of Anaconda’s distribution—conda, Python, and a few other packages. For more information, see the [Miniconda documentation](https://www.anaconda.com/docs/getting-started/miniconda/main).

I only use the command line:
Install Miniconda _or_ Anaconda Distribution! Both installations include conda, the command line package and environment manager. For more information on conda, see the [conda Getting Started page](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) or download the [conda cheatsheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html).

## Should I use Anaconda Navigator or conda?
Anaconda Navigator is a desktop application that is included with every installation of Anaconda Distribution. It is built on top of conda, the open-source package and environment manager, and allows you to manage your packages and environments from a graphical user interface (GUI). This is especially helpful when you’re not comfortable with the command line.

Anaconda Navigator is not included with Miniconda. Use the command `conda install anaconda-navigator` to manually install Navigator onto your computer.

A command line interface (or CLI) is a program on your computer that processes text commands to do various tasks. Conda is a CLI program, which means it can only be used via the command line. On Windows computers, Anaconda recommends that you use the Anaconda Prompt CLI to work with conda. On macOS and Linux, users can use their built-in command line applications.

## Free Anaconda Learning course
In the **Get Started with Anaconda** entry-level course, you’ll learn about packages, conda environments, Jupyter Notebooks, and more. We’ll also guide you through initiating a Python program (in both a notebook and several popular IDEs) and show you what happens when a bug is identified in Python code.

[**Get Started with Anaconda**](https://learning.anaconda.com/courses/get-started-with-anaconda?utm_campaign=learning&utm_medium=documentation&utm_source=anacondadocs&utm_content=getstartedbutton)

## Anaconda Notebooks
[Anaconda Notebooks](https://www.anaconda.com/docs/tools/anaconda-notebooks/main) allow anyone, anywhere to begin their data science journey. Spin up awesome data science projects directly from your browser with all the packages and computing power you need.

[**Launch Anaconda Notebooks**](https://nb.anaconda.com/)

## IDE tutorials
The following tutorials show you the basics of using some popular IDEs (integrated development environments) with Anaconda:

- [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) 
- [PyCharm](https://www.anaconda.com/docs/tools/working-with-conda/ide-tutorials/pycharm)
- [Visual Studio Code (VS Code)](https://www.anaconda.com/docs/tools/working-with-conda/ide-tutorials/vscode)
- [Spyder](https://www.anaconda.com/docs/tools/working-with-conda/ide-tutorials/spyder)

>  Summary:
>  `conda` is a open source package and environment manager
>  Anaconda and Miniconda are just different installers of `conda`. They both comes with `conda`, `python` , their dependencies and some other packages.

