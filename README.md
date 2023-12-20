# freshman_project

大一立项

## 部署方法：

分为几个步骤：

1. 导入模型与资源

在 release 中下载 static 与 model，将解压后的文件夹，放入到本项目中

2. 环境说明

本机环境：

anaconda，缺啥安装啥，推荐使用 pyenv，然后使用 anaconda 的 base

本人之前使用 miniforge，在使用 fengshen 的 pegasus 模型的时候出现一个 vocab 的错误。

3. 导入到数据库

先尝试能不能成功运行`flask run`，打开页面，可以看到是没有内容的

接下来创建一个 notice 数据库，可以用 navigate 或者原生的 sql 语法创建，
具体情查询相应的 manual，
将内容导入数据库，使用 utils/data/csv_into_db.ipynb

然后再在 config.py 中设置与数据库链接的配置信息，测试连通本地的数据库

4. 启动项目

最后再次运行`flask run`
