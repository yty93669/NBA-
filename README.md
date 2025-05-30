NBA 球员检索系统

这是一个易于使用的检索系统，支持通过文本或图像方式搜索 NBA 球员信息。系统集成了网页爬虫、数据预处理、特征提取以及相似度匹配等技术，能快速准确地返回检索结果。

主要功能：
双模式检索：支持文本搜索（球员姓名 / 号码）和图像搜索（上传球员照片）
数据采集：自动从 NBA 官方网站爬取球员信息和图片
图像处理：使用 HOG 特征进行图像相似度分析
文本处理：基于 TF-IDF 算法实现文本检索
Top-10 结果：根据相似度评分返回最相关的 10 个匹配结果

下载 ChromeDriver：
确保您的系统已安装 Chrome 浏览器
从此处下载对应版本的 ChromeDriver 并添加到系统路径

使用说明
1. 数据爬取（可选）
如需更新球员数据库，执行：
bash
python crawler.py
系统将自动抓取最新球员数据和图片到data/目录

3. 启动检索系统
bash
python gui.py
系统将打开图形界面，您可以：
在搜索框中输入文本（球员姓名）
点击 "选择图像" 上传球员照片
点击 "搜索" 按钮获取检索结果
项目结构

plaintext
nba-player-search/
├── crawler/           # 数据爬取模块
├── preprocessor/      # 数据预处理模块
├── feature_extractor/ # 特征提取模块
├── search_engine/     # 检索引擎
├── gui/               # 图形界面
├── data/              # 存储爬取的球员数据
├── images/            # 存储球员图片
├── crawler.py         # 爬虫主程序
├── main.py            # 系统入口
└── requirements.txt   # 依赖列表
