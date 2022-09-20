# KCR



## 简介

`KCR`是一个知识概念推荐模型

RecBole平台

[中文主页](https://recbole.io/cn)|[文档](https://recbole.io/docs/)|[数据集](https://github.com/RUCAIBox/RecDatasets)|[论文](https://arxiv.org/abs/2011.01731)|[博客](https://blog.csdn.net/Turinger_2000/article/details/111182852)



### 环境依赖

`pip install -r requirements.txt`

### 文件结构

`【】`表示对RecBole原文件的改动，共4处

``` lua
recBole-master-- 源码目录
├── asset -- 静态资源
    ├── time_test_result -- 模型的运行时间和内存占用情况
    		├── Context-aware_recommendation.md -- 上下文感知的推荐
    		├── General_recommendation.md -- 通用推荐
				├── Knowledge-based_recommendation.md -- 基于知识的推荐 
    		└── Sequential_recommendation.md -- 序列推荐
    ├── framework.png -- 框架
    └── logo.png -- recbole logo
├── conda -- 
    ├── build.sh -- 
    ├── conda_ralease.sh -- 
    └── meta.yaml -- 
├── dataset -- 数据集存放位置
    └── ml-100k -- ml-100k数据集
    └── kcr-3w					-- 【①新增数据集文件】
    		├── kcr-3w.inter	-- 交互
    		├── kcr-3w.link	-- 映射
    		└── kcr-3w.kg 	-- 知识图谱
├── docs -- 相关文本
    ├── source -- 
    └── Makefile -- 
├── log -- 模型运行的关键日志信息
    ├── LightGCN -- 
    └── KCR -- 
├── recbole -- *recbole模型*
    ├── config -- 加载定义参数的配置器模块
    ├── data -- 数据
    ├── dataset_example -- 数据集示例
    ├── evaluator -- 评估
    ├── model -- 模型      		
    		├── context_aware_recommender -- 
    		├── exlib_recommender -- 
    		├── general_recommender -- 
    				└── GCARec.py -- 【②新增模型文件】 
    		├── knowledge_aware_recommender -- 
    		├── sequential_recommender -- 
    		├── __init__.py -- 
    		├── abstract_recommender.py -- 
    		├── init.py -- 
    		├── layers.py -- 
    		└── loss.py -- 
    ├── properties -- 参数设置
				├── dataset -- 
    		├── model -- 
    				└── GCARec.yaml -- 【③新增模型参数配置文件】
    		├── quick_start_config -- 
    		└── overall.yaml -- 【④文件内新增评价指标参数配置metrics: 
    		--["Recall","MRR","NDCG","Hit","Precision","GAUC","ShannonEntropy","ItemCoverage","AveragePopularity","GiniIndex", "TailPercentage"]】
    ├── quick_start -- 运行时需要导入的run_recbole方法：
    -- config->dataset filtering->dataset splitting->model->trainer->training->evaluation
    ├── sampler -- 采样
    ├── trainer -- 训练
    ├── utils -- 工具
    		├── __init__.py -- 
    		├── argument_list.py -- 
    		├── case_study.py -- 
    		├── enum_type -- 
    		├── logger.py -- 
    		├── url.py -- 
    		├── utils.py -- 
    		└── wandblogger.py -- 
    └── __init__.py -- 
├── run_example -- 示例代码
    ├── case_study_example.py -- 案例研究（对模型算法和用户的深入研究）代码
    ├── save_and_load_example.py -- 保存和加载模型代码
    └── session_based_rec_example.py -- 基于会话的推荐
├── saved 保存的模型
├── tests -- 
    ├── config -- 
    ├── data -- 
    ├── evaluation_setting -- 
    ├── metrics -- 
    ├── model -- 
    └── text_data -- 
├── venv -- 
    ├── bin -- 
    ├── include -- 
    └── lib -- 
├── hyper.test -- 
├── LICNSE -- 
├── MANIFEST.in -- 
├── README.md -- readme文件
├── README_CN.md -- readme中文文件
├── requirements.txt -- 
├── run_hyper.py -- 
├── run_recbole.py -- 源码运行入口
├── run_test.sh -- 
├── setup.py -- 
└── style.cfg -- 
```

## 快速上手
运行（可在控制台和log文件夹下查看日志信息）：  
`python run_recbole.py --model=[model_name] --dataset=[dataset_name]`  
后台不挂断即时打印日志：  
`nohup python -u run_recbole.py --model=[model_name] --dataset=[dataset_name] >[dataset_name]-[model_name]-time.log 2>&1 &`