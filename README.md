# Deep-Learning-Based-Anomaly-Detection

***Anomaly Detection***: The process of detectingdata instances that ***significantly deviate*** from the majority of the whole dataset.

Contributed by Chunyang Zhang.

## [Content](#content)
<table>
<tr><td colspan="2"><a href="#survey">1. Survey</a></td></tr> 
<tr><td colspan="2"><a href="#methodology">2. Methodology</a></td></tr>
<tr>
    <td>&ensp;<a href="#autoencoder">2.1 AutoEncoder</a></td>
    <td>&ensp;<a href="#gan">2.2 GAN</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#flow">2.3 Flow</a></td>
    <td>&ensp;<a href="#diffusion-model">2.4 Diffusion Model</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#transformer">2.5 Transformer</a></td>
    <td>&ensp;<a href="#convolution">2.6 Convolution</a></td>
</tr> 
<tr>
    <td>&ensp;<a href="#gnn">2.7 GNN</a></td>
    <td>&ensp;<a href="#time-series">2.8 Time Series</a></td>
</tr>
<tr> 
    <td>&ensp;<a href="#tabular">2.9 Tabular</a></td>
    <td>&ensp;<a href="#out-of-distribution">2.10 Out of Distribution</a></td>
</tr>
<tr> 
    <td>&ensp;<a href="#large-model">2.11 Large Model</a></td>
    <td>&ensp;<a href="#reinforcement-learning">2.12 Reinforcement Learning</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#in-context-learning">2.13 In-Context Learning</a></td>
    <td>&ensp;<a href="#representation-learning">2.14 Representation Learning</a></td>
</tr>
<tr><td colspan="2"><a href="#mechanism">3. Mechanism</a></td></tr>
<tr>
    <td>&ensp;<a href="#dataset">3.1 Dataset</a></td>
    <td>&ensp;<a href="#library">3.2 Library</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#analysis">3.3 Analysis</a></td>
    <td>&ensp;<a href="#domain-adaptation">3.4 Domain Adaptation</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#loss-function">3.5 Loss Function</a></td>
    <td>&ensp;<a href="#model-selection">3.6 Model Selection</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#knowledge-distillation">3.7 Knowledge Distillation</a></td>
    <td>&ensp;<a href="#data-augmentation">3.8 Data Augmentation</a></td>
</tr>
<tr>
	<td>&ensp;<a href="#outlier-exposure">3.9 Outlier Exposure</a></td>
    <td>&ensp;<a href="#contrastive-learning">3.10 Contrastive Learning</a></td>
</tr>
<tr> 
    <td>&ensp;<a href="#continual-learning">3.11 Continual Learning</a></td>
    <td>&ensp;<a href="#active-learning">3.12 Active Learning</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#statistics">3.13 Statistics</a></td>
    <td>&ensp;<a href="#density-estimation">3.14 Density Estimation</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#support-vector">3.15 Support Vector</a></td>
    <td>&ensp;<a href="#sparse-coding">3.16 Sparse Coding</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#energy-model">3.17 Energy Model</a></td>
    <td>&ensp;<a href="#memory-bank">3.18 Memory Bank</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#cluster">3.19 Cluster</a></td>
    <td>&ensp;<a href="#isolation">3.20 Isolation</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#multi-modal">3.21 Multi Modal</a></td>
    <td>&ensp;<a href="#optimal-transport">3.22 Optimal Transport</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#causal-inference">3.23 Causal Inference</a></td>
    <td>&ensp;<a href="#gaussian-process">3.24 Gaussian Process</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#multi-task">3.25 Multi Task</a></td>
    <td>&ensp;<a href="#interpretability">3.26 Interpretability</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#neural-process">3.27 Neural Process</a></td>
    <td>&ensp;<a href="#nonparametric-approach">3.28 Nonparametric Approach</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#federated-learning">3.29 Federated Learning</a></td>
    <td>&ensp;<a href="#"></a></td>
</tr>
<tr><td colspan="2"><a href="#application">4. Application</a></td></tr>
<tr>
    <td>&ensp;<a href="#finance">4.1 Finance</a></td>
    <td>&ensp;<a href="#point-cloud">4.2 Point Cloud</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#autonomous-driving">4.3 Autonomous Driving</a></td>
    <td>&ensp;<a href="#medical-image">4.4 Medical Image</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#robotics">4.5 Robotics</a></td>
    <td>&ensp;<a href="#cyber-intrusion">4.6 Cyber Intrusion</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#diagnosis">4.7 Diagnosis</a></td>
    <td>&ensp;<a href="#high-performance-computing">4.8 High Performance Computing</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#physics">4.9 Physics</a></td>
    <td>&ensp;<a href="#industry-process">4.10 Industry Process</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#software">4.11 Software</a></td>
    <td>&ensp;<a href="#astronomy">4.12 Astronomy</a></td> 
</tr>
</table>





## [Survey](#content)
1. **A survey of single-scene video anomaly detection.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/document/9271895)

   *Bharathkumar Ramachandra, Michael J. Jones, and Ranga Raju Vatsavai.*

1. **Deep learning for anomaly detection: A review.** ACM Computing Surveys, 2022. [paper](https://dl.acm.org/doi/10.1145/3439950)

   *Guansong Pang, Chunhua Shen, Longbing Cao, and Anton Van Den Hengel.*

1. **A unifying review of deep and shallow anomaly detection.** Proceedings of the IEEE, 2020. [paper](https://ieeexplore.ieee.org/document/9347460)

   *Lukas Ruff, Jacob R. Kauffmann, Robert A. Vandermeulen, GrÉgoire Montavon, Wojciech Samek, Marius Kloft, Thomas G. Dietterich, and Klaus-robert MÜller.*

1. **A review on outlier/anomaly detection in time series data.** ACM Computing Surveys, 2022. [paper](https://dl.acm.org/doi/10.1145/3444690)

   *Ane Blázquez-García, Angel Conde, Usue Mori, and Jose A. Lozano.* 

1. **Anomaly detection in autonomous driving: A survey.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022W/WAD/html/Bogdoll_Anomaly_Detection_in_Autonomous_Driving_A_Survey_CVPRW_2022_paper.html)

   *Daniel Bogdoll, Maximilian Nitsche, and J. Marius Zöllner.* 

1. **A comprehensive survey on graph anomaly detection with deep learning.** TKDE, 2021. [paper](https://ieeexplore.ieee.org/document/9565320)

   *Xiaoxiao Ma, Jia Wu, Shan Xue, Jian Yang, Chuan Zhou, Quan Z. Sheng, and Hui Xiong, and Leman Akoglu.* 

1. **Transformers in time series: A survey.** arXiv, 2022. [paper](https://arxiv.org/abs/2202.07125)

   *Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun.*

1. **A survey on explainable anomaly detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2210.06959)

   *Zhong Li, Yuxuan Zhu, and Matthijs van Leeuwen.*

1. **Deep learning approaches for anomaly-based intrusion detection systems: A survey, taxonomy, and open issues.** KBS, 2020. [paper](https://www.sciencedirect.com/science/article/pii/S0950705119304897)

   *Arwa Aldweesh, Abdelouahid Derhab, and Ahmed Z.Emam.* 

1. **Deep learning-based anomaly detection in cyber-physical systems: Progress and oportunities.** ACM Computing Surveys, 2022. [paper](https://dl.acm.org/doi/10.1145/3453155)

   *Yuan Luo, Ya Xiao, Long Cheng, Guojun Peng, and Danfeng (Daphne) Yao.* 

1. **GAN-based anomaly detection: A review.** Neurocomputing, 2022. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231221019482)

   *Xuan Xia, Xizhou Pan, Nan Lia, Xing He, Lin Ma, Xiaoguang Zhang, and Ning Ding.* 

1. **Unsupervised anomaly detection in time-series: An extensive evaluation and analysis of state-of-the-art methods.** arXiv, 2022. [paper](https://arxiv.org/abs/2212.03637)

   *Nesryne Mejri, Laura Lopez-Fuentes, Kankana Roy, Pavel Chernakov, Enjie Ghorbel, and Djamila Aouada.* 

1. **Deep learning for time series anomaly detection: A survey.** arXiv, 2022. [paper](https://arxiv.org/abs/2211.05244)

   *Zahra Zamanzadeh Darban, Geoffrey I. Webb, Shirui Pan, Charu C. Aggarwal, and Mahsa Salehi.* 

1. **A survey of deep learning-based network anomaly detection.** Cluster Computing, 2019. [paper](https://link.springer.com/article/10.1007/s10586-017-1117-8)

   *Donghwoon Kwon, Hyunjoo Kim, Jinoh Kim, Sang C. Suh, Ikkyun Kim, and Kuinam J. Kim.* 

1. **Survey on anomaly detection using data mining techniques.** Procedia Computer Science, 2015. [paper](https://www.sciencedirect.com/science/article/pii/S1877050915023479)

   *Shikha Agrawal and Jitendra Agrawal.* 

1. **Graph based anomaly detection and description: A survey.** Data Mining and Knowledge Discovery, 2015. [paper](https://link.springer.com/article/10.1007/s10618-014-0365-y)

   *Leman Akoglu, Hanghang Tong, and Danai Koutra.* 

1. **Domain anomaly detection in machine perception: A system architecture and taxonomy.** TPAMI, 2014. [paper](https://ieeexplore.ieee.org/document/6636290)

   *Josef Kittler, William Christmas, Teófilo de Campos, David Windridge, Fei Yan, John Illingworth, and Magda Osman.* 

1. **Graph-based time-series anomaly detection: A Survey.** arXiv, 2023. [paper](https://arxiv.org/abs/2302.00058)

   *Thi Kieu Khanh Ho, Ali Karami, and Narges Armanfard.* 

1. **Weakly supervised anomaly detection: A survey.** arXiv, 2023. [paper](https://arxiv.org/abs/2302.04549)

   *Minqi Jiang, Chaochuan Hou, Ao Zheng, Xiyang Hu, Songqiao Han, Hailiang Huang, Xiangnan He, Philip S. Yu, and Yue Zhao.* 

1. **A comprehensive survey of deep transfer learning for anomaly detection in industrial time series: Methods, applications, and directions.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.05638)

   *Peng Yan, Ahmed Abdulkadir, Matthias Rosenthal, Gerrit A. Schatte, Benjamin F. Grewe, and Thilo Stadelmann.* 

1. **Survey on video anomaly detection in dynamic scenes with moving cameras.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.07050)

   *Runyu Jiao, Yi Wan, Fabio Poiesi, and Yiming Wang.* 

1. **Physics-informed machine learning for data anomaly detection, classification, localization, and mitigation: A review, challenges, and path forward.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.10788)

   *Mehdi Jabbari Zideh, Paroma Chatterjee, and Anurag K. Srivastava.* 

1. **Detecting and learning out-of-distribution data in the open world: Algorithm and theory.** Thesis, 2023. [Ph.D.](https://arxiv.org/abs/2310.06221)

   *Yiyou Sun.* 

1. **Meta-survey on outlier and anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.07101)

   *Madalina Olteanu, Fabrice Rossi, and Florian Yger.* 

1. **Anomaly detection in surveillance videos: A thematic taxonomy of deep models, review and performance analysis.** Artificial Intelligence Review, 2023. [paper](https://link.springer.com/article/10.1007/s10462-022-10258-6)

   *S. Chandrakala, K. Deepak, and G. Revathy.* 

1. **Revisiting VAE for unsupervised time series anomaly detection: A frequency perspective.** WWW, 2024. [paper](https://arxiv.org/abs/2402.02820)

   *Zexin Wang, Changhua Pei, Minghua Ma, Xin Wang, Zhihan Li, Dan Pei, Saravan Rajmohan, Dongmei Zhang, Qingwei Lin, Haiming Zhang, Jianhui Li, and Gaogang Xie.* 

1. **Can tree based approaches surpass deep learning in anomaly detection? A benchmarking study.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.07281)

   *Santonu Sarkar, Shanay Mehta, Nicole Fernandes, Jyotirmoy Sarkar, and Snehanshu Saha.* 

1. **Large language models for forecasting and anomaly detection: A systematic literature review.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.10350)

   *Jing Su, Chufeng Jiang, Xin Jin, Yuxin Qiao, Tingsong Xiao, Hongda Ma, Rong Wei, Zhi Jing, Jiajun Xu, and Junhong Lin.* 

1. **A survey of graph neural networks in real world: Imbalance, noise, privacy and OOD challenges.** arXiv, 2024. [paper](https://arxiv.org/abs/2403.04468)

   *Wei Ju, Siyu Yi, Yifan Wang, Zhiping Xiao, Zhengyang Mao, Hourun Li, Yiyang Gu, Yifang Qin, Nan Yin, Senzhang Wang, Xinwang Liu, Xiao Luo, Philip S. Yu, and Ming Zhang.* 

1. **Anomaly detection in graph structured data: A survey.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.06172)

   *Prabin B Lamichhane and William Eberle.* 


## [Methodology](#content) 
### [AutoEncoder](#content)
1. **Graph regularized autoencoder and its application in unsupervised anomaly detection.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/document/9380495)

   *Imtiaz Ahmed, Travis Galoppo, Xia Hu, and Yu Ding.* 

1. **Innovations autoencoder and its application in one-class anomalous sequence detection.** JMLR, 2022. [paper](https://www.jmlr.org/papers/volume23/21-0735/21-0735.pdf)

   *Xinyi Wang and Lang Tong.* 

1. **Autoencoders-A comparative analysis in the realm of anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022W/WiCV/html/Schneider_Autoencoders_-_A_Comparative_Analysis_in_the_Realm_of_Anomaly_CVPRW_2022_paper.html)

   *Sarah Schneider, Doris Antensteiner, Daniel Soukup, and Matthias Scheutz.* 

1. **Attention guided anomaly localization in images.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-58520-4_29)

   *Shashanka Venkataramanan, Kuan-Chuan Peng, Rajat Vikram Singh, and Abhijit Mahalanobis.* 

1. **Latent space autoregression for novelty detection.** CVPR, 2018. [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Abati_Latent_Space_Autoregression_for_Novelty_Detection_CVPR_2019_paper.html)

   *Davide Abati, Angelo Porrello, Simone Calderara, and Rita Cucchiara.*

1. **Anomaly detection in time series with robust variational quasi-recurrent autoencoders.** ICDM, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/9835268)

   *Tung Kieu, Bin Yang, Chenjuan Guo, Razvan-Gabriel Cirstea, Yan Zhao, Yale Song, and Christian S. Jensen.*

1. **Robust and explainable autoencoders for unsupervised time series outlier detection.** ICDE, 2022. [paper](https://ieeexplore.ieee.org/document/9835554)

   *Tung Kieu, Bin Yang, Chenjuan Guo, Christian S. Jensen, Yan Zhao, Feiteng Huang, and Kai Zheng.*

1. **Latent feature learning via autoencoder training for automatic classification configuration recommendation.** KBS, 2022. [paper](https://www.sciencedirect.com/science/article/pii/S0950705122013144)

   *Liping Deng and Mingqing Xiao.*

1. **Deep autoencoding Gaussian mixture model for unsupervised anomaly detection.** ICLR, 2018. [paper](https://openreview.net/forum?id=BJJLHbb0-)

   *Bo Zongy, Qi Songz, Martin Renqiang Miny, Wei Chengy, Cristian Lumezanuy, Daeki Choy, and Haifeng Chen.* 

1. **Anomaly detection with robust deep autoencoders.** KDD, 2017. [paper](https://dl.acm.org/doi/10.1145/3097983.3098052)

   *Chong Zhou and Randy C. Paffenroth.* 

1. **Unsupervised anomaly detection via variational auto-encoder for seasonal KPIs in web applications.** WWW, 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3178876.3185996)

   *Haowen Xu, Wenxiao Chen, Nengwen Zhao,Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan Pei, Yang Feng, Jie Chen, Zhaogang Wang, and Honglin Qiao.* 

1. **Spatio-temporal autoencoder for video anomaly detection.** MM, 2017. [paper](https://dl.acm.org/doi/abs/10.1145/3123266.3123451)

   *Yiru Zhao, Bing Deng, Chen Shen, Yao Liu, Hongtao Lu, and Xiansheng Hua.* 

1. **Learning discriminative reconstructions for unsupervised outlier removal.** ICCV, 2015. [paper](https://ieeexplore.ieee.org/document/7410534)

   *Yan Xia, Xudong Cao, Fang Wen, Gang Hua, and Jian Sun.* 

1. **Outlier detection with autoencoder ensembles.** ICDM, 2017. [paper](https://research.ibm.com/publications/outlier-detection-with-autoencoder-ensembles)

   *Jinghui Chen, Saket Sathey, Charu Aggarwaly, and Deepak Turaga.*

1. **A study of deep convolutional auto-encoders for anomaly detection in videos.** Pattern Recognition Letters, 2018. [paper](https://www.sciencedirect.com/science/article/pii/S0167865517302489)

   *Manassés Ribeiro, AndréEugênio Lazzaretti, and Heitor Silvério Lopes.*

1. **Classification-reconstruction learning for open-set recognition.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoshihashi_Classification-Reconstruction_Learning_for_Open-Set_Recognition_CVPR_2019_paper.pdf)

   *Ryota Yoshihashi, Shaodi You, Wen Shao, Makoto Iida, Rei Kawakami, and Takeshi Naemura.*

1. **Making reconstruction-based method great again for video anomaly detection.** ICDM, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/10027694/)

   *Yizhou Wang, Can Qin, Yue Bai, Yi Xu, Xu Ma, and Yun Fu.*

1. **Two-stream decoder feature normality estimating network for industrial snomaly fetection.** ICASSP, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10027694/)

   *Chaewon Park, Minhyeok Lee, Suhwan Cho, Donghyeong Kim, and Sangyoun Lee.*

1. **Synthetic pseudo anomalies for unsupervised video anomaly detection: A simple yet efficient framework based on masked autoencoder.** ICASSP, 2023. [paper](https://arxiv.org/abs/2303.05112)

   *Xiangyu Huang, Caidan Zhao, Chenxing Gao, Lvdong Chen, and Zhiqiang Wu.*

1. **Deep autoencoding one-class time series anomaly detection.** ICASSP, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10095724)

   *Xudong Mou, Rui Wang, Tiejun Wang, Jie Sun, Bo Li, Tianyu Wo, and Xudong Liu.*

1. **Reconstruction error-based anomaly detection with few outlying examples.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.10464)

   *Fabrizio Angiulli, Fabio Fassetti, and Luca Ferragina.*

1. **LARA: A light and anti-overfitting retraining approach for unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.05668)

   *Feiyi Chen, Zhen Qing, Yingying Zhang, Shuiguang Deng, Yi Xiao, Guansong Pang, and Qingsong Wen.*

1. **FMM-Head: Enhancing autoencoder-based ECG anomaly detection with prior knowledge.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.05848)

   *Giacomo Verardo, Magnus Boman, Samuel Bruchfeld, Marco Chiesa, Sabine Koch, Gerald Q. Maguire Jr., and Dejan Kostic.*

1. **Online multi-view anomaly detection with disentangled product-of-experts modeling.** MM, 2023. [paper](https://arxiv.org/abs/2310.18728)

   *Hao Wang, Zhiqi Cheng, Jingdong Sun, Xin Yang, Xiao Wu, Hongyang Chen, and Yan Yang.*

1. **Fast particle-based anomaly detection algorithm with variational autoencoder.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.17162)

   *Ryan Liu, Abhijith Gandrakota, Jennifer Ngadiuba, Maria Spiropulu, and Jean-Roch Vlimant.*

1. **Dynamic erasing network based on multi-scale temporal features for weakly supervised video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01764)

   *Chen Zhang, Guorong Li, Yuankai Qi, Hanhua Ye, Laiyun Qing, Ming-Hsuan Yang, and Qingming Huang.*

1. **ACVAE: A novel self-adversarial variational auto-encoder combined with contrast learning for time series anomaly detection.** Neural Networks, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023007281)

   *Xiaoxia Zhang, Shang Shi, HaiChao Sun, Degang Chen, Guoyin Wang, and Kesheng Wu.*

1. **Dual-constraint autoencoder and adaptive weighted similarity spatial attention for unsupervised anomaly detection.** TII, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10504620)

   *Ruifan Zhang, Hao Wang, Mingyao Feng, Yikun Liu, and Gongping Yang.*

### [GAN](#content)
1. **Stabilizing adversarially learned one-class novelty detection using pseudo anomalies.** TIP, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9887825)

   *Muhammad Zaigham Zaheer, Jin-Ha Lee, Arif Mahmood, Marcella Astri, and Seung-Ik Lee.* 

1. **GAN ensemble for anomaly detection.** AAAI, 2021. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16530)

   *Han, Xu, Xiaohui Chen, and Liping Liu.* 

1. **Generative cooperative learning for unsupervised video anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zaheer_Generative_Cooperative_Learning_for_Unsupervised_Video_Anomaly_Detection_CVPR_2022_paper.html)

   *Zaigham Zaheer, Arif Mahmood, M. Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee.* 

1. **GAN-based anomaly detection in imbalance problems.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-65414-6_11)

   *Junbong Kim, Kwanghee Jeong, Hyomin Choi, and Kisung Seo.* 

1. **Old is gold: Redefining the adversarially learned one-class classifier training paradigm.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.html)

   *Muhammad Zaigham Zaheer, Jin-ha Lee, Marcella Astrid, and Seung-Ik Lee.* 

1. **Unsupervised anomaly detection with generative adversarial networks to guide marker discovery.** IPMI, 2017. [paper](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_12)

   *Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, and Georg Langs.* 

1. **Adversarially learned anomaly detection.** ICDM, 2018. [paper](https://ieeexplore.ieee.org/document/8594897)

   *Houssam Zenati, Manon Romain, Chuan-Sheng Foo, Bruno Lecouat, and Vijay Chandrasekhar.* 

1. **BeatGAN: Anomalous rhythm detection using adversarially generated time series.** IJCAI, 2019. [paper](https://www.ijcai.org/proceedings/2019/616)

   *Bin Zhou, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye.* 

1. **Convolutional transformer based dual discriminator generative adversarial networks for video anomaly detection.** MM, 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475693)

   *Xinyang Feng, Dongjin Song, Yuncong Chen, Zhengzhang Chen, Jingchao Ni, and Haifeng Chen.* 

1. **USAD: Unsupervised anomaly detection on multivariate time series.** KDD, 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403392)

   *Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A. Zuluaga.* 

1. **Anomaly detection with generative adversarial networks for multivariate time series.** ICLR, 2018. [paper](https://arxiv.org/abs/1809.04758)

   *Dan Li, Dacheng Chen, Jonathan Goh, and See-kiong Ng.* 

1. **Efficient GAN-based anomaly detection.** ICLR, 2018. [paper](https://arxiv.org/abs/1802.06222)

   *Houssam Zenati, Chuan Sheng Foo, Bruno Lecouat, Gaurav Manek, and Vijay Ramaseshan Chandrasekhar.* 

1. **GANomaly: Semi-supervised anomaly detection via adversarial training.** ACCV, 2019. [paper](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_39)

   *Akcay, Samet, Amir Atapour-Abarghouei, and Toby P. Breckon.* 

1. **f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks.** Medical Image Analysis, 2019. [paper](https://www.sciencedirect.com/science/article/pii/S1361841518302640)

   *Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Georg Langs, and Ursula Schmidt-Erfurth.* 

1. **OCGAN: One-class novelty detection using GANs with constrained latent representations.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Perera_OCGAN_One-Class_Novelty_Detection_Using_GANs_With_Constrained_Latent_Representations_CVPR_2019_paper.pdf)

   *Pramuditha Perera, Ramesh Nallapati, and Bing Xiang.* 

1. **Adversarially learned one-class classifier for novelty detection.** CVPR, 2018. [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.pdf)

   *Mohammad Sabokrou, Mohammad Khalooei, Mahmood Fathy, and Ehsan Adeli.* 

1. **Generative probabilistic novelty detection with adversarial autoencoders.** NIPS, 2018. [paper](https://dl.acm.org/doi/10.5555/3327757.3327787)

   *Stanislav Pidhorskyi, Ranya Almohsen, Donald A. Adjeroh, and Gianfranco Doretto.* 

1. **Image anomaly detection with generative adversarial networks.** ECML PKDD, 2018. [paper](https://link.springer.com/chapter/10.1007/978-3-030-10925-7_1)

   *Lucas Deecke, Robert Vandermeulen, Lukas Ruff, Stephan Mandt, and Marius Kloft.*

1. **RGI: Robust GAN-inversion for mask-free image inpainting and unsupervised pixel-wise anomaly detection.** ICLR, 2023. [paper](https://openreview.net/forum?id=1UbNwQC89a)

   *Shancong Mou, Xiaoyi Gu, Meng Cao, Haoping Bai, Ping Huang, Jiulong Shan, and Jianjun Shi.*

1. **Truncated affinity maximization: One-class homophily modeling for graph anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.00006)

   *Qiao Hezhe and Pang Guansong.*

1. **Anomaly detection under contaminated data with contamination-immune bidirectional GANs.** TKDE, 2024. [paper](https://www.computer.org/csdl/journal/tk/5555/01/10536641/1X9vdwpnhO8)

   *Qinliang Su, Bowen Tian, Hai Wan, and Jian Yin.*

### [Flow](#content)
1. **OneFlow: One-class flow for anomaly detection based on a minimal volume region.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9525256)

   *Lukasz Maziarka, Marek Smieja, Marcin Sendera, Lukasz Struski, Jacek Tabor, and Przemyslaw Spurek.* 

1. **Comprehensive regularization in a bi-directional predictive network for video anomaly detection.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/19898)

   *Chengwei Chen, Yuan Xie, Shaohui Lin, Angela Yao, Guannan Jiang, Wei Zhang, Yanyun Qu, Ruizhi Qiao, Bo Ren, and Lizhuang Ma.* 

1. **Future frame prediction network for video anomaly detection.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9622181/)

   *Weixin Luo, Wen Liu, Dongze Lian, and Shenghua Gao.* 

1. **Graph-augmented normalizing flows for anomaly detection of multiple time series.** ICLR, 2022. [paper](https://openreview.net/forum?id=45L_dgP48Vd)

   *Enyan Dai and Jie Chen.* 

1. **Cloze test helps: Effective video anomaly detection via learning to complete video events.** MM, 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413973)

   *Guang Yu, Siqi Wang, Zhiping Cai, En Zhu, Chuanfu Xu, Jianping Yin, and Marius Kloft.* 

1. **A modular and unified framework for detecting and localizing video anomalies.** WACV, 2022. [paper](https://openaccess.thecvf.com/content/WACV2022/html/Doshi_A_Modular_and_Unified_Framework_for_Detecting_and_Localizing_Video_WACV_2022_paper.html)

   *Keval Doshi and Yasin Yilmaz.*

1. **Video anomaly detection with compact feature sets for online performance.** TIP, 2017. [paper](https://ieeexplore.ieee.org/abstract/document/7903693)

   *Roberto Leyva, Victor Sanchez, and Chang-Tsun Li.*

1. **U-Flow: A U-shaped normalizing flow for anomaly detection with unsupervised threshold.** arXiv, 2017. [paper](https://arxiv.org/abs/2211.12353)

   *Matías Tailanian, Álvaro Pardo, and Pablo Musé.*

1. **Bi-directional frame interpolation for unsupervised video anomaly detection.** WACV, 2023. [paper](https://arxiv.org/abs/2211.12353)

   *Hanqiu Deng, Zhaoxiang Zhang, Shihao Zou, and Xingyu Li.*

1. **AE-FLOW: Autoencoders with normalizing flows for medical images anomaly detection.** ICLR, 2023. [paper](https://openreview.net/forum?id=9OmCr1q54Z)

   *Yuzhong Zhao, Qiaoqiao Ding, and Xiaoqun Zhang.*

1. **A video anomaly detection framework based on appearance-motion semantics representation consistency.** ICASSP, 2023. [paper](https://arxiv.org/abs/2303.05109)

   *Xiangyu Huang, Caidan Zhao, and Zhiqiang Wu.*

1. **Fully convolutional cross-scale-flows for image-based defect detection.** WACV, 2022. [paper](https://openaccess.thecvf.com/content/WACV2022/html/Rudolph_Fully_Convolutional_Cross-Scale-Flows_for_Image-Based_Defect_Detection_WACV_2022_paper.html)

   *Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt.*

1. **CFLOW-AD: Real-time unsupervised anomaly detection with localization via conditional normalizing flows.** WACV, 2022. [paper](https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html)

   *Denis Gudovskiy, Shun Ishizaka, and Kazuki Kozuka.*

1. **Same same but DifferNet: Semi-supervised defect detection with normalizing flows.** WACV, 2021. [paper](https://openaccess.thecvf.com/content/WACV2021/html/Rudolph_Same_Same_but_DifferNet_Semi-Supervised_Defect_Detection_With_Normalizing_Flows_WACV_2021_paper.html)

   *Marco Rudolph, Bastian Wandt, and Bodo Rosenhahn.*

1. **Normalizing flow based feature synthesis for outlier-aware object detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2302.07106v2)

   *Nishant Kumar, Siniša Šegvić, Abouzar Eslami, and Stefan Gumhold.*

1. **DyAnNet: A scene dynamicity guided self-trained video anomaly detection network.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Thakare_DyAnNet_A_Scene_Dynamicity_Guided_Self-Trained_Video_Anomaly_Detection_Network_WACV_2023_paper.html)

   *Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim.*

1. **Multi-scale spatial-temporal interaction network for video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.10239)

   *Zhiyuan Ning, Zhangxun Li, and Liang Song.*

1. **MSFlow: Multi-scale flow-based framework for unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.15300)

   *Yixuan Zhou, Xing Xu, Jingkuan Song, Fumin Shen, and Hengtao Shen.*

1. **PyramidFlow: High-resolution defect contrastive localization using pyramid normalizing flow.** CVPR, 2023. [paper](https://ieeexplore.ieee.org/document/10204306)

   *Jiarui Lei, Xiaobo Hu, Yue Wang, and Dong Liu.*

1. **Topology-matching normalizing flows for out-of-distribution detection in robot learning.** CoRL, 2023. [paper](https://openreview.net/forum?id=BzjLaVvr955)

   *Jianxiang Feng, Jongseok Lee, Simon Geisler, Stephan Günnemann, and Rudolph Triebel.*

1. **Video anomaly detection via spatio-temporal pseudo-anomaly generation : A unified approach.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.16514)

   *Ayush K. Rai, Tarun Krishna, Feiyan Hu, Alexandru Drimbarean, Kevin McGuinness, Alan F. Smeaton, and Noel E. O'Connor.*

1. **Self-supervised normalizing flows for image anomaly detection and localization.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Chiu_Self-Supervised_Normalizing_Flows_for_Image_Anomaly_Detection_and_Localization_CVPRW_2023_paper.html)

   *Li-Ling Chiu and Shang-Hong Lai.*

1. **Normalizing flows for human pose anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html)

   *Or Hirschorn and Shai Avidan.*

1. **Hierarchical Gaussian mixture normalizing flow modeling for unified anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2403.13349v1)

   *Xincheng Yao, Ruoqi Li, Zefeng Qian, Lu Wang, and Chongyang Zhang.*

### [Diffusion Model](#content)
1. **AnoDDPM: Anomaly detection with denoising diffusion probabilistic models using simplex noise.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.html)

   *Julian Wyatt, Adam Leach, Sebastian M. Schmon, and Chris G. Willcocks.* 

1. **Diffusion models for medical anomaly detection.** MICCAI, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_4)

   *Julia Wolleb, Florentin Bieder, Robin Sandkühler, and Philippe C. Cattin.* 

1. **DiffusionAD: Denoising diffusion for anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2303.08730)

   *Hui Zhang, Zheng Wang, Zuxuan Wu, Yugang Jiang.* 

1. **Anomaly detection with conditioned denoising diffusion models.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.15956)

   *Arian Mousakhan, Thomas Brox, and Jawad Tayyub.* 

1. **Unsupervised out-of-distribution detection with diffusion inpainting.** ICML, 2023. [paper](https://openreview.net/forum?id=HiX1ybkFMl)

   *Zhenzhen Liu, Jin Peng Zhou, Yufan Wang, and Kilian Q. Weinberger.* 

1. **On diffusion modeling for anomaly detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=lR3rk7ysXz)

   *Victor Livernoche, Vineet Jain, Yashar Hezaveh, and Siamak Ravanbakhsh.* 

1. **Mask, stitch, and re-sample: Enhancing robustness and generalizability in anomaly detection through automatic diffusion models.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.19643)

   *Cosmin I. Bercea, Michael Neumayr, Daniel Rueckert, and Julia A. Schnabel.* 

1. **Unsupervised anomaly detection in medical images using masked diffusion model.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.19867)

   *Hasan Iqbal, Umar Khalid, Jing Hua, and Chen Chen.* 

1. **Unsupervised anomaly detection in medical images using masked diffusion model.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.19867)

   *Hasan Iqbal, Umar Khalid, Jing Hua, and Chen Chen.* 

1. **ImDiffusion: Imputed diffusion models for multivariate time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.00754)

   *Yuhang Chen, Chaoyun Zhang, Minghua Ma, Yudong Liu, Ruomeng Ding, Bowen Li, Shilin He, Saravan Rajmohan, Qingwei Lin, and Dongmei Zhang.* 

1. **Multimodal motion conditioned diffusion model for skeleton-based video anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Flaborea_Multimodal_Motion_Conditioned_Diffusion_Model_for_Skeleton-based_Video_Anomaly_Detection_ICCV_2023_paper.html)

   *Alessandro Flaborea, Luca Collorone, Guido Maria D’Amely di Melendugno, Stefano D’Arrigo, Bardh Prenkaj, and Fabio Galasso.*

1. **LafitE: Latent diffusion model with feature editing for unsupervised multi-class anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.08059)

   *Haonan Yin, Guanlong Jiao, Qianhui Wu, Borje F. Karlsson, Biqing Huang, and Chin Yew Lin.*

1. **Diffusion models for counterfactual generation and anomaly detection in brain images.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.02062)

   *Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, and Amos Storkey.*

1. **Imputation-based time-series anomaly detection with conditional weight-incremental diffusion models.** KDD, 2023. [paper](https://dl.acm.org/doi/10.1145/3580305.3599391)

   *Chunjing Xiao, Zehua Gou, Wenxin Tai, Kunpeng Zhang, and Fan Zhou.*

1. **MadSGM: Multivariate anomaly detection with score-based generative models.** CIKM, 2023. [paper](https://arxiv.org/abs/2308.15069)

   *Haksoo Lim, Sewon Park, Minjung Kim, Jaehoon Lee, Seonkyu Lim, and Noseong Park.*

1. **Modality cycles with masked conditional diffusion for unsupervised anomaly segmentation in MRI.** MICCAI, 2023. [paper](https://arxiv.org/abs/2308.16150)

   *Ziyun Liang, Harry Anthony, Felix Wagner, and Konstantinos Kamnitsas.*

1. **Controlled graph neural networks with denoising diffusion for anomaly detection.** Expert Systems with Applications, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423020353)

   *Xuan Li, Chunjing Xiao, Ziliang Feng, Shikang Pang, Wenxin Tai, and Fan Zhou.*

1. **Unsupervised surface anomaly detection with diffusion probabilistic model.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)

   *Matic Fučka, Vitjan Zavrtanik, and Danijel Skočaj.*

1. **Transfusion -- A transparency-based diffusion model for anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.09999)

   *Ziyun Liang, Harry Anthony, Felix Wagner, and Konstantinos Kamnitsas.*

1. **Unsupervised anomaly detection using aggregated normative diffusion.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01904)

   *Alexander Frotscher, Jaivardhan Kapoor, Thomas Wolfers, and Christian F. Baumgartner.*

1. **Adversarial denoising diffusion model for unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.04382)

   *Jongmin Yu, Hyeontaek Oh, and Jinhong Yang.*

1. **Guided reconstruction with conditioned diffusion models for unsupervised anomaly detection in brain MRIs.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.04215)

   *Finn Behrendt, Debayan Bhattacharya, Robin Mieling, Lennart Maack, Julia Krüger, Roland Opfer, and Alexander Schlaefer.*

1. **DiAD: A diffusion-based framework for multi-class anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.06607)

   *Haoyang He, Jiangning Zhang, Hongxu Chen, Xuhai Chen, Zhishan Li, Xu Chen, Yabiao Wang, Chengjie Wang, and Lei Xie.*

1. **Feature prediction diffusion model for video anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.html)

   *Cheng Yan, Shiyu Zhang, Yang Liu, Guansong Pang, and Wenjun Wang.*

1. **Removing anomalies as noises for industrial defect localization.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.html)

   *Fanbin Lu, Xufeng Yao, Chi-Wing Fu, and Jiaya Jia.*

1. **DATAELIXIR: Purifying poisoned dataset to mitigate backdoor attacks via diffusion models.** AAAI, 2024. [paper](https://arxiv.org/abs/2312.11057)

   *Jiachen Zhou, Peizhuo Lv, Yibing Lan, Guozhu Meng, Kai Chen, and Hualong Ma.*

1. **Controlled graph neural networks with denoising diffusion for anomaly detection.** Expert Systems with Applications, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423020353)

   *Xuan Li, Chunjing Xiao, Ziliang Feng, Shikang Pang, Wenxin Tai, and Fan Zhou.*

1. **D3AD: Dynamic denoising diffusion probabilistic model for anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.04463)

   *Justin Tebbe and Jawad Tayyub.*

1. **TauAD: MRI-free Tau anomaly detection in PET imaging via conditioned diffusion models.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.13199)

   *Lujia Zhong, Shuo Huang, Jiaxin Yue, Jianwei Zhang, Zhiwei Deng, Wenhao Chi, and Yonggang Shi.*

### [Transformer](#content)
1. **Video anomaly detection via prediction network with enhanced spatio-temporal memory exchange.** ICASSP, 2022. [paper](https://ieeexplore.ieee.org/document/9747376)

   *Guodong Shen, Yuqi Ouyang, and Victor Sanchez.* 

1. **TranAD: Deep transformer networks for anomaly detection in multivariate time series data.** VLDB, 2022. [paper](https://dl.acm.org/doi/abs/10.14778/3514061.3514067)

   *Shreshth Tuli, Giuliano Casale, and Nicholas R. Jennings.* 

1. **Pixel-level anomaly detection via uncertainty-aware prototypical transformer.** MM, 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548082)

   *Chao Huang, Chengliang Liu, Zheng Zhang, Zhihao Wu, Jie Wen, Qiuping Jiang, and Yong Xu.* 

1. **AddGraph: Anomaly detection in dynamic graph using attention-based temporal GCN.** IJCAI, 2019. [paper](https://www.ijcai.org/proceedings/2019/614)

   *Li Zheng, Zhenpeng Li, Jian Li, Zhao Li, and Jun Gao.* 

1. **Anomaly transformer: Time series anomaly detection with association discrepancy.** ICLR, 2022. [paper](https://openreview.net/pdf?id=LzQQ89U1qm_)

   *Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long.* 

1. **Constrained adaptive projection with pretrained features for anomaly detection.** IJCAI, 2022. [paper](https://www.ijcai.org/proceedings/2022/0286.pdf)

   *Xingtai Gui, Di Wu, Yang Chang, and Shicai Fan.* 

1. **Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20028)

   *Shuo Li, Fang Liu, and Licheng Jiao.* 

1. **Beyond outlier detection: Outlier interpretation by attention-guided triplet deviation network.** WWW, 2021. [paper](https://dl.acm.org/doi/10.1145/3442381.3449868)

   *Hongzuo Xu, Yijie Wang, Songlei Jian, Zhenyu Huang, Yongjun Wang, Ning Liu, and Fei Li.* 

1. **Framing algorithmic recourse for anomaly detection.** KDD, 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539344)

   *Debanjan Datta, Feng Chen, and Naren Ramakrishnan.* 

1. **Inpainting transformer for anomaly detection.** ICIAP, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-06430-2_33)

   *Jonathan Pirnay and Keng Chai.* 

1. **Self-supervised and interpretable anomaly detection using network transformers.** arXiv, 2022. [paper](https://arxiv.org/abs/2202.12997)

   *Daniel L. Marino, Chathurika S. Wickramasinghe, Craig Rieger, and Milos Manic.* 

1. **Anomaly detection in surveillance videos using transformer based attention model.** arXiv, 2022. [paper](https://arxiv.org/abs/2206.01524)

   *Kapil Deshpande, Narinder Singh Punn, Sanjay Kumar Sonbhadra, and Sonali Agarwal.* 

1. **Multi-contextual predictions with vision transformer for video anomaly detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2206.08568?context=cs)

   *Joo-Yeon Lee, Woo-Jeoung Nam, and Seong-Whan Lee.* 

1. **Transformer based models for unsupervised anomaly segmentation in brain MR images.** arXiv, 2022. [paper](https://arxiv.org/abs/2207.02059)

   *Ahmed Ghorbel, Ahmed Aldahdooh, Shadi Albarqouni, and Wassim Hamidouche.* 

1. **HaloAE: An HaloNet based local transformer auto-encoder for anomaly detection and localization.** arXiv, 2022. [paper](https://arxiv.org/abs/2208.03486)

   *E. Mathian, H. Liu, L. Fernandez-Cuesta, D. Samaras, M. Foll, and L. Chen.* 

1. **Generalizable industrial visual anomaly detection with self-induction vision transformer.** arXiv, 2022. [paper](https://arxiv.org/abs/2211.12311)

   *Haiming Yao and Xue Wang.* 

1. **VT-ADL: A vision transformer network for image anomaly detection and localization.** ISIE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9576231)

   *Pankaj Mishra, Riccardo Verk, Daniele Fornasier, Claudio Piciarelli, and Gian Luca Foresti.* 

1. **Video event restoration based on keyframes for video anomaly detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2304.05112)

   *Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu.* 

1. **AnomalyBERT: Self-supervised Transformer for time series anomaly detection using data degradation scheme.** ICLR, 2023. [paper](https://arxiv.org/abs/2305.04468)

   *Yungi Jeong, Eunseok Yang, Jung Hyun Ryu, Imseong Park, and Myungjoo Kang.* 

1. **HAN-CAD: Hierarchical attention network for context anomaly detection in multivariate time series.** WWW, 2023. [paper](https://link.springer.com/article/10.1007/s11280-023-01171-1)

   *Haicheng Tao, Jiawei Miao, Lin Zhao, Zhenyu Zhang, Shuming Feng, Shu Wang, and Jie Cao.* 

1. **DCdetector: Dual attention contrastive representation learning for time series anomaly detection.** KDD, 2023. [paper](https://arxiv.org/abs/2306.10347)

   *Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun.* 

1. **SelFormaly: Towards task-agnostic unified anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.12540)

   *Yujin Lee, Harin Lim, and Hyunsoo Yoon.* 

1. **MIM-OOD: Generative masked image modelling for out-of-distribution detection in medical images.** MICCAI, 2023. [paper](https://arxiv.org/abs/2307.14701)

   *Sergio Naval Marimont, Vasilis Siomos, and Giacomo Tarroni.* 

1. **Focus the discrepancy: Intra- and Inter-correlation learning for image anomaly detection.** ICCV, 2023. [paper](https://arxiv.org/abs/2308.02983)

   *Xincheng Yao, Ruoqi Li, Zefeng Qian, Yan Luo, and Chongyang Zhang.* 

1. **Sparse binary Transformers for multivariate time series modeling.** KDD, 2023. [paper](https://arxiv.org/abs/2308.04637)

   *Matt Gorbett, Hossein Shirazi, and Indrakshi Ray.* 

1. **ADFA: Attention-augmented differentiable top-k feature adaptation for unsupervised medical anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.15280)

   *Yiming Huang, Guole Liu, Yaoru Luo, and Ge Yang.* 

1. **Mask2Anomaly: Mask Transformer for universal open-set segmentation.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.04573)

   *Shyam Nandan Rai, Fabio Cermelli, Barbara Caputo, and Carlo Masone.* 

1. **Hierarchical vector quantized Transformer for multi-class unsupervised anomaly detection.** NIPS, 2023. [paper](https://openreview.net/forum?id=clJTNssgn6)

   *Ruiying Lu, YuJie Wu, Long Tian, Dongsheng Wang, Bo Chen, Xiyang Liu, and Ruimin Hu.* 

1. **Attention modules improve image-level anomaly detection for industrial inspection: A DifferNet case study.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.02747)

   *Andre Luiz Vieira e Silva, Francisco Simoes, Danny Kowerko2 Tobias Schlosser, Felipe Battisti, and Veronica Teichrieb.* 

1. **Exploring plain ViT reconstruction for multi-class unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.07495)

   *Jiangning Zhang, Xuhai Chen, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, Ming-Hsuan Yang, and Dacheng Tao.* 

1. **Self-supervised masked convolutional transformer block for anomaly detection.** TPAMI, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10273635)

   *Neelu Madan, Nicolae-Cătălin Ristea, Radu Tudor Ionescu, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B. Moeslund, and Mubarak Shah.* 

1. **Transformer-based multivariate time series anomaly detection using inter-variable attention mechanism.** KBS, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10273635)

   *Hyeongwon Kang and Pilsung Kang.* 

1. **Sub-adjacent Transformer: Improving time series anomaly detection with reconstruction error from sub-adjacent neighborhoods.** IJCAI, 2024. [paper](https://arxiv.org/abs/2404.18948)

   *Wenzhen Yue, Xianghua Ying, Ruohao Guo, DongDong Chen, Ji Shi, Bowei Xing, Yuqing Zhu, and Taiyan Chen.* 

1. **Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.14325)

   *Jia Guo, Shuai Lu, Weihang Zhang, and Huiqi Li.* 

1. **How to train your ViT for OOD Detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.17447)

   *Maximilian Mueller and Matthias Hein.* 

### [Convolution](#content)
1. **Self-supervised predictive convolutional attentive block for anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.html)

   *Nicolae-Catalin Ristea, Neelu Madan, Radu Tudor Ionescu, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B. Moeslund, and Mubarak Shah.* 

1. **Catching both gray and black swans: Open-set supervised anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.html)

   *Choubo Ding, Guansong Pang, and Chunhua Shen.* 

1. **Learning memory-guided normality for anomaly detection.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf)

   *Hyunjong Park, Jongyoun No, and Bumsub Ham.* 

1. **CutPaste: Self-supervised learning for anomaly detection and localization.** CVPR, 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf)

   *Chunliang Li, Kihyuk Sohn, Jinsung Yoon, and Tomas Pfister.* 

1. **Object-centric auto-encoders and dummy anomalies for abnormal event detection in video.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf)

   *Radu Tudor Ionescu, Fahad Shahbaz Khan, Mariana-Iuliana Georgescu, and Ling Shao.* 

1. **Mantra-Net: Manipulation tracing network for detection and localization of image forgeries with anomalous features.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_ManTra-Net_Manipulation_Tracing_Network_for_Detection_and_Localization_of_Image_CVPR_2019_paper.html)

   *Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan.* 

1. **Grad-CAM: Visual explanations from deep networks via gradient-based localization.** ICCV, 2017. [paper](https://ieeexplore.ieee.org/document/8237336)

   *Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra.* 

1. **A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data.** AAAI, 2019. [paper](https://dl.acm.org/doi/10.1609/aaai.v33i01.33011409)

   *Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, and Nitesh V. Chawla.* 

1. **Real-world anomaly detection in surveillance videos.** CVPR, 2018. [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf)

   *Waqas Sultani, Chen Chen, and Mubarak Shah.* 

1. **FastAno: Fast anomaly detection via spatio-temporal patch transformation.** WACV, 2022. [paper](https://openaccess.thecvf.com/content/WACV2022/papers/Park_FastAno_Fast_Anomaly_Detection_via_Spatio-Temporal_Patch_Transformation_WACV_2022_paper.pdf)

   *Chaewon Park, MyeongAh Cho, Minhyeok Lee, and Sangyoun Lee.* 

1. **Object class aware video anomaly detection through image translation.** CRV, 2022. [paper](https://www.computer.org/csdl/proceedings-article/crv/2022/977400a090/1GeCy7y5kgU)

   *Mohammad Baradaran and Robert Bergevin.* 

1. **Anomaly detection in video sequence with appearance-motion correspondence.** ICCV, 2019. [paper](https://ieeexplore.ieee.org/document/9009067)

   *Trong-Nguyen Nguyen and Jean Meunier.* 

1. **Joint detection and recounting of abnormal events by learning deep generic knowledge.** ICCV, 2017. [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Hinami_Joint_Detection_and_ICCV_2017_paper.html)

   *Ryota Hinami, Tao Mei, and Shin’ichi Satoh.* 

1. **Deep-cascade: Cascading 3D deep neural networks for fast anomaly detection and localization in crowded scenes.** TIP, 2017. [paper](https://ieeexplore.ieee.org/abstract/document/7858798)

   *Mohammad Sabokrou, Mohsen Fayyaz, Mahmood Fathy, and Reinhard Klette.* 

1. **Towards interpretable video anomaly detection.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.html)

   *Keval Doshi and Yasin Yilmaz.* 

1. **Lossy compression for robust unsupervised time-series anomaly detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2212.02303)

   *Christopher P. Ley and Jorge F. Silva.* 

1. **Learning second order local anomaly for general face forgery detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Fei_Learning_Second_Order_Local_Anomaly_for_General_Face_Forgery_Detection_CVPR_2022_paper.html)

   *Jianwei Fei, Yunshu Dai, Peipeng Yu, Tianrun Shen, Zhihua Xia, and Jian Weng.* 

### [GNN](#content)
1. **Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.html)

   *Jiaxing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H. Li, and Ge Li.* 

1. **Towards open set video anomaly detection.** ECCV, 2019. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19830-4_23)

   *Yuansheng Zhu, Wentao Bao, and Qi Yu.* 

1. **Decoupling representation learning and classification for GNN-based anomaly detection.** SIGIR, 2021. [paper](https://dl.acm.org/doi/10.1145/3404835.3462944)

   *Yanling Wan,, Jing Zhang, Shasha Guo, Hongzhi Yin, Cuiping Li, and Hong Chen.* 

1. **Crowd-level abnormal behavior detection via multi-scale motion consistency learning.** AAAI, 2023. [paper](https://arxiv.org/abs/2212.00535)

   *Linbo Luo, Yuanjing Li, Haiyan Yin, Shangwei Xie, Ruimin Hu, and Wentong Cai.* 

1. **Rethinking graph neural networks for anomaly detection.** ICML, 2022. [paper](https://proceedings.mlr.press/v162/tang22b/tang22b.pdf)

   *Jianheng Tang, Jiajin Li, Ziqi Gao, and Jia Li.* 

1. **Cross-domain graph anomaly detection via anomaly-aware contrastive alignment.** AAAI, 2023. [paper](https://arxiv.org/abs/2212.01096)

   *Qizhou Wang, Guansong Pang, Mahsa Salehi, Wray Buntine, and Christopher Leckie.* 

1. **A causal inference look at unsupervised video anomaly detection.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20053)

   *Xiangru Lin, Yuyang Chen, Guanbin Li, and Yizhou Yu.* 

1. **NetWalk: A flexible deep embedding approach for anomaly detection in dynamic networks.** KDD, 2018. [paper](https://dl.acm.org/doi/10.1145/3219819.3220024)

   *Wenchao Yu, Wei Cheng, Charu C. Aggarwal, Kai Zhang, Haifeng Chen, and Wei Wang.* 

1. **LUNAR: Unifying local outlier detection methods via graph neural networks.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20629)

   *Adam Goodge, Bryan Hooi, See-Kiong Ng, and Wee Siong Ng.* 

1. **Series2Graph: Graph-based subsequence anomaly detection for time series.** VLDB, 2022. [paper](https://dl.acm.org/doi/10.14778/3407790.3407792)

   *Paul Boniol and Themis Palpanas.* 

1. **Graph embedded pose clustering for anomaly detection.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Markovitz_Graph_Embedded_Pose_Clustering_for_Anomaly_Detection_CVPR_2020_paper.html)

   *Amir Markovitz, Gilad Sharir, Itamar Friedman, Lihi Zelnik-Manor, and Shai Avidan.* 

1. **Fast memory-efficient anomaly detection in streaming heterogeneous graphs.** KDD, 2016. [paper](https://dl.acm.org/doi/abs/10.1145/2939672.2939783)

   *Emaad Manzoor, Sadegh M. Milajerdi, and Leman Akoglu.*

1. **Raising the bar in graph-level anomaly detection.** IJCAI, 2022. [paper](https://www.ijcai.org/proceedings/2022/0305.pdf)

   *Chen Qiu, Marius Kloft, Stephan Mandt, and Maja Rudolph.*

1. **SpotLight: Detecting anomalies in streaming graphs.** KDD, 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220040)

   *Dhivya Eswaran, Christos Faloutsos, Sudipto Guha, and Nina Mishra.* 

1. **Graph anomaly detection via multi-scale contrastive learning networks with augmented view.** AAAI, 2023. [paper](https://arxiv.org/abs/2212.00535)

   *Jingcan Duan, Siwei Wang, Pei Zhang, En Zhu, Jingtao Hu, Hu Jin, Yue Liu, and Zhibin Dong.* 

1. **Counterfactual graph learning for anomaly detection on attributed networks.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10056298)

   *Chunjing Xiao, Xovee Xu, Yue Lei, Kunpeng Zhang, Siyuan Liu, and Fan Zhou.*

1. **Deep variational graph convolutional recurrent network for multivariate time series anomaly detection.** ICML, 2022. [paper](https://proceedings.mlr.press/v162/chen22x.html)

   *Wenchao Chen, Long Tian, Bo Chen, Liang Dai, Zhibin Duan, and Mingyuan Zhou.*

1. **SAD: Semi-supervised anomaly detection on dynamic graphs.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.13573)

   *Sheng Tian, Jihai Dong, Jintang Li, Wenlong Zhao, Xiaolong Xu, Baokun wang, Bowen Song, Changhua Meng, Tianyi Zhang, and Liang Chen.*

1. **Improving generalizability of graph anomaly detection models via data augmentation.** TKDE, 2023. [paper](https://arxiv.org/abs/2306.10534)

   *Shuang Zhou, Xiao Huang, Ninghao Liu, Huachi Zhou, Fu-Lai Chung, and Long-Kai Huang.*

1. **Anomaly detection in networks via score-based generative models.** ICML, 2023. [paper](https://arxiv.org/abs/2306.15324)

   *Dmitrii Gavrilev and Evgeny Burnaev.*

1. **Generated graph detection.** ICML, 2023. [paper](https://openreview.net/forum?id=OoTa4H6Bnz)

   *Yihan Ma, Zhikun Zhang, Ning Yu, Xinlei He, Michael Backes, Yun Shen, and Yang Zhang.*

1. **Graph-level anomaly detection via hierarchical memory networks.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.00755)

   *Chaoxi Niu, Guansong Pang, and Ling Chen.*

1. **CSCLog: A component subsequence correlation-aware log anomaly detection method.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.03359)

   *Ling Chen, Chaodu Song, Xu Wang, Dachao Fu, and Feifei Li.*

1. **A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.03759)

   *Ming Jin, Huan Yee Koh, Qingsong Wen, Daniele Zambon, Cesare Alippi, Geoffrey I. Webb, Irwin King, and Shirui Pan.*

1. **Correlation-aware spatial-temporal graph learning for multivariate time-series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.08390)

   *Yu Zheng, Huan Yee Koh, Ming Jin, Lianhua Chi, Khoa T. Phan, Shirui Pan, Yi-Ping Phoebe Chen, and Wei Xiang.*

1. **Graph anomaly detection at group level: A topology pattern enhanced unsupervised approach.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.01063)

   *Xing Ai, Jialong Zhou, Yulin Zhu, Gaolei Li, Tomasz P. Michalak, Xiapu Luo, and Kai Zhou.*

1. **HRGCN: Heterogeneous graph-level anomaly detection with hierarchical relation-augmented graph neural networks.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.14340)

   *Jiaxi Li, Guansong Pang, Ling Chen, Mohammad-Reza and Namazi-Rad.*

1. **Revisiting adversarial attacks on graph neural networks for graph classification.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10243054)

   *Xin Wang, Heng Chang, Beini Xie, Tian Bian, Shiji Zhou, Daixin Wang, Zhiqiang Zhang, and Wenwu Zhu.*

1. **Normality learning-based graph anomaly detection via multi-scale contrastive learning.** MM, 2023. [paper](https://arxiv.org/abs/2309.06034)

   *Jingcan Duan, Pei Zhang, Siwei Wang, Jingtao Hu, Hu Jin, Jiaxin Zhang, Haifang Zhou, and Haifang Zhou.*

1. **GLAD: Content-aware dynamic graphs for log anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.05953)

   *Yufei Li, Yanchi Liu, Haoyu Wang, Zhengzhang Chen, Wei Cheng, Yuncong Chen, Wenchao Yu, Haifeng Chen, and Cong Liu.*

1. **ARISE: Graph anomaly detection on attributed networks via substructure awareness.** TNNLS, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10258476)

   *Jingcan Duan, Bin Xiao, Siwei Wang, Haifang Zhou, and Xinwang Liu.*

1. **Rayleigh quotient graph neural networks for graph-level anomaly detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=4UIBysXjVq)

   *Xiangyu Dong, Xingyi Zhang, and Sibo Wang.*

1. **Self-discriminative modeling for anomalous graph detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.06261)

   *Jinyu Cai, Yunhe Zhang, and Jicong Fan.*

1. **CVTGAD: Simplified transformer with cross-view attention for unsupervised graph-level anomaly detection.** ECML PKDD, 2023. [paper](https://link.springer.com/chapter/10.1007/978-3-031-43412-9_11)

   *Jindong Li, Qianli Xing, Qi Wang, and Yi Chang.*

1. **PREM: A simple yet effective approach for node-level graph anomaly detection.** ICDM, 2023. [paper](https://arxiv.org/abs/2310.11676)

   *Junjun Pan, Yixin Liu, Yizhen Zheng, and Shirui Pan.*

1. **THGNN: An embedding-based model for anomaly detection in dynamic heterogeneous social networks.** CIKM, 2023. [paper](https://dl.acm.org/doi/10.1145/3583780.3615079)

   *Yilin Li, Jiaqi Zhu, Congcong Zhang, Yi Yang, Jiawen Zhang, Ying Qiao, and Hongan Wang.*

1. **Learning node abnormality with weak supervision.** CIKM, 2023. [paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614950)

   *Qinghai Zhou, Kaize Ding, Huan Liu, and Hanghang Tong.*

1. **RustGraph: Robust anomaly detection in dynamic graphs by jointly learning structural-temporal dependency.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10301657)

   *Jianhao Guo, Siliang Tang, Juncheng Li, Kaihang Pan, and Lingfei Wu.*

1. **An efficient adaptive multi-kernel learning with safe screening rule for outlier detection.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/document/10310242)

   *Xinye Wang, Lei Duan, Chengxin He, Yuanyuan Chen, and Xindong Wu.*

1. **Anomaly detection in continuous-time temporal provenance graphs.** NIPS, 2023. [paper](https://nips.cc/virtual/2023/76336)

   *Jakub Reha, Giulio Lovisotto, Michele Russo, Alessio Gravina, and Claas Grohnfeldt.*

1. **Open-set graph anomaly detection via normal structure regularisation.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.06835)

   *Qizhou Wang, Guansong Pang, Mahsa Salehi, Wray Buntine, and Christopher Leckie.* 

1. **ADAMM: Anomaly detection of attributed multi-graphs with metadata: A unified neural network approach.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.07355)

   *Konstantinos Sotiropoulos, Lingxiao Zhao, Pierre Jinghong Liang, and Leman Akoglu.* 

1. **Deep joint adversarial learning for anomaly detection on attribute networks.** Information Sciences, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025523014251)

   *Haoyi Fan, Ruidong Wang, Xunhua Huang, Fengbin Zhang, Zuoyong Li, and Shimei Su.* 

1. **Few-shot message-enhanced contrastive learning for graph anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.10370)

   *Fan Xu, Nan Wang, Xuezhi Wen, Meiqi Gao, Chaoqun Guo, and Xibin Zhao.* 

1. **OCGEC: One-class graph embedding classification for DNN backdoor detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01585)

   *Haoyu Jiang, Haiyang Yu, Nan Li, and Ping Yi.* 

1. **Reinforcement neighborhood selection for unsupervised graph anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.05526)

   *Yuanchen Bei, Sheng Zhou, Qiaoyu Tan, Hao Xu, Hao Chen, Zhao Li, and Jiajun Bu.* 

1. **ADA-GAD: Anomaly-denoised autoencoders for graph anomaly detection.** AAAI, 2024. [paper](https://arxiv.org/abs/2312.14535)

   *Junwei He, Qianqian Xu, Yangbangyan Jiang, Zitai Wang, and Qingming Huang.* 

1. **Boosting graph anomaly detection with adaptive message passing.** ICLR, 2024. [paper](https://openreview.net/forum?id=CanomFZssu)

   *Anonymous authors.* 

1. **Frequency domain-oriented complex graph neural networks for graph classification.** TNNLS, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10409552)

   *Youfa Liu and Bo Du.* 

1. **FGAD: Self-boosted knowledge distillation for an effective federated graph anomaly detection framework.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.12761)

   *Jinyu Cai, Yunhe Zhang, Zhoumin Lu, Wenzhong Guo, and See-kiong Ng.* 

1. **Generative semi-supervised graph anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.11887)

   *Hezhe Qiao, Qingsong Wen, Xiaoli Li, Ee-Peng Lim, and Guansong Pang.* 

1. **Graph structure reshaping against adversarial attacks on graph neural networks.** TKDE, 2024. [paper](https://www.computer.org/csdl/journal/tk/5555/01/10538390/1XcOSbOdJD2)

   *Haibo Wang, Chuan Zhou, Xin Chen, Jia Wu, Shirui Pan, Zhao Li, Jilong Wang, and Philip S. Yu.* 

1. **SmoothGNN: Smoothing-based GNN for unsupervised node anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.17525)

   *Xiangyu Dong, Xingyi Zhang, Yanni Sun, Lei Chen, Mingxuan Yuan, and Sibo Wang.* 

1. **Learning-based link anomaly detection in continuous-time dynamic graphs.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.18050)

   *Tim Poštuvan, Claas Grohnfeldt, Michele Russo, and Giulio Lovisotto.* 

### [Time Series](#content)
1. **Variational LSTM enhanced anomaly detection for industrial big data.** TII, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9195000)

   *Xiaokang Zhou, Yiyong Hu, Wei Liang, Jianhua Ma, and Qun Jin.* 

1. **Robust anomaly detection for multivariate time series through stochastic recurrent neural network.** KDD, 2019. [paper](https://dl.acm.org/doi/10.1145/3292500.3330672)

   *Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei.* 

1. **DeepLog: Anomaly detection and diagnosis from system logs through deep learning.** CCS, 2017. [paper](https://dl.acm.org/doi/10.1145/3133956.3134015)

   *Min Du, Feifei Li, Guineng Zheng, and Vivek Srikumar.* 

1. **Unsupervised anomaly detection with LSTM neural networks.** TNNLS, 2019. [paper](https://ieeexplore.ieee.org/abstract/document/8836638)

   *Tolga Ergen and Suleyman Serdar Kozat.* 

1. **LogAnomaly: Unsupervised detection of sequential and quantitative anomalies in unstructured logs.** IJCAI, 2019. [paper](https://www.ijcai.org/proceedings/2019/658)

   *Weibin Meng, Ying Liu, Yichen Zhu, Shenglin Zhang, Dan Pei, Yuqing Liu, Yihao Chen, Ruizhi Zhang, Shimin Tao, Pei Sun, and Rong Zhou.* 

1. **Outlier detection for time series with recurrent autoencoder ensembles.** IJCAI, 2019. [paper](https://dl.acm.org/doi/abs/10.5555/3367243.3367418)

   *Tung Kieu, Bin Yang, Chenjuan Guo, and Christian S. Jensen.* 

1. **Learning regularity in skeleton trajectories for anomaly detection in videos.** CVPR, 2019. [paper](https://dl.acm.org/doi/abs/10.5555/3367243.3367418)

   *Romero Morais, Vuong Le, Truyen Tran, Budhaditya Saha, Moussa Mansour, and Svetha Venkatesh.* 

1. **LSTM-based encoder-decoder for multi-sensor anomaly detection.** arXiv, 2016. [paper](https://arxiv.org/abs/1607.00148)

   *Pankaj Malhotra, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, and Gautam Shroff.* 

1. **CrossFuN: Multi-view joint cross fusion network for time series anomaly detection.** TIM, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10254685)

   *Yunfei Bai, Jing Wang, Xueer Zhang, Xiangtai Miao, and Youfang Linf.* 

1. **Unsupervised anomaly detection by densely contrastive learning for time series data.** Neural Networks, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023005385)

   *Wei Zhu, Weijian Li, E. Ray Dorsey, and Jiebo Luo.* 

1. **Algorithmic recourse for anomaly detection in multivariate time series.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.16896)

   *Xiao Han, Lu Zhang, Yongkai Wu, and Shuhan Yuan.* 

1. **Unravel anomalies: An end-to-end seasonal-trend decomposition approach for time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.00268)

   *Zhenwei Zhang, Ruiqi Wang, Ran Ding, and Yuantao Gu.*

1. **MAG: A novel approach for effective anomaly detection in spacecraft telemetry data.** TII, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10269707)

   *Bing Yu, Yang Yu, Jiakai Xu, Gang Xiang, and Zhiming Yang.*

1. **Duogat: Dual time-oriented graph attention networks for accurate, efficient and explainable anomaly detection on time-series.** CIKM, 2023. [paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614857)

   *Jongsoo Lee, Byeongtae Park, and Dong-Kyu Chae.*

1. **An enhanced spatio-temporal constraints network for anomaly detection in multivariate time series.** KBS, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S095070512300919X)

   *Di Ge, Zheng Dong, Yuhang Cheng, and Yanwen Wu.*

1. **Asymmetric autoencoder with SVD regularization for multivariate time series anomaly detection.** Neural Networks, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006469)

   *Yueyue Yao, Jianghong Ma, Shanshan Feng, and Yunming Ye.*

1. **Unraveling the anomaly in time series anomaly detection: A self-supervised tri-domain solution.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.11235)

   *Yuting Sun, Guansong Pang, Guanhua Ye, Tong Chen, Xia Hu, and Hongzhi Yin.*

1. **A filter-augmented auto-encoder with learnable normalization for robust multivariate time series anomaly detection.** Neural Networks, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006706)

   *Jiahao Yu, Xin Gao, Baofeng Li, Feng Zhai, Jiansheng Lu, Bing Xue, Shiyuan Fu, and Chun Xiao.*

1. **MEMTO: Memory-guided Transformer for multivariate time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.02530)

   *Junho Song, Keonwoo Kim, Jeonglyul Oh, and Sungzoon Cho.*

1. **Entropy causal graphs for multivariate time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.09478)

   *Falih Gozi Febrinanto, Kristen Moore, Chandra Thapa, Mujie Liu, Vidya Saikrishna, Jiangang Ma, and Feng Xia.*

1. **Label-free multivariate time series anomaly detection.** TKDE, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10380724)

   *Qihang Zhou, Shibo He, Haoyu Liu, Jiming Chen, and Wenchao Meng.*

1. **Quantile-long short term memory: A robust, time series anomaly detection method.** TAI, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10398596)

   *Snehanshu Saha, Jyotirmoy Sarkar, Soma Dhavala, Preyank Mota, and Santonu Sarkar.*

1. **PatchAD: Patch-based mlp-mixer for time series anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.09793)

   *Zhijie Zhong, Zhiwen Yu, Yiyuan Yang, Weizheng Wang, and Kaixiang Yang.*

1. **MELODY: Robust semi-supervised hybrid model for entity-level online anomaly detection with multivariate time series.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.10338v1)

   *Jingchao Ni, Gauthier Guinet, Peihong Jiang, Laurent Callot, and Andrey Kan.*

1. **Understanding time series anomaly state detection through one-class classification.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.02007v1)

   *Hanxu Zhou, Yuan Zhang, Guangjie Leng, Ruofan Wang, and Zhi-Qin John Xu.*

1. **Asymptotic consistent graph structure learning for multivariate time-series anomaly detection.** TIM, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10445316)

   *Huaxin Pang, Shikui Wei, Youru Li, Ting Liu, Huaqi Zhang, Ying Qin, and Yao Zhao.*

1. **Anomaly detection via graph attention networks-augmented mask autoregressive flow for multivariate time series.** IoT, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10453361)

   *Hao Liu, Wang Luo, Lixin Han, Peng Gao, Weiyong Yang, and Guangjie Han.*

1. **From chaos to clarity: Time series anomaly detection in astronomical observations.** arXiv, 2024. [paper](https://arxiv.org/abs/2403.10220)

   *Xinli Hao, Yile Chen, Chen Yang, Zhihui Du, Chaohong Ma, Chao Wu, and Xiaofeng Meng.*

1. **DACAD: Domain adaptation contrastive learning for anomaly detection in multivariate time series.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.11269)

   *Zahra Zamanzadeh Darban, Geoffrey I. Webb, and Mahsa Salehi.*

1. **Variate associated domain adaptation for unsupervised multivariate time series anomaly detection.** TKDD, 2024. [paper](https://dl.acm.org/doi/10.1145/3663573)

   *Yifan He, Yatao Bian, Xi Ding, Bingzhe Wu, Jihong Guan, Ji Zhang, and Shuigeng Zhou.*

1. **Quo vadis, unsupervised time series anomaly detection?** ICML, 2024. [paper](https://arxiv.org/abs/2405.02678)

   *M. Saquib Sarfraz, Meiyen Chen, Lukas Layer, Kunyu Peng, and Marios Koulakis.*

1. **SiET: Spatial information enhanced transformer for multivariate time series anomaly detection.** KBS, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705124005628)

   *Weixuan Xiong, Peng Wang, Xiaochen Sun, and Jun Wang.*

1. **Disentangled anomaly detection for multivariate time seriesn.** WWW, 2024. [paper](https://dl.acm.org/doi/abs/10.1145/3589335.3651492)

   *Xin Jie, Xixi Zhou, Chanfei Su, Zijun Zhou, Yuqing Yuan, Jiajun Bu, and Haishuai Wang.*

1. **PATE: Proximity-aware time series anomaly evaluation.** KDD, 2024. [paper](https://arxiv.org/abs/2405.12096)

   *Ramin Ghorbani, Marcel J.T. Reinders, and David M.J. Tax.*

1. **Large language models can be zero-shot anomaly detectors for time series?** KDD, 2024. [paper](https://arxiv.org/abs/2405.14755)

   *Sarah Alnegheimish, Linh Nguyen, Laure Berti-Equille, and Kalyan Veeramachaneni.*

1. **LARA: A light and anti-overfitting retraining approach for unsupervised time series anomaly detection.** WWW, 2024. [paper](https://dl.acm.org/doi/abs/10.1145/3589334.3645472)

   *Feiyi Chen, Zhen Qin, Mengchu Zhou, Yingying Zhang, Shuiguang Deng, Lunting Fan, Guansong Pang, and Qingsong Wen.*

1. **Variate associated domain adaptation for unsupervised multivariate time series anomaly detection.** TKDD, 2024. [paper](https://dl.acm.org/doi/abs/10.1145/3663573)

   *Yifan He, Yatao Bian, Xi Ding, Bingzhe Wu, Jihong Guan, Ji Zhang, and Shuigeng Zhou.*

1. **Uni-directional graph structure learning-based multivariate time series anomaly detection with dynamic prior knowledge.** International Journal of Machine Learning and Cybernetics, 2024. [paper](https://link.springer.com/article/10.1007/s13042-024-02212-5)

   *Shiming He, Genxin Li, Jin Wang, Kun Xie, and Pradip Kumar Sharma.*

1. **Towards a general time series anomaly detector with adaptive bottlenecks and dual adversarial decoders.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.15273)

   *Qichao Shentu, Beibu Li, Kai Zhao, Yang shu, Zhongwen Rao, Lujia Pan, Bin Yang, and Chenjuan Guo.*

1. **USD: Unsupervised soft contrastive learning for fault detection in multivariate time series.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.16258)

   *Hong Liu, Xiuxiu Qiu, Yiming Shi, and Zelin Zang.*

1. **LARA: A light and anti-overfitting retraining approach for unsupervised time series anomaly detection.** WWW, 2024. [paper](https://dl.acm.org/doi/abs/10.1145/3589334.3645472)

   *Feiyi Chen, Zhen Qin, Mengchu Zhou, Yingying Zhang, Shuiguang Deng, Lunting Fan, Guansong Pang, and Qingsong Wen.*

1. **Unraveling anomalies in time: Unsupervised discovery and isolation of anomalous behavior in bio-regenerative life support system telemetry.** ECML PKDD, 2024. [paper](https://arxiv.org/abs/2406.09825)

   *Ferdinand Rewicki, Jakob Gawlikowski, Julia Niebling, and Joachim Denzler.*

### [Tabular](#content)
1. **Beyond individual input for deep anomaly detection on tabular data.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.15121)

   *Hugo Thimonier, Fabrice Popineau, Arpad Rimmel, and Bich-Liên Doan.* 

1. **Fascinating supervisory signals and where to find them: Deep anomaly detection with scale learning.** ICML, 2023. [paper](https://arxiv.org/abs/2305.16114)

   *Hongzuo Xu, Yijie Wang, Juhui Wei, Songlei Jian, Yizhou Li, and Ning Liu.* 

1. **TabADM: Unsupervised tabular anomaly detection with diffusion models.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.12336)

   *Guy Zamberg, Moshe Salhov, Ofir Lindenbaum, and Amir Averbuch.* 

1. **ATDAD: One-class adversarial learning for tabular data anomaly detection.** Computers & Security, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167404823003590)

   *Xiaohui Yang and Xiang Li.* 

1. **Understanding the limitations of self-supervised learning for tabular anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.08374)

   *Kimberly T. Mai, Toby Davies, and Lewis D. Griffin.* 

1. **Unmasking the chameleons: A benchmark for out-of-distribution detection in medical tabular data.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.16220)

   *Mohammad Azizmalayeri, Ameen Abu-Hanna, and Giovanni Ciná.* 

1. **TDeLTA: A light-weight and robust table detection method based on learning text arrangement.** AAAI, 2024. [paper](https://arxiv.org/abs/2309.16220)

   *Yang Fan, Xiangping Wu, Qingcai Chen, Heng Li, Yan Huang, Zhixiang Cai, and Qitian Wu.* 

1. **How to overcome curse-of-dimensionality for out-of-distribution detection?** AAAI, 2024. [paper](https://arxiv.org/abs/2312.11043)

   *Soumya Suvra Ghosal, Yiyou Sun, and Yixuan Li.* 

1. **MCM: Masked cell modeling for anomaly detection in tabular data.** ICLR, 2024. [paper](https://openreview.net/forum?id=lNZJyEDxy4)

   *Anonymous authors.* 

### [Out of Distribution](#content)
1. **Your out-of-distribution detection method is not robust!** NIPS, 2022. [paper](https://arxiv.org/abs/2209.15246)

   *Mohammad Azizmalayeri, Arshia Soltani Moakhar, Arman Zarei, Reihaneh Zohrabi, Mohammad Taghi Manzuri, and Mohammad Hossein Rohban.* 

1. **Exploiting mixed unlabeled data for detecting samples of seen and unseen out-of-distribution classes.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20814)

   *Yixuan Sun and Wei Wang.* 

1. **Detect, distill and update: Learned DB systems facing out of distribution data.** SIGMOD, 2023. [paper](https://arxiv.org/abs/2210.05508)

   *Meghdad Kurmanji and Peter Triantafillou.* 

1. **Beyond mahalanobis distance for textual OOD detection.** NIPS, 2022. [paper](https://openreview.net/forum?id=ReB7CCByD6U)

   *Pierre Colombo, Eduardo Dadalto Câmara Gomes, Guillaume Staerman, Nathan Noiry, and Pablo Piantanida.* 

1. **Exploring the limits of out-of-distribution detection.** NIPS, 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/3941c4358616274ac2436eacf67fae05-Abstract.html)

   *Stanislav Fort, Jie Ren, and Balaji Lakshminarayanan.* 

1. **Is out-of-distribution detection learnable?** ICLR, 2022. [paper](https://openreview.net/forum?id=sde_7ZzGXOE)

   *Zhen Fang, Yixuan Li, Jie Lu, Jiahua Dong, Bo Han, and Feng Liu.* 

1. **Out-of-distribution detection is not all you need.** AAAI, 2023. [paper](https://arxiv.org/abs/2211.16158)

   *Joris Guerin, Kevin Delmas, Raul Sena Ferreira, and Jérémie Guiochet.* 

1. **iDECODe: In-distribution equivariance for conformal out-of-distribution detection.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20670)

   *Ramneet Kaur, Susmit Jha, Anirban Roy, Sangdon Park, Edgar Dobriban, Oleg Sokolsky, and Insup Lee.* 

1. **Out-of-distribution detection using an ensemble of self supervised leave-out classifiers.** ECCV, 2018. [paper](https://link.springer.com/chapter/10.1007/978-3-030-01237-3_34)

   *Apoorv Vyas, Nataraj Jammalamadaka, Xia Zhu, Dipankar Das, Bharat Kaul, and Theodore L. Willke.* 

1. **Self-supervised learning for generalizable out-of-distribution detection.** AAAI, 2020. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/5966)

   *Sina Mohseni, Mandar Pitale, JBS Yadawa, and Zhangyang Wang.* 

1. **Augmenting softmax information for selective classification with out-of-distribution data.** ACCV, 2022. [paper](https://openaccess.thecvf.com/content/ACCV2022/html/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.html)

   *Guoxuan Xia and Christos-Savvas Bouganis.* 

1. **Robustness to spurious correlations improves semantic out-of-distribution detection.** AAAI, 2023. [paper](https://arxiv.org/abs/2302.04132)

   *Lily H. Zhang and Rajesh Ranganath.* 

1. **Out-of-distribution detection with implicit outlier transformation.** ICLR, 2023. [paper](https://openreview.net/forum?id=hdghx6wbGuD)

   *Qizhou Wang, Junjie Ye, Feng Liu, Quanyu Dai, Marcus Kalander, Tongliang Liu, Jianye Hao, and Bo Han.* 

1. **Out-of-distribution representation learning for time series classification.** ICLR, 2023. [paper](https://openreview.net/forum?id=gUZWOE42l6Q)

   *Wang Lu, Jindong Wang, Xinwei Sun, Yiqiang Chen, and Xing Xie.* 

1. **Out-of-distribution detection based on in-distribution data patterns memorization with modern Hopfield energy.** ICLR, 2023. [paper](https://openreview.net/forum?id=KkazG4lgKL)

   *Jinsong Zhang, Qiang Fu, Xu Chen, Lun Du, Zelin Li, Gang Wang, xiaoguang Liu, Shi Han, and Dongmei Zhang.* 

1. **Diversify and disambiguate: Out-of-distribution robustness via disagreement.** ICLR, 2023. [paper](https://openreview.net/forum?id=RVTOp3MwT3n)

   *Yoonho Lee, Huaxiu Yao, and Chelsea Finn.* 

1. **Rethinking out-of-distribution (OOD) detection: Masked image nodeling is all you need.** CVPR, 2023. [paper](https://arxiv.org/abs/2302.02615v2)

   *Jingyao Li, Pengguang Chen, Shaozuo Yu, Zexin He, Shu Liu, and Jiaya Jia.* 

1. **LINe: Out-of-distribution detection by leveraging important neurons.** CVPR, 2023. [paper](https://arxiv.org/abs/2303.13995)

   *Yong Hyun Ahn, Gyeong-Moon Park, and Seong Tae Kim.* 

1. **Block selection method for using feature norm in out-of-distribution detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2212.02295)

   *Yeonguk Yu, Sungho Shin, Seongju Lee, Changhyun Jun, and Kyoobin Lee.* 

1. **Devil is in the queries: Advancing mask transformers for real-world medical image segmentation and out-of-distribution localization.** CVPR, 2023. [paper](https://arxiv.org/abs/2304.00212)

   *Mingze Yuan, Yingda Xia, Hexin Dong, Zifan Chen, Jiawen Yao, Mingyan Qiu, Ke Yan, Xiaoli Yin, Yu Shi, Xin Chen, Zaiyi Liu, Bin Dong, Jingren Zhou, Le Lu, Ling Zhang, and Li Zhang.* 

1. **Unleashing mask: Explore the intrinsic out-of-distribution detection capability.** ICML, 2023. [paper](https://arxiv.org/abs/2306.03715)

   *Jianing Zhu, Hengzhuang Li, Jiangchao Yao, Tongliang Liu, Jianliang Xu, and Bo Han.* 

1. **DOS: Diverse outlier sampling for out-of-distribution detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.02031)

   *Wenyu Jiang, Hao Cheng, Mingcai Chen, Chongjun Wang, and Hongxin Wei.* 

1. **POEM: Out-of-distribution detection with posterior sampling.** ICML, 2022. [paper](https://proceedings.mlr.press/v162/ming22a.html)

   *Yifei Ming, Ying Fan, and Yixuan Li.* 

1. **Balanced energy regularization loss for out-of-distribution detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2306.10485)

   *Hyunjun Choi, Hawook Jeong, and Jin Young Choi.* 

1. **A cosine similarity-based method for out-of-distribution detection.** ICML, 2023. [paper](https://arxiv.org/abs/2306.14920)

   *Nguyen Ngoc-Hieu, Nguyen Hung-Quang, The-Anh Ta, Thanh Nguyen-Tang, Khoa D Doan, and Hoang Thanh-Tung.* 

1. **Beyond AUROC & co. for evaluating out-of-distribution detection performance.** CVPR, 2023. [paper](https://arxiv.org/abs/2306.14658)

   *Galadrielle Humblot-Renaux, Sergio Escalera, and Thomas B. Moeslund.* 

1. **Feed two birds with one scone: Exploiting wild data for both out-of-distribution generalization and detection.** ICML, 2023. [paper](https://arxiv.org/abs/2306.09158)

   *Haoyue Bai, Gregory Canal, Xuefeng Du, Jeongyeol Kwon, Robert Nowak, and Yixuan Li.* 

1. **Key feature replacement of in-distribution samples for out-of-distribution detection.** AAAI, 2023. [paper](https://arxiv.org/abs/2301.13012)

   *Jaeyoung Kim, Seo Taek Kong, Dongbin Na, and Kyu-Hwan Jung.* 

1. **Heatmap-based out-of-distribution detection.** AAAI, 2023. [paper](https://ieeexplore.ieee.org/document/10030868/)

   *Julia Hornauer and Vasileios Belagiannis.* 

1. **RankFeat: Rank-1 feature removal for out-of-distribution detection.** NIPS, 2022. [paper](https://openreview.net/forum?id=-deKNiSOXLG)

   *Yue Song, Nicu Sebe, and Wei Wang.* 

1. **Delving into out-of-distribution detection with vision-language representations.** NIPS, 2022. [paper](https://openreview.net/forum?id=KnCS9390Va)

   *Yifei Ming, Ziyang Cai, Jiuxiang Gu, Yiyou Sun, Wei Li, and Yixuan Li.* 

1. **Detecting out-of-distribution data through in-distribution class prior.** ICML, 2023. [paper](https://openreview.net/forum?id=charggEv8v)

   *Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, and Bo Han.* 

1. **Out-of-distribution detection for monocular depth estimation.** ICCV, 2023. [paper](https://arxiv.org/abs/2308.06072)

   *Julia Hornauer, Adrian Holzbock, and Vasileios Belagiannis.* 

1. **Expecting the unexpected: Towards broad out-of-distribution detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.11480)

   *Charles Guille-Escuret, Pierre-André Noël, Ioannis Mitliagkas, David Vazquez, and Joao Monteiro.* 

1. **ATTA: Anomaly-aware test-time adaptation for out-of-distribution detection in segmentation.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.05994)

   *Zhitong Gao, Shipeng Yan, and Xuming He.* 

1. **Meta OOD learning for continuously adaptive OOD detection.** ICCV, 2023. [paper](https://arxiv.org/abs/2309.11705)

   *Xinheng Wu, Jie Lu, Zhen Fang, and Guangquan Zhang.* 

1. **Nearest neighbor guidance for out-of-distribution detection.** ICCV, 2023. [paper](https://arxiv.org/abs/2309.14888)

   *Jaewoo Park, Yoon Gyo Jung, and Andrew Beng Jin Teoh.* 

1. **Can pre-trained networks detect familiar out-of-distribution data?** arXiv, 2023. [paper](https://arxiv.org/abs/2310.00847)

   *Atsuyuki Miyai, Qing Yu, Go Irie, and Kiyoharu Aizawa.* 

1. **Understanding the feature norm for out-of-distribution detection.** ICCV, 2023. [paper](https://arxiv.org/abs/2310.05316)

   *Jaewoo Park, Jacky Chen Long Chai, Jaeho Yoon, and Andrew Beng Jin Teoh.* 

1. **Detecting out-of-distribution through the lens of neural collapse.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.01479)

   *Litian Liu and Yao Qin.* 

1. **Learning to augment distributions for out-of-distribution detection.** NIPS, 2023. [paper](https://openreview.net/forum?id=OtU6VvXJue)

   *Qizhou Wang, Zhen Fang, Yonggang Zhang, Feng Liu, Yixuan Li, and Bo Han.* 

1. **Incremental object-based novelty detection with feedback loop.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.09004)

   *Simone Caldarella, Elisa Ricci, and Rahaf Aljundi.* 

1. **Out-of-distribution knowledge distillation via confidence amendment.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.07975)

   *Zhilin Zhao, Longbing Cao, and Yixuan Zhang.* 

1. **Fast trainable projection for robust fine-tuning.** NIPS, 2023. [paper](https://openreview.net/forum?id=Tb7np0MInj)

   *Junjiao Tian, Yencheng Liu, James Seale Smith, and Zsolt Kira.* 

1. **Trainable projected gradient method for robust fine-tuning.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Tian_Trainable_Projected_Gradient_Method_for_Robust_Fine-Tuning_CVPR_2023_paper.html)

   *Junjiao Tian, Xiaoliang Dai, Chih-Yao Ma, Zecheng He, Yen-Cheng Liu, and Zsolt Kira.* 

1. **GAIA: Delving into gradient-based attribution abnormality for out-of-distribution detection.** NIPS, 2023. [paper](https://openreview.net/forum?id=XEBzQP3e7B)

   *Jinggang Chen, Junjie Li, Xiaoyang Qu, Jianzong Wang, Jiguang Wan, and Jing Xiao.* 

1. **Domain aligned CLIP for few-shot classification.** WACV, 2024. [paper](https://arxiv.org/abs/2311.09191)

   *Muhammad Waleed Gondal, Jochen Gast, Inigo Alonso Ruiz, Richard Droste, Tommaso Macri, Suren Kumar, and Luitpold Staudigl.* 

1. **Towards few-shot out-of-distribution detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.12076)

   *Jiuqing Dong, Yongbin Gao, Heng Zhou, Jun Cen, Yifan Yao, Sook Yoon, and Park Dong Sun.* 

1. **Denoising diffusion models for out-of-distribution detection.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Graham_Denoising_Diffusion_Models_for_Out-of-Distribution_Detection_CVPRW_2023_paper.html)

   *Mark S. Graham, Walter H.L. Pinaya, Petru-Daniel Tudosiu, Parashkev Nachev, Sebastien Ourselin, and Jorge Cardoso.* 

1. **RankFeat&RankWeight: Rank-1 feature/weight removal for out-of-distribution detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.13959)

   *Yue Song, Nicu Sebe, and Wei Wang.* 

1. **ID-like prompt learning for few-shot out-of-distribution detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.15243)

   *Yichen Bai, Zongbo Han, Changqing Zhang, Bing Cao, Xiaoheng Jiang, and Qinghua Hu.* 

1. **Segment every out-of-distribution object.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.16516)

   *Wenjie Zhao, Jia Li, Xin Dong, Yu Xiang, and Yunhui Guo.* 

1. **Raising the Bar of AI-generated image detection with CLIP.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.00195)

   *Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner, and Luisa Verdoliva.* 

1. **Likelihood-aware semantic alignment for full-spectrum out-of-distribution detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01732)

   *Fan Lu, Kai Zhu, Kecheng Zheng, Wei Zhai, and Yang Cao.* 

1. **How low can you go? Surfacing prototypical in-distribution samples for unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.03804)

   *Felix Meissen, Johannes Getzner, Alexander Ziller, Georgios Kaissis, and Daniel Rueckert.* 

1. **Residual pattern learning for pixel-wise out-of-distribution detection in semantic segmentation.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Residual_Pattern_Learning_for_Pixel-Wise_Out-of-Distribution_Detection_in_Semantic_Segmentation_ICCV_2023_paper.html)

   *Yuyuan Liu, Choubo Ding, Yu Tian, Guansong Pang, Vasileios Belagiannis, Ian Reid, and Gustavo Carneiro.* 

1. **EAT: Towards long-tailed out-of-distribution detection.** AAAI, 2024. [paper](https://arxiv.org/abs/2312.08939)

   *Tong Wei, Bolin Wang, and Minling Zhang.* 

1. **GROOD: GRadient-aware out-Of-distribution detection in interpolated manifolds.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.14427)

   *Mostafa ElAraby, Sabyasachi Sahoo, Yann Pequignot, Paul Novello, and Liam Paull.* 

1. **On the robustness of ChatGPT: An adversarial and out-of-distribution perspective.** arXiv, 2023. [paper](https://arxiv.org/abs/2302.12095)

   *Jindong Wang, Xixu Hu, Wenxin Hou, Hao Chen, Runkai Zheng, Yidong Wang, Linyi Yang, Haojun Huang, Wei Ye, Xiubo Geng, Binxin Jiao, Yue Zhang, and Xing Xie.* 

1. **Out-of-distribution detection in long-tailed recognition with calibrated outlier class learning.** AAAI, 2024. [paper](https://arxiv.org/abs/2312.10686)

   *Wenjun Miao, Guansong Pang, Tianqi Li, Xiao Bai, and Jin Zheng.* 

1. **Towards optimal feature-shaping methods for out-of-distribution detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=dm8e7gsH0d)

   *Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, and Stephen Gould.* 

1. **Learning with mixture of prototypes for out-of-distribution detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=uNkKaD3MCs)

   *Haodong Lu, Dong Gong, Shuo Wang, Jason Xue, Lina Yao, and Kristen Moore.* 

1. **Zero-shot object-level OOD detection with context-aware inpainting.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.03292)

   *Quang-Huy Nguyen, Jin Peng Zhou, Zhenzhen Liu, Khanh-Huyen Bui, Kilian Q. Weinberger, and Dung D. Le.* 

1. **How does wild data provably help OOD detection?** ICLR, 2024. [paper](https://openreview.net/forum?id=jlEjB8MVGa)

   *Xuefeng Du, Zhen Fang, Ilias Diakonikolas, and Yixuan Li.* 

1. **Optimal parameter and neuron pruning for out-of-distribution detection.** NIPS, 2023. [paper](https://openreview.net/forum?id=TtCPFN5fhO&referrer=%5Bthe%20profile%20of%20Ze%20Chen%5D(%2Fprofile%3Fid%3D~Ze_Chen3))

   *Chao Chen, Zhihang Fu, Kai Liu, Ze Chen, Mingyuan Tao, and Jieping Ye.* 

1. **Out-of-distribution detection should use conformal prediction (and vice-versa)?** arXiv, 2024. [paper](https://arxiv.org/abs/2403.11532)

   *Paul Novello, Joseba Dalmau, and Léo Andeol.* 

1. **Tagfog: Textual anchor guidance and fake outlier generation for visual out-of-distribution detection.** AAAI, 2024. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/27871)

   *Jiankang Chen, Tong Zhang, Weishi Zheng, and Ruixuan Wang.* 

1. **Learning transferable negative prompts for out-of-distribution detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.03248)

   *Tianqi Li, Guansong Pang, Xiao Bai, Wenjun Miao, and Jin Zheng.* 

1. **Unexplored faces of robustness and out-of-distribution: Covariate shifts in environment and sensor domains.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.15882)

   *Alvin Heng, Alexandre H. Thiery, and Harold Soh.* 

1. **Out-of-distribution detection with a single unconditional diffusion model.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.11881)

   *Eunsu Baek, Keondo Park, Jiyoon Kim, and Hyung-Sin Kim.* 

1. **MultiOOD: Scaling out-of-distribution detection for multiple modalities.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.17419)

   *Hao Dong, Yue Zhao, Eleni Chatzi, and Olga Fink.* 

1. **When and how does in-distribution label help out-of-distribution detection?** ICML, 2024. [paper](https://arxiv.org/abs/2405.18635)

   *Xuefeng Du, Yiyou Sun, and Yixuan Li.* 

1. **Test-time linear out-of-distribution detection.** CVPR, 2024. [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Test-Time_Linear_Out-of-Distribution_Detection_CVPR_2024_paper.html)

   *Ke Fan, Tong Liu, Xingyu Qiu, Yikai Wang, Lian Huai, Zeyu Shangguan, Shuang Gou, Fengjian Liu, Yuqian Fu, Yanwei Fu, and Xingqun Jiang.* 

### [Large Model](#content)
1. **WinCLIP: Zero-/few-shot anomaly classification and segmentation.** CVPR, 2023. [paper](https://arxiv.org/abs/2303.05047)

   *Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, and Onkar Dabeer.* 

1. **Semantic anomaly detection with large language models.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.11307)

   *Amine Elhafsi, Rohan Sinha, Christopher Agia, Edward Schmerling, Issa Nesnas, and Marco Pavone.* 

1. **AnomalyGPT: Detecting industrial anomalies using large vision-language models.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.15366)

   *Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, and Jinqiao Wang.* 

1. **AnoVL: Adapting vision-language models for unified zero-shot anomaly localization.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.15939)

   *Hanqiu Deng, Zhaoxiang Zhang, Jinan Bao, and Xingyu Li.* 

1. **LogGPT: Exploring ChatGPT for log-based anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.01189)

   *Jiaxing Qi, Shaohan Huang, Zhongzhi Luan, Carol Fung, Hailong Yang, and Depei Qian.* 

1. **CLIPN for zero-shot OOD detection: Teaching CLIP to say no.** ICCV, 2023. [paper](https://arxiv.org/abs/2308.12213)

   *Hualiang Wang, Yi Li, Huifeng Yao, and Xiaomeng Li.* 

1. **LogGPT: Log anomaly detection via GPT.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.14482)

   *Xiao Han, Shuhan Yuan, and Mohamed Trabelsi.* 

1. **Semantic scene difference detection in daily life patroling by mobile robots using pre-trained large-scale vision-language model.** IROS, 2023. [paper](https://arxiv.org/abs/2309.16552)

   *Yoshiki Obinata, Kento Kawaharazuka, Naoaki Kanazawa, Naoya Yamaguchi, Naoto Tsukamoto, Iori Yanokura, Shingo Kitagawa, Koki Shinjo, Kei Okada, and Masayuki Inaba.* 

1. **HuntGPT: Integrating machine learning-based anomaly detection and explainable AI with large language models (LLMs).** arXiv, 2023. [paper](https://arxiv.org/abs/2309.16021)

   *Tarek Ali and Panos Kostakos.* 

1. **Graph neural architecture search with GPT-4.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.01436)

   *Haishuai Wang, Yang Gao, Xin Zheng, Peng Zhang, Hongyang Chen, and Jiajun Bu.* 

1. **Exploring large language models for multi-modal out-of-distribution detection.** EMNLP, 2023. [paper](https://arxiv.org/abs/2310.08027)

   *Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, and Yongbin Li.* 

1. **Detecting pretraining data from large language models.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.16789)

   *Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer.* 

1. **AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=buC4E91xZE)

   *Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, and Jiming Chen.* 

1. **CLIP-AD: A language-guided staged dual-path model for zero-shot anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.00453)

   *Xuhai Chen, Jiangning Zhang, Guanzhong Tian, Haoyang He, Wuhao Zhang, Yabiao Wang, Chengjie Wang, Yunsheng Wu, and Yong Liu.* 

1. **Exploring grounding potential of VQA-oriented GPT-4V for zero-shot anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.02612)

   *Jiangning Zhang, Xuhai Chen, Zhucun Xue, Yabiao Wang, Chengjie Wang, and Yong Liu.* 

1. **Open-vocabulary video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.07042)

   *Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang.* 

1. **Distilling out-of-distribution robustness from vision-language foundation models.** NIPS, 2023. [paper](https://neurips.cc/virtual/2023/poster/70716)

   *Andy Zhou, Jindong Wang, Yuxiong Wang, and Haohan Wang.* 

1. **Weakly supervised detection of gallucinations in LLM activations.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.02798)

   *Miriam Rateike, Celia Cintas, John Wamburu, Tanya Akumu, and Skyler Speakman.* 

1. **How well does GPT-4V(ision) adapt to distribution shifts? A preliminary investigation.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.07424)

   *Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Xing Xie, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, and Kun Zhang.* 

1. **overlooked video classification in weakly supervised video anomaly detection.** WACV, 2024. [paper](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Tan_Overlooked_Video_Classification_in_Weakly_Supervised_Video_Anomaly_Detection_WACVW_2024_paper.html)

   *Weijun Tan, Qi Yao, and Jingfeng Liu.* 

1. **Video anomaly detection and explanation via large language models.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.05702)

   *Hui Lv and Qianru Sun.* 

1. **OVOR: OnePrompt with virtual outlier regularization for rehearsal-free class-incremental learning.** ICLR, 2024. [paper](https://openreview.net/forum?id=FbuyDzZTPt)

   *Weicheng Huang, Chunfu Chen, and Hsiang Hsu.* 

1. **Large language model guided knowledge distillation for time series anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.15123v1)

   *Chen Liu, Shibo He, Qihang Zhou, Shizhong Li, and Wenchao Meng.* 

1. **Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts.** CVPR, 2024. [paper](https://arxiv.org/abs/2403.06495)

   *Jiawen Zhu and Guansong Pang.* 

1. **Adapting visual-language models for generalizable anomaly detection in medical images.** CVPR, 2024. [paper](https://arxiv.org/abs/2403.12570v1)

   *Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, and Yanfeng Wang.* 

1. **Harnessing large language models for training-free video anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.01014)

   *Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci.* 

1. **Collaborative learning of anomalies with privacy (CLAP) for unsupervised video anomaly detection: A new baseline.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.00847)

   *Anas Al-lahham, Muhammad Zaigham Zaheer, Nurbek Tastan, and Karthik Nandakumar.* 

1. **PromptAD: Learning prompts with only normal samples for few-shot anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.05231)

   *Xiaofan Li, Zhizhong Zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yuan Xie, and Lizhuang Ma.* 

1. **Dynamic distinction learning: Adaptive pseudo anomalies for video anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.04986)

   *Demetris Lappas, Vasileios Argyriou, and Dimitrios Makris.* 

1. **Your finetuned large language model is already a powerful out-of-distribution detector.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.08679)

   *Andi Zhang, Tim Z. Xiao, Weiyang Liu, Robert Bamler, and Damon Wischik.* 

1. **Do LLMs understand visual anomalies? Uncovering LLM capabilities in zero-shot anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.09654)

   *Jiaqi Zhu, Shaofeng Cai, Fang Deng, and Junran Wu.* 

1. **Text prompt with normality guidance for weakly supervised video anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.08531)

   *Zhiwei Yang, Jing Liu, and Peng Wu.* 

1. **FiLo: Zero-shot anomaly detection by fine-grained description and high-quality localization.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.08531)

   *Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Hao Li, Ming Tang, and Jinqiao Wang.* 

1. **AnomalyDINO: Boosting patch-based few-shot anomaly detection with DINOv2.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.14529)

   *Simon Damm, Mike Laszkiewicz, Johannes Lederer, and Asja Fischer.* 

1. **Large language models can deliver accurate and interpretable time series anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.15370)

   *Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and Ying-Cong Chen.* 

1. **Hawk: Learning to understand open-world video anomalies.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.15370)

   *Jun Liu, Chaoyun Zhang, Jiaxu Qian, Minghua Ma, Si Qin, Chetan Bansal, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang.* 

1. **ARC: A generalist graph anomaly detector with in-context learning.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.16771)

   *Yixin Liu, Shiyuan Li, Yu Zheng, Qingfeng Chen, Chengqi Zhang, and Shirui Pan.* 

### [Reinforcement Learning](#content)
1. **Towards experienced anomaly detector through reinforcement learning.** AAAI, 2018. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/12130)

   *Chengqiang Huang, Yulei Wu, Yuan Zuo, Ke Pei, and Geyong Min.* 

1. **Sequential anomaly detection using inverse reinforcement learning.** KDD, 2019. [paper](https://dl.acm.org/doi/10.1145/3292500.3330932)

   *Min-hwan Oh and Garud Iyengar.* 

1. **Toward deep supervised anomaly detection: Reinforcement learning from partially labeled anomaly data.** KDD, 2021. [paper](https://dl.acm.org/doi/10.1145/3447548.3467417)

   *Guansong Pang, Anton van den Hengel, Chunhua Shen, and Longbing Cao.* 

1. **Automated anomaly detection via curiosity-guided search and self-imitation learning.** TNNLS, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9526875)

   *Yuening Li, Zhengzhang Chen, Daochen Zha, Kaixiong Zhou, Haifeng Jin, Haifeng Chen, and Xia Hu.* 

1. **Meta-AAD: Active anomaly detection with deep reinforcement learning.** ICDM, 2020. [paper](https://ieeexplore.ieee.org/document/9338270)

   *Daochen Zha, Kwei-Herng Lai, Mingyang Wan, and Xia Hu.* 

1. **Semi-supervised learning via DQN for log anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.03151)

   *Yingying He, Xiaobing Pei, and Lihong Shen.* 

1. **OIL-AD: An anomaly detection framework for sequential decision sequences.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.04567)

   *Chen Wang, Sarah Erfani, Tansu Alpcan, and Christopher Leckie.* 

### [In-Context Learning](#content)
1. **Prompt-enhanced multiple instance learning for weakly supervised video anomaly detection.** CVPR, 2024. [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html)

   *Junxi Chen, Liang Li, Li Su, Zheng-jun Zha, and Qingming Huang.* 

### [Representation Learning](#content)
1. **Localizing anomalies from weakly-labeled videos.** TIP, 2021. [paper](https://ieeexplore.ieee.org/document/9408419)

   *Hui Lv, Chuanwei Zhou, Zhen Cui, Chunyan Xu, Yong Li, and Jian Yang.* 

1. **PAC-Wrap: Semi-supervised PAC anomaly detection.** KDD, 2022. [paper](https://arxiv.org/abs/2205.10798)

   *Shuo Li, Xiayan Ji, Edgar Dobriban, Oleg Sokolsky, and Insup Lee.* 

1. **Effective end-to-end unsupervised outlier detection via inlier priority of discriminative network.** NIPS, 2019. [paper](https://proceedings.neurips.cc/paper/2019/hash/6c4bb406b3e7cd5447f7a76fd7008806-Abstract.html)

   *Siqi Wang, Yijie Zeng, Xinwang Liu, En Zhu, Jianping Yin, Chuanfu Xu, and Marius Kloft.* 

1. **AnomalyHop: An SSL-based image anomaly localization method.** ICVCIP, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9675385)

   *Kaitai Zhang, Bin Wang, Wei Wang, Fahad Sohrab, Moncef Gabbouj, and C.-C. Jay Kuo.* 

1. **Learning representations of ultrahigh-dimensional data for random distance-based outlier detection.** KDD, 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220042)

   *Guansong Pang, Longbing Cao, Ling Chen, and Huan Liu.* 

1. **Federated disentangled representation learning for unsupervised brain anomaly detection.** NMI, 2022. [paper](https://www.nature.com/articles/s42256-022-00515-2)

   *Cosmin I. Bercea, Benedikt Wiestler, Daniel Rueckert, and Shadi Albarqouni.* 

1. **DSR–A dual subspace re-projection network for surface anomaly detection.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31)

   *Vitjan Zavrtanik, Matej Kristan, and Danijel Skočaj.* 

1. **LGN-Net: Local-global normality network for video anomaly detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2211.07454)

   *Mengyang Zhao, Yang Liu, Jing Liu, Di Li, and Xinhua Zeng.* 

1. **Glancing at the patch: Anomaly localization with global and local feature comparison.** CVPR, 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Glancing_at_the_Patch_Anomaly_Localization_With_Global_and_Local_CVPR_2021_paper.html)

   *Shenzhi Wang, Liwei Wu, Lei Cui, and Yujun Shen.* 

1. **SPot-the-difference self-supervised pre-training for anomaly detection and segmentation.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_23)

   *Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer.* 

1. **SSD: A unified framework for self-supervised outlier detection.** ICLR, 2021. [paper](https://openreview.net/forum?id=v5gjXpmR8J)

   *Vikash Sehwag, Mung Chiang, and Prateek Mittal.* 

1. **NETS: Extremely fast outlier detection from a data stream via set-based processing.** VLDB, 2019. [paper](https://openreview.net/forum?id=v5gjXpmR8J)

   *Susik Yoon, Jae-Gil Lee, and Byung Suk Lee.* 

1. **XGBOD: Improving supervised outlier detection with unsupervised representation learning.** IJCNN, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8489605)

   *Yue Zhao and Maciej K. Hryniewicki.* 

1. **Red PANDA: Disambiguating anomaly detection by removing nuisance factors.** ICLR, 2023. [paper](https://openreview.net/forum?id=z37tDDHHgi)

   *Niv Cohen, Jonathan Kahana, and Yedid Hoshen.* 

1. **TimesNet: Temporal 2D-variation modeling for general time series analysis.** ICLR, 2023. [paper](https://openreview.net/forum?id=ju_Uqw384Oq)

   *Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long.* 

1. **SimpleNet: A simple network for image anomaly detection and localization.** CVPR, 2023. [paper](https://arxiv.org/abs/2303.15140)

   *Zhikang Liu, Yiming Zhou, Yuansheng Xu, and Zilei Wang.* 

1. **Unsupervised anomaly detection via nonlinear manifold learning.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.09441)

   *Amin Yousefpour, Mehdi Shishehbor, Zahra Zanjani Foumani, and Ramin Bostanabad.* 

1. **Representation learning in anomaly detection: Successes, limits and a grand challenge.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.11085v1)

   *Yedid Hoshen.* 

1. **A lightweight video anomaly detection model with weak supervision and adaptive instance selection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.05330)

   *Yang Wang, Jiaogen Zhou, and Jihong Guan.* 

1. **MGFN: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection.** AAAI, 2023. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25112)

   *Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu.* 

1. **TeD-SPAD: Temporal distinctiveness for self-supervised privacy-preservation for video anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Fioresi_TeD-SPAD_Temporal_Distinctiveness_for_Self-Supervised_Privacy-Preservation_for_Video_Anomaly_Detection_ICCV_2023_paper.html)

   *Joseph Fioresi, Ishan Rajendrakumar Dave, and Mubarak Shah.* 

1. **Deep orthogonal hypersphere compression for anomaly detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=cJs4oE4m9Q)

   *Yunhe Zhang, Yan Sun, Jinyu Cai, and Jicong Fan.* 

1. **VI-OOD: A unified representation learning framework for textual out-of-distribution detection.** COLING, 2024. [paper](https://arxiv.org/abs/2404.06217)

   *Yunhe Zhang, Yan Sun, Jinyu Cai, and Jicong Fan.* 




## [Mechanism](#content)
### [Dataset](#content)
1. **DoTA: Unsupervised detection of traffic anomaly in driving videos.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/document/9712446)

   *Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Yuchen Wang, Ella Atkins, and David Crandall.* 

1. **Revisiting time series outlier detection: Definitions and benchmarks.** NIPS, 2021. [paper](https://openreview.net/forum?id=r8IvOsnHchr)

   *Kwei-Herng Lai, Daochen Zha, Junjie Xu, Yue Zhao, Guanchu Wang, and Xia Hu.* 

1. **Street scene: A new dataset and evaluation protocol for video anomaly detection.** WACV, 2020. [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf)

   *Bharathkumar Ramachandra and Michael J. Jones.*

1. **The eyecandies dataset for unsupervised multimodal anomaly detection and localization.** ACCV, 2020. [paper](https://openaccess.thecvf.com/content/ACCV2022/html/Bonfiglioli_The_Eyecandies_Dataset_for_Unsupervised_Multimodal_Anomaly_Detection_and_Localization_ACCV_2022_paper.html)

   *Luca Bonfiglioli, Marco Toschi, Davide Silvestri, Nicola Fioraio, and Daniele De Gregorio.*

1. **Not only look, but also listen: Learning multimodal violence detection under weak supervision.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_20)

   *Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang.* 

1. **A revisit of sparse coding based anomaly detection in stacked RNN framework.** ICCV, 2017. [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Luo_A_Revisit_of_ICCV_2017_paper.html)

   *Weixin Luo, Wen Liu, and Shenghua Gao.* 

1. **The MVTec anomaly detection dataset: A comprehensive real-world dataset for unsupervised anomaly detection.** IJCV, 2021. [paper](https://link.springer.com/article/10.1007/s11263-020-01400-4)

   *Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger.* 

1. **Anomaly detection in crowded scenes.** CVPR, 2010. [paper](https://ieeexplore.ieee.org/abstract/document/5539872/)

   *Vijay Mahadevan, Weixin Li, Viral Bhalodia, and Nuno Vasconcelos.* 

1. **Abnormal event detection at 150 FPS in MATLAB.** ICCV, 2013. [paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/html/Lu_Abnormal_Event_Detection_2013_ICCV_paper.html)

   *Cewu Lu, Jianping Shi, and Jiaya Jia.* 

1. **Surface defect saliency of magnetic tile.** The Visual Computer, 2020. [paper](https://link.springer.com/article/10.1007/s00371-018-1588-5)

   *Yibin Huang, Congying Qiu, and Kui Yuan.* 

1. **Audio-visual dataset and method for anomaly detection in traffic videos.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.15084)

   *Błażej Leporowski, Arian Bakhtiarnia, Nicole Bonnici, Adrian Muscat, Luca Zanella, Yiming Wang, and Alexandros Iosifidis.* 

1. **Flow-Bench: A dataset for computational workflow anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.09930)

   *George Papadimitriou, Hongwei Jin, Cong Wang, Krishnan Raghavan, Anirban Mandal, Prasanna Balaprakash, and Ewa Deelman.* 

1. **In or Out? Fixing ImageNet out-of-distribution detection evaluation.** ICML, 2023. [paper](https://arxiv.org/abs/2306.00826)

   *Julian Bitterwolf, Maximilian Müller, and Matthias Hein.* 

1. **Temporal graphs anomaly emergence detection: Benchmarking for social media interactions.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.05268)

   *Teddy Lazebnik and Or Iny.* 

1. **PKU-GoodsAD: A supermarket goods dataset for unsupervised anomaly detection and segmentation.** RAL, 2024. [paper](https://ieeexplore.ieee.org/document/10387569/)

   *Jian Zhang, Runwei Ding, Miaoju Ban, and Linhui Dai.* 

1. **MIAD: A maintenance inspection dataset for unsupervised anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/html/Bao_MIAD_A_Maintenance_Inspection_Dataset_for_Unsupervised_Anomaly_Detection_ICCVW_2023_paper.html)

   *Tianpeng Bao, Jiadong Chen, Wei Li, Xiang Wang, Jingjing Fei, Liwei Wu, Rui Zhao, and Ye Zheng.* 

1. **PAD: A dataset and benchmark for pose-agnostic anomaly detection.** NIPS, 2023. [paper](https://openreview.net/forum?id=kxFKgqwFNk)

   *Qiang Zhou, Weize Li, Lihan Jiang, Guoliang Wang, Guyue Zhou, Shanghang Zhang, and Hao Zhao.* 

1. **The voraus-AD dataset for anomaly detection in robot applications.** TRO, 2023. [paper](https://ieeexplore.ieee.org/document/10315239/)

   *Jan Thieß Brockmann, Marco Rudolph, Bodo Rosenhahn, and Bastian Wandt.* 

1. **A new comprehensive benchmark for semi-supervised video anomaly detection and anticipation.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.html)

   *Congqi Cao, Yue Lu, Peng Wang, and Yanning Zhang.* 

1. **Advancing anomaly detection: An adaptation model and a new dataset.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.04857)

   *Liyun Zhu, Arjun Raj, and Lei Wang.* 

1. **TimeSeriesBench: An industrial-grade benchmark for time series anomaly detection models.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.10802)

   *Haotian Si, Changhua Pei, Hang Cui, Jingwen Yang, Yongqian Sun, Shenglin Zhang, Jingjing Li, Haiming Zhang, Jing Han, Dan Pei, Jianhui Li, and Gaogang Xie.* 

1. **Towards fair graph anomaly detection: Problem, new datasets, and evaluation.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.15988)

   *Neng Kai Nigel Neo, Yeon-Chang Lee, Yiqiao Jin, Sang-Wook Kim, and Srijan Kumar.* 

1. **Real-IAD: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2403.12580v1)

   *Chengjie Wang, Wenbing Zhu, Binbin Gao, Zhenye Gan, Jianning Zhang, Zhihao Gu, Shuguang Qian, Mingang Chen, and Lizhuang Ma.* 

1. **MTMMC: A large-scale real-world multi-modal camera tracking benchmark.** CVPR, 2024. [paper](https://arxiv.org/abs/2403.20225)

   *Sanghyun Woo, Kwanyong Park, Inkyu Shin, Myungchul Kim, and In So Kweon.* 

1. **IPAD: Industrial process anomaly detection dataset.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.04857)

   *Jinfan Liu, Yichao Yan, Junjie Li, Weiming Zhao, Pengzhi Chu, Xingdong Sheng, Yunhui Liu, and Xiaokang Yang.* 

1. **Supervised anomaly detection for complex industrial images.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.04953)

   *Aimira Baitieva, David Hurych, Victor Besnier, and Olivier Bernard.* 

1. **ADer: A comprehensive benchmark for multi-class visual anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2406.03262)

   *Jiangning Zhang, Haoyang He, Zhenye Gan, Qingdong He, Yuxuan Cai, Zhucun Xue, Yabiao Wang, Chengjie Wang, Lei Xie, and Yong Liu.* 

### [Library](#content)
1. **ADBench: Anomaly detection benchmark.** NIPS, 2022. [paper](https://openreview.net/forum?id=foA_SFQ9zo0)

   *Songqiao Han, Xiyang Hu, Hailiang Huang, Minqi Jiang, and Yue Zhao.* 

1. **TSB-UAD: An end-to-end benchmark suite for univariate time-series anomaly detection.** VLDB, 2022. [paper](https://dl.acm.org/doi/abs/10.14778/3529337.3529354)

   *John Paparrizos, Yuhao Kang, Paul Boniol, Ruey S. Tsay, Themis Palpanas, and Michael J. Franklin.* 

1. **PyOD: A python toolbox for scalable outlier detection.** JMLR, 2019. [paper](https://www.jmlr.org/papers/v20/19-011.html)

   *Yue Zhao, Zain Nasrullah, and Zheng Li.* 

1. **OpenOOD: Benchmarking generalized out-of-distribution detection.** NIPS, 2022. [paper](https://arxiv.org/abs/2210.07242)

   *Jingkang Yang, Pengyun Wang, Dejian Zou, Zitang Zhou, Kunyuan Ding, Wenxuan Peng, Haoqi Wang, Guangyao Chen, Bo Li, Yiyou Sun, Xuefeng Du, Kaiyang Zhou, Wayne Zhang, Dan Hendrycks, Yixuan Li, and Ziwei Liu.* 

1. **Towards a rigorous evaluation of rime-series anomaly detection.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20680)

   *Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, and Sungroh Yoon.* 

1. **Volume under the surface: A new accuracy evaluation measure for time-series anomaly detection.** VLDB, 2022. [paper](https://dl.acm.org/doi/abs/10.14778/3551793.3551830)

   *John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsa, Aaron Elmore, and Michael J. Franklin.* 

1. **AnomalyKiTS: Anomaly detection toolkit for time series.** AAAI, 2020. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/21730)

   *Dhaval Patel, Giridhar Ganapavarapu, Srideepika Jayaraman, Shuxin Lin, Anuradha Bhamidipaty, and Jayant Kalagnanam.* 

1. **TODS: An automated time series outlier detection system.** AAAI, 2021. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/18012)

   *Kwei-Herng Lai, Daochen Zha, Guanchu Wang, Junjie Xu, Yue Zhao, Devesh Kumar, Yile Chen, Purav Zumkhawaka, Minyang Wan, Diego Martinez, and Xia Hu.* 

1. **BOND: Benchmarking unsupervised outlier node detection on static attributed graphs.** NIPS, 2022. [paper](https://openreview.net/forum?id=YXvGXEmtZ5N)

   *Kay Liu, Yingtong Dou, Yue Zhao, Xueying Ding, Xiyang Hu, Ruitong Zhang, Kaize Ding, Canyu Chen, Hao Peng, Kai Shu, Lichao Sun, Jundong Li, George H. Chen, Zhihao Jia, and Philip S. Yu.* 

1. **Ubnormal: New benchmark for supervised open-set video anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Acsintoae_UBnormal_New_Benchmark_for_Supervised_Open-Set_Video_Anomaly_Detection_CVPR_2022_paper.html)

   *Andra Acsintoae, Andrei Florescu, Mariana-Iuliana Georgescu, Tudor Mare, Paul Sumedrea, Radu Tudor Ionescu, Fahad Shahbaz Khan, and Mubarak Shah.* 

1. **A new comprehensive benchmark for semi-supervised video anomaly detection and anticipation.** CVPR, 2023. [paper](https://arxiv.org/abs/2305.13611)

   *Congqi Cao, Yue Lu, Peng Wang, and Yanning Zhang.* 

1. **A framework for benchmarking class-out-of-distribution detection and its application to ImageNet.** ICLR, 2023. [paper](https://openreview.net/forum?id=Iuubb9W6Jtk)

   *Ido Galil, Mohammed Dabbah, and Ran El-Yaniv.* 

1. **BMAD: Benchmarks for medical anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.11876)

   *Jinan Bao, Hanshi Sun, Hanqiu Deng, Yinsheng He, Zhaoxiang Zhang, and Xingyu Li.* 

1. **GADBench: Revisiting and benchmarking supervised Graph anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.12251)

   *Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, and Jia Li.* 

1. **A framework for benchmarking class-out-of-distribution detection and its application to ImageNet.** ICLR, 2023. [paper](https://arxiv.org/abs/2302.11893)

   *Ido Galil, Mohammed Dabbah, and Ran El-Yaniv.* 

1. **A large-scale benchmark for log parsing.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.10828)

   *Zhihan Jiang, Jinyang Liu, Junjie Huang, Yichen Li, Yintong Huo, Jiazhen Gu, Zhuangbin Chen, Jieming Zhu, and Michael R. Lyu.* 

1. **Making the end-user a priority in benchmarking: OrionBench for unsupervised time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.17748)

   *Sarah Alnegheimish, Laure Berti-Equille, and Kalyan Veeramachaneni.* 

1. **Towards scalable 3D anomaly detection and localization: A benchmark via 3D anomaly synthesis and a self-supervised learning network.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.14897)

   *Wenqiao Li and Xiaohao Xu.* 

1. **AUPIMO: Redefining visual anomaly detection benchmarks with high speed and low tolerance.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.01984)

   *Joao P. C. Bertoldo, Dick Ameln, Ashwin Vaidya, and Samet Akçay.* 

1. **MTAD: Tools and benchmarks for multivariate time series anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.06175)

   *Jinyang Liu, Wenwei Gu, Zhuangbin Chen, Yichen Li, Yuxin Su, and Michael R. Lyu.* 

### [Analysis](#content)
1. **Are we certain it’s anomalous?** arXiv, 2022. [paper](https://arxiv.org/abs/2211.09224)

   *Alessandro Flaborea, Bardh Prenkaj, Bharti Munjal, Marco Aurelio Sterpa, Dario Aragona, Luca Podo, and Fabio Galasso.* 

1. **Understanding anomaly detection with deep invertible networks through hierarchies of distributions and features.** NIPS, 2020. [paper](https://proceedings.neurips.cc/paper/2020/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html)

   *Robin Schirrmeister, Yuxuan Zhou, Tonio Ball, and Dan Zhang.* 

1. **Further analysis of outlier detection with deep generative models.** NIPS, 2018. [paper](http://proceedings.mlr.press/v137/wang20a.html)

   *Ziyu Wang, Bin Dai, David Wipf, and Jun Zhu.* 

1. **Learning temporal regularity in video sequences.** CVPR, 2016. [paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Hasan_Learning_Temporal_Regularity_CVPR_2016_paper.html)

   *Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K. Roy-Chowdhury, and Larry S. Davis.*

1. **Local evaluation of time series anomaly detection algorithms.** KDD, 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539339)

   *Alexis Huet, Jose Manuel Navarro, and Dario Rossi.* 

1. **Adaptive model pooling for online deep anomaly detection from a complex evolving data stream.** KDD, 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539348)

   *Susik Yoon, Youngjun Lee, Jae-Gil Lee, and Byung Suk Lee.* 

1. **Anomaly detection in time series: A comprehensive evaluation.** VLDB, 2022. [paper](https://dl.acm.org/doi/abs/10.14778/3538598.3538602)

   *Sebastian Schmidl, Phillip Wenig, and Thorsten Papenbrock.* 

1. **Anomaly detection requires better representations.** arXiv, 2022. [paper](https://arxiv.org/abs/2210.10773)

   *Tal Reiss, Niv Cohen, Eliahu Horwitz, Ron Abutbul, and Yedid Hoshen.* 

1. **Is it worth it? An experimental comparison of six deep and classical machine learning methods for unsupervised anomaly detection in time series.** arXiv, 2022. [paper](https://arxiv.org/abs/2212.11080)

   *Ferdinand Rewicki, Joachim Denzler, and Julia Niebling.* 

1. **FAPM: Fast adaptive patch memory for real-time industrial anomaly detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2210.07548)

   *Shinji Yamada, Satoshi Kamiya, and Kazuhiro Hotta.* 

1. **Detecting data errors: Where are we and what needs to be done?** VLDB, 2016. [paper](https://dl.acm.org/doi/10.14778/2994509.2994518)

   *Ziawasch Abedjan, Xu Chu, Dong Deng, Raul Castro Fernandez, Ihab F. Ilyas, Mourad Ouzzani, Paolo Papotti, Michael Stonebraker, and Nan Tang.* 

1. **Data cleaning: Overview and emerging challenges.** KDD, 2015. [paper](https://dl.acm.org/doi/10.1145/2882903.2912574)

   *Xu Chu, Ihab F. Ilyas, Sanjay Krishnan, and Jiannan Wang.* 

1. **Video anomaly detection by solving decoupled spatio-temporal Jigsaw puzzles.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_29)

   *uodong Wang, Yunhong Wang, Jie Qin, Dongming Zhang, Xiuguo Bao, and Di Huang.* 

1. **Learning causal temporal relation and feature discrimination for anomaly detection.** TIP, 2021. [paper](https://ieeexplore.ieee.org/document/9369126)

   *Peng Wu and Jing Liu.* 

1. **Unmasking the abnormal events in video.** ICCV, 2017. [paper](https://dl.acm.org/doi/10.1145/2882903.2912574)

   *Radu Tudor Ionescu, Sorina Smeureanu, Bogdan Alexe, and Marius Popescu.* 

1. **Temporal cycle-consistency learning.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.pdf)

   *Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman.* 

1. **Look at adjacent frames: Video anomaly detection without offline training.** ECCV, 2022. [paper](https://arxiv.org/abs/2207.13798)

   *Yuqi Ouyang, Guodong Shen, and Victor Sanchez.* 

1. **How to allocate your label budget? Choosing between active learning and learning to reject in anomaly detection.** AAAI, 2023. [paper](https://arxiv.org/abs/2301.02909)

   *Lorenzo Perini, Daniele Giannuzzi, and Jesse Davis.* 

1. **Deep anomaly detection under labeling budget constraints.** arXiv, 2023. [paper](https://arxiv.org/abs/2302.07832)

   *Aodong Li, Chen Qiu, Padhraic Smyth, Marius Kloft, Stephan Mandt, and Maja Rudolph.* 

1. **Diversity-measurable anomaly detection.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Diversity-Measurable_Anomaly_Detection_CVPR_2023_paper.html)

   *Wenrui Liu, Hong Chang, Bingpeng Ma, Shiguang Shan, and Xilin Chen.* 

1. **Transferring the contamination factor between anomaly detection domains by shape similarity.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20331)

   *Lorenzo Perini, Vincent Vercruyssen, and Jesse Davis.* 

1. **Are transformers effective for time series forecasting?** AAAI, 2023. [paper](https://arxiv.org/abs/2205.13504)

   *Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu.* 

1. **AnoRand: A semi supervised deep learning anomaly detection method by random labeling.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.18389)

   *Mansour Zoubeirou A Mayaki and Michel Riveill.* 

1. **AnoOnly: Semi-supervised anomaly detection without loss on normal data.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.18798)

   *Yixuan Zhou, Peiyu Yang, Yi Qu, Xing Xu, Fumin Shen, and Heng Tao Shen.* 

1. **No free lunch: The Hazards of over-expressive representations in anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.07284)

   *Tal Reiss, Niv Cohen, and Yedid Hoshen.* 

1. **Refining the optimization target for automatic univariate time series anomaly detection in monitoring services.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.10653)

   *Manqing Dong, Zhanxiang Zhao, Yitong Geng, Wentao Li, Wei Wang, and Huai Jiang.* 

1. **Beyond sharing: Conflict-aware multivariate time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.08915)

   *Haotian Si, Changhua Pei, Zhihan Li, Yadong Zhao, Jingjing Li, Haiming Zhang, Zulong Diao, Jianhui Li, Gaogang Xie, and Dan Pei.* 

1. **Neural network training strategy to enhance anomaly detection performance: A perspective on reconstruction loss amplification.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.14595)

   *YeongHyeon Park, Sungho Kang, Myung Jin Kim, Hyeonho Jeong, Hyunkyu Park, Hyeong Seok Kim, and Juneho Yi.* 

1. **Self-supervision for tackling unsupervised anomaly detection: Pitfalls and opportunities.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.14380)

   *Leman Akoglu and Jaemin Yoo.* 

1. **Tackling diverse minorities in imbalanced classification.** CIKM, 2023. [paper](https://arxiv.org/abs/2308.14838)

   *Kwei-Herng Lai, Daochen Zha, Huiyuan Chen, Mangesh Bendre, Yuzhong Chen, Mahashweta Das, Hao Yang, and Xia Hu.* 

1. **IOMatch: Simplifying open-set semi-supervised learning with joint inliers and outliers utilization.** ICCV, 2023. [paper](https://arxiv.org/abs/2308.13168)

   *Zekun Li, Lei Qi, Yinghuan Shi, and Yang Gao.* 

1. **Environment-biased feature ranking for novelty detection robustness.** ICCV, 2024. [paper](https://arxiv.org/abs/2309.12301)

   *Stefan Smeu, Elena Burceanu, Emanuela Haller, and Andrei Liviu Nicolicioiu.* 

1. **Going beyond familiar features for deep anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.00797)

   *Sarath Sivaprasad and Mario Fritz.* 

1. **Template-guided hierarchical feature restoration for anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Template-guided_Hierarchical_Feature_Restoration_for_Anomaly_Detection_ICCV_2023_paper.pdf)

   *Hewei Guo, Liping Ren, Jingjing Fu, Yuwang Wang, Zhizheng Zhang, Cuiling Lan, Haoqian Wang, and Xinwen Hou.* 

1. **Anomaly detection in the presence of irrelevant features.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.13057)

   *Marat Freytsis, Maxim Perelstein, and Yik Chuen San.* 

1. **BatchNorm-based weakly supervised video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.15367)

   *Yixuan Zhou, Yi Qu, Xing Xu, Fumin Shen, Jingkuan Song, and Hengtao Shen.* 

1. **ADT: Agent-based dynamic thresholding for anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01488)

   *Xue Yang, Enda Howley, and Micheal Schukat.* 

1. **PNI : Industrial anomaly detection using position and neighborhood information.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.html)

   *Jaehyeok Bae, Jae-Han Lee, and Seyun Kim.* 

1. **Estimating the contamination factor’s distribution in unsupervised anomaly detection.** ICML, 2022. [paper](https://proceedings.mlr.press/v202/perini23a.html)

   *Lorenzo Perini, Paul-Christian Bürkner, and Arto Klami.* 

1. **F1-EV Score: Measuring the likelihood of estimating a good decision threshold for semi-supervised anomaly detection.** ICASSP, 2024. [paper](https://arxiv.org/abs/2312.09143)

   *Kevin Wilkinghoff and Keisuke Imoto.* 

1. **Can untrained neural networks detect anomalies?** TII, 2024. [paper](https://ieeexplore.ieee.org/document/10391266)

   *Seunghyoung Ryu, Yonggyun Yu, and Hogeon Seo.* 

1. **Interleaving one-class and weakly-supervised models with adaptive thresholding for unsupervised video anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.13551)

   *Yongwei Nie, Hao Huang, Chengjiang Long, Qing Zhang, Pradipta Maji, and Hongmin Cai.* 

1. **Reimagining anomalies: What if anomalies were normal?** arXiv, 2024. [paper](https://arxiv.org/abs/2402.14469)

   *Philipp Liznerski, Saurabh Varshneya, Ece Calikus, Sophie Fellenz, and Marius Kloft.* 

1. **Long-tailed anomaly detection with learnable class names.** CVPR, 2024. [paper](https://arxiv.org/abs/2403.20236)

   *Chih-Hui Ho, Kuan-Chuan Peng, and Nuno Vasconcelos.* 

1. **Anomaly detection by context contrasting.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.18848)

   *Alain Ryser, Thomas M. Sutter, Alexander Marx, and Julia E. Vogt.* 

1. **From zero to hero: Cold-start anomaly detection.** ACL, 2024. [paper](https://arxiv.org/abs/2405.20341)

   *Tal Reiss, George Kour, Naama Zwerdling, Ateret Anaby-Tavor, and Yedid Hoshen.* 

### [Domain Adaptation](#content)
1. **Few-shot domain-adaptive anomaly detection for cross-site brain imagess.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/document/9606561)

   *Jianpo Su, Hui Shen, Limin Peng, and Dewen Hu.* 

1. **Registration based few-shot anomaly detection.** ECCV, 2021. [paper](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_18)

   *Chaoqin Huang, Haoyan Guan, Aofan Jiang, Ya Zhang, Michael Spratling, and Yanfeng Wang.*

1. **Learning unsupervised metaformer for anomaly detection.** CVPR, 2021. [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Learning_Unsupervised_Metaformer_for_Anomaly_Detection_ICCV_2021_paper.html)

   *Jhih-Ciang Wu, Dingjie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu.*

1. **Generic and scalable framework for automated time-series anomaly detection.** KDD, 2019. [paper](https://dl.acm.org/doi/10.1145/2783258.2788611)

   *Nikolay Laptev, Saeed Amizadeh, and Ian Flint.* 

1. **Transfer learning for anomaly detection through localized and unsupervised instance selection.** AAAI, 2020. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/6068)

   *Vincent Vercruyssen, Wannes Meert, and Jesse Davis.* 

1. **FewSome: Few shot anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2301.06957)

   *Niamh Belton, Misgina Tsighe Hagos, Aonghus Lawlor, and Kathleen M. Curran.* 

1. **Cross-domain video anomaly detection without target domain adaptation.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Aich_Cross-Domain_Video_Anomaly_Detection_Without_Target_Domain_Adaptation_WACV_2023_paper.html)

   *Abhishek Aich, Kuanchuan Peng, and Amit K. Roy-Chowdhury.* 

1. **Zero-shot anomaly detection without foundation models.** arXiv, 2023. [paper](https://arxiv.org/abs/2302.07849)

   *Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Maja Rudolph, and Stephan Mandt.* 

1. **Pushing the limits of fewshot anomaly detection in industry vision: A graphcore.** ICLR, 2023. [paper](https://openreview.net/forum?id=xzmqxHdZAwO)

   *Guoyang Xie, Jinbao Wang, Jiaqi Liu, Yaochu Jin, and Feng Zheng.* 

1. **Meta-learning for robust anomaly detection.** AISTATS, 2023. [paper](https://proceedings.mlr.press/v206/kumagai23a.html)

   *Atsutoshi Kumagai, Tomoharu Iwata, Hiroshi Takahashi, and Yasuhiro Fujiwara.* 

1. **OneShotSTL: One-shot seasonal-trend decomposition for online time series anomaly detection and forecasting.** arXiv, 2023. [paper](https://arxiv.org/abs/2304.01506)

   *Xiao He, Ye Li, Jian Tan, Bin Wu, and Feifei Li.* 

1. **Context-aware domain adaptation for time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2304.07453)

   *Kwei-Herng Lai, Lan Wang, Huiyuan Chen, Kaixiong Zhou, Fei Wang, Hao Yang, and Xia Hu.* 

1. **MetaGAD: Learning to meta transfer for few-shot graph anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.10668)

   *Xiongxiao Xu, Kaize Ding, Canyu Chen, and Kai Shu.* 

1. **A zero-/few-shot anomaly classification and segmentation method for CVPR 2023 VAND workshop challenge tracks 1&2: 1st place on zero-shot AD and 4th place on few-shot AD.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.17382)

   *Xuhai Chen, Yue Han, and Jiangning Zhang.* 

1. **Winning solution for the CVPR2023 visual anomaly and novelty detection challenge: Multimodal prompting for data-centric anomaly detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2306.09067)

   *Yunkang Cao, Xiaohao Xu, Chen Sun, Yuqi Cheng, Liang Gao, and Weiming Shen.* 

1. **Zero-shot anomaly detection with pre-trained segmentation models.** VAND, 2023. [paper](https://arxiv.org/abs/2306.09269)

   *Matthew Baugh, James Batten, Johanna P. Müller, and Bernhard Kainz.* 

1. **Optimizing PatchCore for few/many-shot anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.10792)

   *João Santos, Triet Tran, and Oliver Rippel.* 

1. **Multi-scale memory comparison for zero-/few-shot anomaly detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2308.04789)

   *Chaoqin Huang, Aofan Jiang, Ya Zhang, and Yanfeng Wang.* 

1. **AutoML for outlier detection with optimal Ttransport distances.** IJCAI, 2023. [paper](https://www.ijcai.org/proceedings/2023/843)

   *Prabhant Singh and Joaquin Vanschoren.*

1. **AutoML for outlier detection with optimal Ttransport distances.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)

   *Zheng Fang, Xiaoyang Wang, Haocheng Li, Jiejie Liu, Qiugui Hu, and Jimin Xiao.*

1. **Tight rates in supervised outlier transfer learning.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.04686)

   *Mohammadreza M. Kalan and Samory Kpotufe.*

1. **Few-shot anomaly detection with adversarial loss for robust feature representations.** BMVC, 2023. [paper](https://proceedings.bmvc2023.org/202/)

   *Jae Young Lee, Wonjun Lee, Jaehyun Choi, Yongkwi Lee, Young Seog Yoon, and Samory Kpotufe.*

1. **When model meets new normals: Test-time adaptation for unsupervised time-series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.11976)

   *Dongmin Kim, Sunghyun Park, and Jaegul Choo.*

1. **Few shot part segmentation reveals compositional logic for industrial anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.13783)

   *Soopil Kim, Sion An, Philip Chikontwe, Myeongkyun Kang, Ehsan Adeli, Kilian M. Pohl, and Sanghyun Park.*

1. **METER: A dynamic concept adaptation framework for online anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.16831)

   *Jiaqi Zhu, Shaofeng Cai, Fang Deng, Beng Chin Ooi, and Wenqiao Zhang.*

1. **Zero-shot versus many-shot: Unsupervised texture anomaly detection.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Aota_Zero-Shot_Versus_Many-Shot_Unsupervised_Texture_Anomaly_Detection_WACV_2023_paper.html)

   *Toshimichi Aota, Lloyd Teh Tzer Tong, and Takayuki Okatani.*

1. **Anomaly detection with domain adaptation.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Yang_Anomaly_Detection_With_Domain_Adaptation_CVPRW_2023_paper.html)

   *Ziyi Yang, Iman Soltani, and Eric Darve.*

1. **MuSc: Zero-shot anomaly classification and segmentation by mutual scoring of the unlabeled images.** ICLR, 2024. [paper](https://openreview.net/forum?id=AHgc5SMdtd)

   *Anonymous authors.* 

1. **Unified entropy optimization for open-set test-time adaptation.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.06065)

   *Zhengqing Gao, Xuyao Zhang, and Chenglin Liu.*

### [Loss Function](#content)
1. **Detecting regions of maximal divergence for spatio-temporal anomaly detection.** TPAMI, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8352745)

   *Björn Barz, Erik Rodner, Yanira Guanche Garcia, and Joachim Denzler.* 

1. **Convex formulation for learning from positive and unlabeled data.** ICML, 2015. [paper](https://dl.acm.org/doi/10.5555/3045118.3045266)

   *Marthinus Christoffel Du Plessis, Gang Niu, and Masashi Sugiyama.* 

1. **Anomaly detection with score distribution discrimination.** KDD, 2023. [paper](https://arxiv.org/abs/2306.14403)

   *Minqi Jiang, Songqiao Han, and Hailiang Huang.* 

1. **AdaFocal: Calibration-aware adaptive focal loss.** NIPS, 2022. [paper](https://openreview.net/forum?id=CoMOKHYWf2)

   *Arindam Ghosh, Arindam_Ghosh, and Thomas Schaaf, Matthew R. Gormley.* 

1. **DSV: An alignment validation loss for self-supervised outlier model selection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.06534)

   *Jaemin Yoo, Yue Zhao, Lingxiao Zhao, and Leman Akoglu.* 

1. **Simple and effective out-of-distribution detection via cosine-based softmax loss.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Noh_Simple_and_Effective_Out-of-Distribution_Detection_via_Cosine-based_Softmax_Loss_ICCV_2023_paper.pdf)

   *SoonCheol Noh, DongEon Jeong, and Jee-Hyong Lee.* 

1. **Temporal shift - multi-objective loss function for improved anomaly fall detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.02863)

   *Stefan Denkovski, Shehroz S. Khan, and Alex Mihailidis.* 

1. **MOODv2: Masked image modeling for out-of-distribution detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.02611)

   *Jingyao Li, Pengguang Chen, Shaozuo Yu, Shu Liu, and Jiaya Jia.* 

### [Model Selection](#content)
1. **Automatic unsupervised outlier model selection.** NIPS, 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html)

   *Yue Zhao, Ryan Rossi, and Leman Akoglu.*

1. **Toward unsupervised outlier model selection.** ICDM, 2022. [paper](https://www.andrew.cmu.edu/user/yuezhao2/papers/22-icdm-elect.pdf)

   *Yue Zhao, Sean Zhang, and Leman Akoglu.*

1. **Unsupervised model selection for time-series anomaly detection.** ICLR, 2023. [paper](https://openreview.net/forum?id=gOZ_pKANaPW)

   *Mononito Goswami, Cristian Ignacio Challu, Laurent Callot, Lenon Minorics, and Andrey Kan.* 

1. **Fast Unsupervised deep outlier model selection with hypernetworks.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.10529)

   *Xueying Ding, Yue Zhao, and Leman Akoglu.* 

1. **ADGym: Design choices for deep anomaly detection.** NIPS, 2023. [paper](https://openreview.net/forum?id=9CKx9SsSSc)

   *Minqi Jiang, Chaochuan Hou, Ao Zheng, Songqiao Han,Hailiang Huang, Qingsong Wen, Xiyang Hu, and Yue Zha.* 

1. **Model selection of anomaly detectors in the absence of labeled validation data.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.10461)

   *Clement Fung, Chen Qiu, Aodong Li, and Maja Rudolph.* 

1. **TransNAS-TSAD: Harnessing transformers for multi-objective neural architecture search in time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.18061)

   *Ijaz Ul Haq and Byung Suk Lee.*

1. **HyperMix: Out-of-distribution detection and classification in few-shot settings.** WACV, 2024. [paper](https://openaccess.thecvf.com/content/WACV2024/html/Mehta_HyperMix_Out-of-Distribution_Detection_and_Classification_in_Few-Shot_Settings_WACV_2024_paper.html)

   *Nikhil Mehta, Kevin J Liang, Jing Huang, Fujen Chu, Li Yin, and Tal Hassner.* 

### [Knowledge Distillation](#content)
1. **Anomaly detection via reverse distillation from one-class embedding.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html)

   *Hanqiu Deng and Xingyu Li.* 

1. **Multiresolution knowledge distillation for anomaly detection.** CVPR, 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Salehi_Multiresolution_Knowledge_Distillation_for_Anomaly_Detection_CVPR_2021_paper.html)

   *Mohammadreza Salehi, Niousha Sadjadi, Soroosh Baselizadeh, Mohammad H. Rohban, and Hamid R. Rabiee.* 

1. **Uninformed students: Student-teacher anomaly detection with discriminative latent embeddings.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html)

   *Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger.* 

1. **Reconstructed student-teacher and discriminative networks for anomaly detection.** IROS, 2022. [paper](https://arxiv.org/abs/2210.07548)

   *Shinji Yamada, Satoshi Kamiya, and Kazuhiro Hotta.* 

1. **Anomaly detection via reverse distillation from one-class embedding.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html)

   *Hanqiu Deng and Xingyu Li.* 

1. **DeSTSeg: Segmentation guided denoising student-teacher for anomaly detection.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.html)

   *Xuan Zhang, Shiyu Li, Xi Li, Ping Huang, Jiulong Shan, and Ting Chen.* 

1. **Asymmetric student-teacher networks for industrial anomaly detection.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.html)

   *Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt.* 

1. **In-painting radiography images for unsupervised anomaly detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2111.13495)

   *Tiange Xiang, Yongyi Lu, Alan L. Yuille, Chaoyi Zhang, Weidong Cai, and Zongwei Zhou.* 

1. **Self-distilled masked auto-encoders are efficient video anomaly detectors.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.12041)

   *Nicolae-Catalin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, and Mubarak Shah.* 

1. **Contextual affinity distillation for image anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.03101)

   *Jie Zhang, Masanori Suganuma, and Takayuki Okatani.* 

1. **Reinforcement learning by guided safe exploration.** ECAI, 2023. [paper](https://arxiv.org/abs/2307.14316)

   *Qisong Yang, Thiago D. Simão, Nils Jansen, Simon H. Tindemans, and Matthijs T. J. Spaan.* 

1. **Prior knowledge guided network for video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.01682)

   *Zhewen Deng, Dongyue Chen, and Shizhuo Deng.* 

1. **Asymmetric Student-Teacher Networks for Industrial Anomaly Detection.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.html)

   *Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt.* 

1. **Attention-conditioned augmentations for self-supervised anomaly detection and localization.** AAAI, 2023. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/26720)

   *Behzad Bozorgtabar and Dwarikanath Mahapatra.* 

1. **EfficientAD: Accurate visual anomaly detection at millisecond-level latencies.** WACV, 2024. [paper](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html)

   *Kilian Batzner, Lars Heckler, and Rebecca König.* 

1. **Remembering normality: Memory-guided knowledge distillation for unsupervised anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Gu_Remembering_Normality_Memory-guided_Knowledge_Distillation_for_Unsupervised_Anomaly_Detection_ICCV_2023_paper.html)

   *Zhihao Gu, Liang Liu, Xu Chen, Ran Yi, Jiangning Zhang, Yabiao Wang, Chengjie Wang, Annan Shu, Guannan Jiang, and Lizhuang Ma.* 

1. **Revisiting reverse distillation for anomaly detection.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.html)

   *Tran Dinh Tien, Anh Tuan Nguyen, Nguyen Hoang Tran, Ta Duc Huy, Soan T.M. Duong, Chanh D. Tr. Nguyen, and Steven Q. H. Truong.* 

1. **Dual-student knowledge distillation networks for unsupervised anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.00448)

   *Liyi Yao and Shaobing Gao.* 

1. **Structural teacher-student normality learning for multi-class anomaly detection and localization.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.17091)

   *Hanqiu Deng and Xingyu Li.* 

1. **Score distillation for anomaly detection.** KBS, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705124004763)

   *Jeongmin Hong and Seokho Kang.* 

### [Data Augmentation](#content)
1. **Interpretable, multidimensional, multimodal anomaly detection with negative sampling for detection of device failure.** ICML, 2020. [paper](https://proceedings.mlr.press/v119/sipple20a.html)

   *John Sipple.* 

1. **Doping: Generative data augmentation for unsupervised anomaly detection with GAN.** ICDM, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8594955)

   *Swee Kiat Lim, Yi Loo, Ngoc-Trung Tran, Ngai-Man Cheung, Gemma Roig, and Yuval Elovici.* 

1. **Detecting anomalies within time series using local neural transformations.** arXiv, 2022. [paper](https://arxiv.org/abs/2202.03944)

   *Tim Schneider, Chen Qiu, Marius Kloft, Decky Aspandi Latif, Steffen Staab, Stephan Mandt, and Maja Rudolph.* 

1. **Deep anomaly detection using geometric transformations.** NIPS, 2018. [paper](https://papers.nips.cc/paper/2018/hash/5e62d03aec0d17facfc5355dd90d441c-Abstract.html)

   *Izhak Golan and Ran El-Yaniv.* 

1. **Locally varying distance transform for unsupervised visual anomaly detection.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_21)

   *Wenyan Lin, Zhonghang Liu, and Siying Liu.* 

1. **DAGAD: Data augmentation for graph anomaly detection.** ICDM, 2022. [paper](https://arxiv.org/abs/2210.09766)

   *Fanzhen Liu, Xiaoxiao Ma, Jia Wu, Jian Yang, Shan Xue†, Amin Beheshti, Chuan Zhou, Hao Peng, Quan Z. Sheng, and Charu C. Aggarwal.*

1. **Unsupervised dimension-contribution-aware embeddings transformation for anomaly detection.** KBS, 2022. [paper](https://www.sciencedirect.com/science/article/pii/S0950705122013053)

   *Liang Xi, Chenchen Liang, Han Liu, and Ao Li.*

1. **No shifted augmentations (NSA): Compact distributions for robust self-supervised anomaly detection.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Yousef_No_Shifted_Augmentations_NSA_Compact_Distributions_for_Robust_Self-Supervised_Anomaly_WACV_2023_paper.html)

   *Mohamed Yousef, Marcel Ackermann, Unmesh Kurup, and Tom Bishop.*

1. **End-to-end augmentation hyperparameter tuning for self-supervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.12033)

   *Jaemin Yoo, Lingxiao Zhao, and Leman Akoglu.*

1. **Data augmentation is a hyperparameter: Cherry-picked self-supervision for unsupervised anomaly detection is creating the illusion of success.** TMLR, 2023. [paper](https://openreview.net/forum?id=HyzCuCV1jH)

   *Jaemin Yoo, Tiancheng Zhao, and Leman Akoglu.*

1. **Diverse data augmentation with diffusions for effective test-time prompt tuning.** ICCV, 2023. [paper](https://arxiv.org/abs/2308.06038)

   *Chunmei Feng, Kai Yu, Yong Liu, Salman Khan, and Wangmeng Zuo.*

1. **GraphPatcher: Mitigating degree bias for graph neural networks via test-time augmentation.** NIPS, 2023. [paper](https://neurips.cc/virtual/2023/poster/70390)

   *Mingxuan Ju, Tong Zhao, Wenhao Yu, Neil Shah, and Yanfang Ye.*

1. **Towards reliable AI model deployments: Multiple input mixup for out-of-distribution detection.** AAAI, 2024. [paper](https://arxiv.org/abs/2312.15514)

   *Dasol Choi and Dongbin Na.*

1. **Data augmentation for supervised graph outlier detection with latent diffusion models.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.17679)

   *Kay Liu, Hengrui Zhang, Ziqing Hu, Fangxin Wang, and Philip S. Yu.*

1. **No shifted augmentations (NSA): Compact distributions for robust self-supervised anomaly detection.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Yousef_No_Shifted_Augmentations_NSA_Compact_Distributions_for_Robust_Self-Supervised_Anomaly_WACV_2023_paper.html)

   *Mohamed Yousef, Marcel Ackermann, Unmesh Kurup, and Tom Bishop.*

1. **What makes a good data augmentation for few-shot unsupervised image anomaly detection?** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Zhang_What_Makes_a_Good_Data_Augmentation_for_Few-Shot_Unsupervised_Image_CVPRW_2023_paper.html)

   *Lingrui Zhang, Shuheng Zhang, Guoyang Xie, Jiaqi Liu, Hua Yan, Jinbao Wang, Feng Zheng, and Yaochu Jin.*

1. **Consistency training with learnable data augmentation for graph anomaly detection with limited supervision.** ICLR, 2024. [paper](https://openreview.net/forum?id=elMKXvhhQ9)

   *Anonymous authors.*

### [Outlier Exposure](#content)
1. **Latent outlier exposure for anomaly detection with contaminated data.** ICML, 2022. [paper](https://arxiv.org/abs/2202.08088)

   *Chen Qiu, Aodong Li, Marius Kloft, Maja Rudolph, and Stephan Mandt.*

1. **Deep anomaly detection with outlier exposure.** ICLR, 2019. [paper](https://openreview.net/forum?id=HyxCxhRcY7)

   *Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich.*

1. **A simple and effective baseline for out-of-distribution detection using abstention.** ICLR, 2021. [paper](https://openreview.net/forum?id=q_Q9MMGwSQu)

   *Sunil Thulasidasan, Sushil Thapa, Sayera Dhaubhadel, Gopinath Chennupati, Tanmoy Bhattacharya, and Jeff Bilmes.*

1. **Does your dermatology classifier know what it doesn’t know? Detecting the long-tail of unseen conditions.** Medical Image Analysis, 2022. [paper](https://www.sciencedirect.com/science/article/pii/S1361841521003194)

   *Abhijit Guha Roy, Jie Ren, Shekoofeh Azizi, Aaron Loh, Vivek Natarajan, Basil Mustafa, Nick Pawlowski, Jan Freyberg, Yuan Liu, Zach Beaver, Nam Vo, Peggy Bui, Samantha Winter, Patricia MacWilliams, Greg S. Corrado, Umesh Telang, Yun Liu, Taylan Cemgil, Alan Karthikesalingam, Balaji Lakshminarayanan, and Jim Winkens.*

1. **OpenMix: Exploring outlier samples for misclassification detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2303.17093)

   *Fei Zhu, Zhen Cheng, Xuyao Zhang, and Chenglin Liu.*

1. **VOS: Learning what you don't know by virtual outlier synthesis.** ICLR, 2023. [paper](https://openreview.net/forum?id=TW7d65uYu5M)

   *Xuefeng Du, Zhaoning Wang, Mu Cai, and Yixuan Li.*

1. **Deep anomaly detection under labeling budget constraints.** ICML, 2023. [paper](https://openreview.net/forum?id=VjopP4ejwB)

   *Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Stephan Mandt, and Maja Rudolph.*

1. **Pseudo outlier exposure for out-of-distribution detection using pretrained Transformers.** ACL, 2023. [paper](https://aclanthology.org/2023.findings-acl.95/)

   *Jaeyoung Kim, Kyuheon Jung, Dongbin Na, Sion Jang, Eunbin Park, and Sungchul Choi.*

1. **Harder synthetic anomalies to improve OOD detection in medical images.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.01412)

   *Sergio Naval Marimont and Giacomo Tarroni.*

1. **AutoLog: A log sequence synthesis framework for anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.09324)

   *Yintong Huo, Yichen Li, Yuxin Su, Pinjia He, Zifan Xie, and Michael R. Lyu.*

1. **Non-parametric outlier synthesis.** ICLR, 2023. [paper](https://openreview.net/forum?id=JHklpEZqduQ)

   *Leitian Tao, Xuefeng Du, Jerry Zhu, and Yixuan Li.* 

1. **Dream the impossible: Outlier imagination with diffusion models.** NIPS, 2023. [paper](https://openreview.net/forum?id=tnRboxQIec)

   *Xuefeng Du, Yiyou Sun, Xiaojin Zhu, and Yixuan Li.* 

1. **On the powerfulness of textual outlier exposure for visual OOD detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.16492)

   *Sangha Park, Jisoo Mok, Dahuin Jung, Saehyung Lee, and Sungroh Yoon.* 

1. **A coarse-to-fine pseudo-labeling (C2FPL) framework for unsupervised video anomaly detection.** WACV, 2024. [paper](https://arxiv.org/abs/2310.17650)

   *Anas Al-lahham, Nurbek Tastan, Zaigham Zaheer, and Karthik Nandakumar.* 

1. **Diversified outlier exposure for out-of-distribution detection via informative extrapolation.** NIPS, 2023. [paper](https://openreview.net/forum?id=RuxBLfiEqI)

   *Jianing Zhu, Geng Yu, Jiangchao Yao, Tongliang Liu, Gang Niu, Masashi Sugiyama, and Bo Han.* 

1. **Out-of-distribution detection learning with unreliable out-of-distribution sources.** NIPS, 2023. [paper](https://openreview.net/forum?id=RuxBLfiEqI)

   *Haotian Zheng, Qizhou Wang, Zhen Fang, Xiaobo Xia, Feng Liu, Tongliang Liu, and Bo Han.* 

1. **NNG-Mix: Improving semi-supervised anomaly detection with pseudo-anomaly generation.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.11961)

   *Hao Dong, Gaëtan Frusque, Yue Zhao, Eleni Chatzi, and Olga Fink.* 

1. **Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.html)

   *Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, and Ming-Hsuan Yang.* 

1. **Generating anomalies for video anomaly detection with prompt-based feature mapping.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Generating_Anomalies_for_Video_Anomaly_Detection_With_Prompt-Based_Feature_Mapping_CVPR_2023_paper.html)

   *Zuhao Liu, Xiaoming Wu, Dian Zheng, Kunyu Lin, and Weishi Zheng.* 

1. **Generating anomalies for video anomaly detection with prompt-based feature mapping.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Generating_Anomalies_for_Video_Anomaly_Detection_With_Prompt-Based_Feature_Mapping_CVPR_2023_paper.html)

   *Zuhao Liu, Xiaoming Wu, Dian Zheng, Kunyu Lin, and Weishi Zheng.* 

1. **Text-guided variational image generation for industrial anomaly detection and segmentation.** CVPR, 2024. [paper](https://arxiv.org/abs/2403.06247)

   *Mingyu Lee and Jongwon Choi.* 

1. **RealNet: A feature selection network with realistic synthetic anomaly for anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2403.05897)

   *Ximiao Zhang, Min Xu, and Xiuzhuang Zhou.* 

1. **Negative label guided OOD detection with pretrained vision-language models.** ICLR, 2024. [paper](https://openreview.net/forum?id=xUO1HXz4an)

   *Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, and Bo Han.* 

1. **Envisioning outlier exposure by large language models for out-of-distribution detection.** ICML, 2024. [paper](https://arxiv.org/abs/2406.00806)

   *Chentao Cao, Zhun Zhong, Zhanke Zhou, Yang Liu, Tongliang Liu, and Bo Han.* 

### [Contrastive Learning](#content)
1. **Graph anomaly detection via multi-scale contrastive learning networks with augmented view.** AAAI, 2023. [paper](https://arxiv.org/abs/2212.00535)

   *Jingcan Duan, Siwei Wang, Pei Zhang, En Zhu, Jingtao Hu, Hu Jin, Yue Liu, and Zhibin Dong.* 

1. **Partial and asymmetric contrastive learning for out-of-distribution detection in long-tailed recognition.** ICML, 2022. [paper](https://proceedings.mlr.press/v162/wang22aq.html)

   *Haotao Wang, Aston Zhang, Yi Zhu, Shuai Zheng, Mu Li, Alex Smola, and Zhangyang Wang.* 

1. **Focus your distribution: Coarse-to-fine non-contrastive learning for anomaly detection and localization.** ICME, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9859925)

   *Ye Zheng, Xiang Wang, Rui Deng, Tianpeng Bao, Rui Zhao, and Liwei Wu.* 

1. **MGFN: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2211.15098)

   *Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu.* 

1. **On the effectiveness of out-of-distribution data in self-supervised long-tail learning.** ICLR, 2023. [paper](https://openreview.net/forum?id=v8JIQdiN9Sh)

   *Jianhong Bai, Zuozhu Liu, Hualiang Wang, Jin Hao, Yang Feng, Huanpeng Chu, and Haoji Hu.*

1. **Hierarchical semantic contrast for scene-aware video anomaly detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2303.13051)

   *Shengyang Sun and Xiaojin Gong.*

1. **Hierarchical semi-supervised contrastive learning for contamination-resistant anomaly detection.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19806-9_7)

   *Gaoang Wang, Yibing Zhan, Xinchao Wang, Mingli Song, and Klara Nahrstedt.*

1. **Reconstruction enhanced multi-view contrastive learning for anomaly detection on attributed networks.** IJCAI, 2022. [paper](https://www.ijcai.org/proceedings/2022/0330)

   *Jiaqiang Zhang, Senzhang Wang, and Songcan Chen.*

1. **SimTS: Rethinking contrastive representation learning for time series forecasting.** arXiv, 2023. [paper](https://arxiv.org/abs/2303.18205)

   *Xiaochen Zheng, Xingyu Chen, Manuel Schürch, Amina Mollaysa, Ahmed Allam, and Michael Krauthammer.*

1. **CARLA: A self-supervised contrastive representation learning approach for time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.09296)

   *Zahra Zamanzadeh Darban, Geoffrey I. Webb, Shirui Pan, and Mahsa Salehi.*

1. **Unilaterally aggregated contrastive learning with hierarchical augmentation for anomaly detection.** ICCV, 2023. [paper](https://arxiv.org/abs/2308.10155)

   *Guodong Wang, Yunhong Wang, Jie Qin, Dongming Zhang, Xiuguo Bao, and Di Huang.*

1. **Cross-domain graph anomaly detection via anomaly-aware contrastive alignment.** AAAI, 2023. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25591)

   *Qizhou Wang, Guansong Pang, Mahsa Salehi, Wray Buntine, and Christopher Leckie.*

1. **Robust fraud detection via supervised contrastive learning.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.10055)

   *Vinay M.S., Shuhan Yuan, and Xintao Wu.*

1. **Understanding normalization in contrastive representation learning and out-of-distribution detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.15288)

   *Tai Le-Gia and Jaehyun Ahn.*

1. **Generating and reweighting dense contrastive patterns for unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.15911)

   *Songmin Dai, Yifan Wu, Xiaoqiang Li, and Xiangyang Xue.*

1. **Mean-shifted contrastive loss for anomaly detection.** AAAI, 2023. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25309)

   *Tal Reiss and Yedid Hoshen.*

1. **Hierarchical semantic contrast for scene-aware video anomaly detection.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Hierarchical_Semantic_Contrast_for_Scene-Aware_Video_Anomaly_Detection_CVPR_2023_paper.html)

   *Shengyang Sun and Xiaojin Gong.*

1. **Motif-aware Riemannian graph neural network with generative-contrastive learning.** AAAI, 2024. [paper](https://arxiv.org/abs/2401.01232)

   *Li Sun, Zhenhao Huang, Zixi Wang, Feiyang Wang, Hao Peng, and Philip Yu.* 

1. **UAC-AD: Unsupervised adversarial contrastive learning for anomaly detection on multi-modal data in microservice systems.** TSC, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10552111)

   *Hongyi Liu, Xiaosong Huang, Mengxi Jia, Tong Jia, Jing Han, Ying Li, and Zhonghai Wu.* 

1. **Model-guided contrastive fine-tuning for industrial anomaly detection.** CVPR, 2024. [paper](https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Artola_Model-guided_Contrastive_Fine-tuning_for_Industrial_Anomaly_Detection_CVPRW_2024_paper.html)

   *Aitor Artola, Yannis Kolodziej, Jean-Michel Morel, and Thibaud Ehret.* 

1. **Universal novelty detection through adaptive contrastive learning.** CVPR, 2024. [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Mirzaei_Universal_Novelty_Detection_Through_Adaptive_Contrastive_Learning_CVPR_2024_paper.html)

   *Hossein Mirzaei, Mojtaba Nafez, Mohammad Jafari, Mohammad Bagher Soltani, Mohammad Azizmalayeri, Jafar Habibi, Mohammad Sabokrou, and Mohammad Hossein Rohban.* 

### [Continual Learning](#content)
1. **PANDA: Adapting pretrained features for anomaly detection and segmentation.** CVPR, 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Reiss_PANDA_Adapting_Pretrained_Features_for_Anomaly_Detection_and_Segmentation_CVPR_2021_paper.html)

   *Tal Reiss, Niv Cohen, Liron Bergman, and Yedid Hoshen.* 

1. **Continual learning for anomaly detection in surveillance videos.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w15/Doshi_Continual_Learning_for_Anomaly_Detection_in_Surveillance_Videos_CVPRW_2020_paper.html)

   *Keval Doshi and Yasin Yilmaz.* 

1. **Rethinking video anomaly detection-A continual learning approach.** WACV, 2022. [paper](https://openaccess.thecvf.com/content/WACV2022/html/Doshi_Rethinking_Video_Anomaly_Detection_-_A_Continual_Learning_Approach_WACV_2022_paper.html)

   *Keval Doshi and Yasin Yilmaz.* 

1. **Continual learning for anomaly detection with variational autoencoder.** ICASSP, 2019. [paper](https://ieeexplore.ieee.org/abstract/document/8682702)

   *Felix Wiewel and Bin Yang.* 

1. **Lifelong anomaly detection through unlearning.** CCS, 2019. [paper](https://dl.acm.org/doi/10.1145/3319535.3363226)

   *Min Du, Zhi Chen, Chang Liu, Rajvardhan Oak, and Dawn Song.* 

1. **xStream: Outlier detection in feature-evolving data streams.** KDD, 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220107)

   *Emaad Manzoor, Hemank Lamba, and Leman Akoglu.* 

1. **Continual learning approaches for anomaly detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2212.11192)

   *Davide Dalle Pezze, Eugenia Anello, Chiara Masiero, and Gian Antonio Susto.* 

1. **Towards lightweight, model-agnostic and diversity-aware active anomaly detection.** ICLR, 2023. [paper](https://openreview.net/forum?id=-vKlt84fHs)

   *Xu Zhang, Yuan Zhao, Ziang Cui, Liqun Li, Shilin He, Qingwei Lin, Yingnong Dang, Saravan Rajmohan, and Dongmei Zhang.* 

1. **An iterative method for unsupervised robust anomaly detection under data contamination.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.09436)

   *Minkyung Kim, Jongmin Yu, Junsik Kim, Tae-Hyun Oh, and Jun Kyun Choi.* 

1. **Look at me, no replay! SurpriseNet: Anomaly detection inspired class incremental learning.** CIKM, 2023. [paper](https://dl.acm.org/doi/10.1145/3583780.3615236)

   *Anton Lee, Yaqian Zhang, Heitor Murilo Gomes, Albert Bifet, and Bernhard Pfahringer.* 

### [Active Learning](#content)
1. **DADMoE: Anomaly detection with mixture-of-experts from noisy labels.** AAAI, 2023. [paper](https://arxiv.org/abs/2208.11290)

   *Yue Zhao, Guoqing Zheng, Subhabrata Mukherjee, Robert McCann, and Ahmed Awadallah.* 

1. **Incorporating expert feedback into active anomaly discovery.** ICDM, 2016. [paper](https://ieeexplore.ieee.org/document/7837915)

   *Shubhomoy Das, Weng-Keen Wong, Thomas Dietterich, Alan Fern, and Andrew Emmott.* 

1. **Training ensembles with inliers and outliers for semi-supervised active learning.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.03741)

   *Vladan Stojnić, Zakaria Laskar, and Giorgos Tolias.* 

1. **Active anomaly detection based on deep one-class classification.** Pattern Recognition Letters, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865522003774)

   *Minkyung Kim, Junsik Kim, Jongmin Yu, and Jun Kyun Choi.* 

1. **Self-supervised anomaly detection via neural autoregressive flows with active learning.** NIPS, 2021. [paper](https://openreview.net/forum?id=LdWEo5mri6)

   *Jiaxin Zhang, Kyle Saleeby, Thomas Feldhausen, Sirui Bi, Alex Plotkowski, and David Womble.* 

1. **Multitask active learning for graph anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.13210)

   *Wenjing Chang, Kay Liu, Kaize Ding, Philip S. Yu, and Jianjun Yu.* 

### [Statistics](#content)
1. **(1+ε)-class classification: An anomaly detection method for highly imbalanced or incomplete data sets.** JMLR, 2021. [paper](https://dl.acm.org/doi/10.5555/3455716.3455788)

   *Maxim Borisyak, Artem Ryzhikov, Andrey Ustyuzhanin, Denis Derkach, Fedor Ratnikov, and Olga Mineeva.* 

1. **Deep semi-supervised anomaly detection.** ICLR, 2020. [paper](https://openreview.net/forum?id=HkgH0TEYwH)

   *Lukas Ruff, Robert A. Vandermeulen, Nico Görnitz, Alexander Binder, Emmanuel Müller, Klaus-Robert Müller, and Marius Kloft.* 

1. **Online learning and sequential anomaly detection in trajectories.** TPAMI, 2014. [paper](https://ieeexplore.ieee.org/document/6598676)

   *Rikard Laxhammar and Göran Falkman.* 

1. **COPOD: Copula-based outlier detection.** ICDM, 2020. [paper](https://ieeexplore.ieee.org/abstract/document/9338429)

   *Zheng Li, Yue Zhao, Nicola Botta, Cezar Ionescu, and Xiyang Hu.* 

1. **ECOD: Unsupervised outlier detection using empirical cumulative distribution functions.** TKDE, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9737003)

   *Zheng Li, Yue Zhao, Xiyang Hu, Nicola Botta, Cezar Ionescu, and George Chen.* 

1. **GLAD: A global-to-local anomaly detector.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Artola_GLAD_A_Global-to-Local_Anomaly_Detector_WACV_2023_paper.html)

   *Aitor Artola, Yannis Kolodziej, Jean-Michel Morel, and Thibaud Ehret.* 

1. **Anomaly dtection via Gumbel noise score matching.** arXiv, 2023. [paper](https://arxiv.org/abs/2304.03220)

   *Ahsan Mahmood, Junier Oliva, and Martin Styner.* 

1. **Unsupervised anomaly detection with rejection.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.13189)

   *Lorenzo Perini and Jesse Davis.* 

1. **A robust likelihood model for novelty detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2306.03331v1)

   *Ranya Almohsen, Shivang Patel, Donald A. Adjeroh, and Gianfranco Doretto.* 

1. **Spatially smoothed robust covariance estimation for local outlier detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.05371)

   *Patricia Puchhammer and Peter Filzmoser.* 

1. **Anomaly detection using score-based perturbation resilience.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Shin_Anomaly_Detection_using_Score-based_Perturbation_Resilience_ICCV_2023_paper.pdf)

   *Woosang Shin,Jonghyeon Lee, Taehan Lee, Sangmoon Lee, and Jong Pil Yun.* 

1. **Weighted subspace anomaly detection in high-dimensional space.** Pattern Recognition, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323007537)

   *Jiankai Tu, Huan Liu, and Chunguang Li.* 

1. **Mutual information maximization for semi-supervised anomaly detection.** KBS, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705123009462)

   *Shuo Liu and Maozai Tian.* 

1. **Sparse anomaly detection across referentials: A rank-based higher criticism approach.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.04924)

   *Ivo V. Stoepker, Rui M. Castro, and Ery Arias-Castro.* 

1. **Hyperbolic anomaly detection.** CVPR, 2024. [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hyperbolic_Anomaly_Detection_CVPR_2024_paper.html)

   *Huimin Li, Zhentao Chen, Yunhao Xu, and Junlin Hu.* 

### [Density Estimation](#content)
1. **DenseHybrid: Hybrid anomaly detection for dense open-set recognition.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19806-9_29)

   *Matej Grcić, Petra Bevandić., and Siniša Šegvić.* 

1. **Adaptive multi-stage density ratio estimation for learning latent space energy-based model.** NIPS, 2022. [paper](https://openreview.net/forum?id=kS5KG3mpSY)

   *Zhisheng Xiao, and Tian Han.* 

1. **Ultrafast local outlier detection from a data stream with stationary region skipping.** KDD, 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403171)

   *Susik Yoon, Jae-Gil Lee, and Byung Suk Lee.* 

1. **A discriminative framework for anomaly detection in large videos.** ECCV, 2016. [paper](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_21)

   *Allison Del Giorno, J. Andrew Bagnell, and Martial Hebert.* 

1. **Hierarchical density estimates for data clustering, visualization, and outlier detection.** ACM Transactions on Knowledge Discovery from Data, 2015. [paper](https://dl.acm.org/doi/10.1145/2733381)

   *Ricardo J. G. B. Campello, Davoud Moulavi, Arthur Zimek, and Jörg Sander.* 

1. **Unsupervised anomaly detection by robust density estimation.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20328)

   *Boyang Liu, Pangning Tan, and Jiayu Zhou.* 

1. **Understanding and mitigating data contamination in deep anomaly detection: A kernel-based approach.** IJCAI, 2022. [paper](https://www.ijcai.org/proceedings/2022/322)

   *Shuang Wu, Jingyu Zhao, and Guangjian Tian.* 

1. **Anomaly detection with variance stabilized density estimation.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.00582)

   *Amit Rozner, Barak Battash, Henry Li, Lior Wolf, and Ofir Lindenbaum.* 

1. **Beyond the benchmark: Detecting diverse anomalies in videos.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.01904)

   *Yoav Arad and Michael Werman.* 

1. **Quantile-based maximum likelihood training for outlier detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.06085)

   *Masoud Taghikhah, Nishant Kumar, Siniša Šegvić, Abouzar Eslami, and Stefan Gumhold.* 

1. **Unsupervised anomaly detection & diagnosis: A Stein variational gradient descent approach.** CIKM, 2023. [paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615167)

   *Zhichao Chen, Leilei Ding, Jianmin Huang, Zhixuan Chu, Qingyang Dai, and Hao Wang.* 

1. **Set features for anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.14773)

   *Niv Cohen, Issar Tzachor, and Yedid Hoshen.* 

1. **N-PAD: Neighboring pixel-based industrial anomaly detection.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Jang_N-Pad_Neighboring_Pixel-Based_Industrial_Anomaly_Detection_CVPRW_2023_paper.html)

   *JunKyu Jang, Eugene Hwang, and Sung-Hyuk Park.* 

1. **ConjNorm: Tractable density estimation for out-of-distribution detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=1pSL2cXWoz)

   *Bo Peng, Yadan Luo, Yonggang Zhang, Yixuan Li, and Zhen Fang.* 

1. **Dense projection for anomaly detection.** AAAI, 2024. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/28682)

   *Dazhi Fu, Zhao Zhang, nd Jicong Fan.* 

### [Support Vector](#content)
1. **Patch SVDD: Patch-level SVDD for anomaly detection and segmentation.** ACCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-69544-6_23)

   *Jihun Yi and Sungroh Yoon.* 

1. **Multiclass anomaly detector: The CS++ support vector machine.** JMLR, 2020. [paper](https://dl.acm.org/doi/10.5555/3455716.3455929)

   *Alistair Shilton, Sutharshan Rajasegarar, and Marimuthu Palaniswami.* 

1. **Timeseries anomaly detection using temporal hierarchical one-class network.** NIPS, 2020. [paper](https://proceedings.neurips.cc/paper/2020/hash/97e401a02082021fd24957f852e0e475-Abstract.html)

   *Lifeng Shen, Zhuocong Li, and James Kwok.* 

1. **LOSDD: Leave-out support vector data description for outlier detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2212.13626)

   *Daniel Boiar, Thomas Liebig, and Erich Schubert.* 

1. **Anomaly detection using one-class neural networks.** arXiv, 2018. [paper](https://arxiv.org/abs/1802.06360)

   *Raghavendra Chalapathy, Aditya Krishna Menon, and Sanjay Chawla.* 

1. **Deep graph stream SVDD: Anomaly detection in cyber-physical systems.** PAKDD, 2023. [paper](https://arxiv.org/pdf/2302.12918)

   *Ehtesamul Azim, Dongjie Wang, and Yanjie Fu.* 

1. **Regression-based hyperparameter learning for support vector machines.** TNNLS, 2023. [paper](https://ieeexplore.ieee.org/document/10286874)

   *Shili Peng, Wenwu Wang, Yinli Chen, Xueling Zhong, and Qinghua Hu.* 

### [Sparse Coding](#content)
1. **Video anomaly detection with sparse coding inspired deep neural networks.** TPAMI, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/8851288/)

   *Weixin Luo, Wen Liu, Dongze Lian, Jinhui Tang, Lixin Duan, Xi Peng, and Shenghua Gao.* 

1. **Self-supervised sparse representation for video anomaly detection.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_42)

   *Jhihciang Wu, Heyen Hsieh, Dingjie Chen, Chioushann Fuh, and Tyngluh Liu.* 

1. **Fast abnormal event detection.** IJCV, 2019. [paper](https://link.springer.com/article/10.1007/s11263-018-1129-8)

   *Cewu Lu, Jianping Shi, Weiming Wang, and Jiaya Jia.* 

1. **A revisit of sparse coding based anomaly detection in stacked RNN framework.** ICCV, 2017. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_42)

   *Weixin Luo, Wen Liu, and Shenghua Gao.* 

1. **HashNWalk: Hash and random walk based anomaly detection in hyperedge streams.** IJCAI, 2022. [paper](https://www.ijcai.org/proceedings/2022/0296.pdf)

   *Geon Lee, Minyoung Choe, and Kijung Shin.* 

### [Energy Model](#content)
1. **Deep structured energy based models for anomaly detection.** ICML, 2016. [paper](https://dl.acm.org/doi/10.5555/3045390.3045507)

   *Shuangfei Zhai, Yu Cheng, Weining Lu, and Zhongfei Zhang.* 

1. **Energy-based out-of-distribution detection.** NIPS, 2020. [paper](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)

   *Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li.* 

1. **Learning neural set functions under the optimal subset oracle.** NIPS, 2022. [paper](https://openreview.net/forum?id=GXOC0zL0ZI)

   *Zijing Ou, Tingyang Xu, Qinliang Su, Yingzhen Li, Peilin Zhao, and Yatao Bian.* 

1. **Energy-based out-of-distribution detection for graph neural networks.** ICLR, 2023. [paper](https://openreview.net/forum?id=zoz7Ze4STUL)

   *Qitian Wu, Yiting Chen, Chenxiao Yang, and Junchi Yan.* 

1. **Latent space energy-based model for fine-grained open set recognition.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.10711)

   *Wentao Bao, Qi Yu, and Yu Kong.* 

1. **Energy-based models for anomaly detection: A manifold diffusion recovery approach.** NIPS, 2023. [paper](https://openreview.net/forum?id=4nSDDokpfK)

   *Sangwoong Yoon, Young-Uk Jin, Yung-Kyun Noh, and Frank C. Park.* 

### [Memory Bank](#content)
1. **Towards total recall in industrial anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)

   *Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler.* 

1. **Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection.** ICCV, 2019. [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html)

   *Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel.* 

1. **SQUID: Deep feature in-painting for unsupervised anomaly detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2111.13495)

   *Tiange Xiang, Yixiao Zhang, Yongyi Lu, Alan L. Yuille, Chaoyi Zhang, Weidong Cai, and Zongwei Zhou.* 

1. **Shape-guided dual-memory learning for 3D anomaly detection.** ICML, 2023. [paper](https://openreview.net/forum?id=IkSGn9fcPz)

   *Yumin Chu, Liu Chieh, Ting-I Hsieh, Hwann-Tzong Chen, and Tyng-Luh Liu.* 

1. **That's BAD: Blind anomaly detection by implicit local feature clustering.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.03243)

   *Jie Zhang, Masanori Suganuma, and Takayuki Okatani.* 

### [Cluster](#content)
1. **MIDAS: Microcluster-based detector of anomalies in edge streams.** AAAI, 2020. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/5724)

   *Siddharth Bhatia, Bryan Hooi, Minji Yoon, Kijung Shin, and Christos Faloutsos.* 

1. **Multiple dynamic outlier-detection from a data stream by exploiting duality of data and queries.** SIGMOD, 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3448016.3452810)

   *Susik Yoon, Yooju Shin, Jae-Gil Lee, and Byung Suk Lee.* 

1. **Dynamic local aggregation network with adaptive clusterer for anomaly detection.** ECCV, 2022. [paper](https://dl.acm.org/doi/abs/10.1007/978-3-031-19772-7_24)

   *Zhiwei Yang, Peng Wu, Jing Liu, and Xiaotao Liu.* 

1. **Clustering and unsupervised anomaly detection with L2 normalized deep auto-encoder representations.** IJCNN, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8489068)

   *Caglar Aytekin, Xingyang Ni, Francesco Cricri, and Emre Aksu.*

1. **Clustering driven deep autoencoder for video anomaly detection.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_20)

   *Yunpeng Chang, Zhigang Tu, Wei Xie, and Junsong Yuan.*

1. **Cluster purging: Efficient outlier detection based on rate-distortion theory.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/document/9511218)

   *Maximilian B. Toller, Bernhard C. Geiger, and Roman Kern.*

1. **Outlier detection: How to Select k for k-nearest-neighbors-based outlier detectors.** Pattern Recognition Letters, 2023. [paper](https://ieeexplore.ieee.org/document/9511218)

   *Jiawei Yang, Xu Tan, and Sylwan Rahardja.*

1. **Improved outlier robust seeding for k-means.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.02710)

   *Amit Deshpande and Rameshwar Pratap.*

1. **Outlier detection using three-way neighborhood characteristic regions and corresponding fusion measurement.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10239460)

   *Xianyong Zhang, Zhong Yuan, and Duoqian Miao.*

1. **Autonomous anomaly detection for streaming data.** KBS, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705123009851)

   *Muhammad Yunus Bin Iqbal Basheer, Azliza Mohd Ali, Nurzeatul Hamimah Abdul Hamid, Muhammad Azizi Mohd Ariffin, Rozianawaty Osman, Sharifalillah Nordin, and Xiaowei Gu.*

1. **Bagged regularized k-distances for anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01046)

   *Yuchao Cai, Yuheng Ma, Hanfang Yang, and Hanyuan Hang.*

1. **Smoothing outlier scores is all you need to improve outlier detectors.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/document/10347456)

   *Jiawei Yang, Susanto Rahardja, and Pasi Fränti.*

1. **K-NNN: Nearest neighbors of neighbors for anomaly detection.** WACV, 2024. [paper](https://openaccess.thecvf.com/content/WACV2024W/ASTAD/html/Nizan_K-NNN_Nearest_Neighbors_of_Neighbors_for_Anomaly_Detection_WACVW_2024_paper.html)

   *Ori Nizan and Ayellet Tal.*

1. **Robust multi-kernel nearest neighborhood for outlier detection.** TKDE, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10433793)

   *Xinye Wang, Lei Duan, Zhenyang Yu, Chengxin He, and Zhifeng Bao.*

1. **Towards a unified framework of clustering-based anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2406.00452)

   *Zeyu Fang, Ming Gu, Sheng Zhou, Jiawei Chen, Qiaoyu Tan, Haishuai Wang, and Jiajun Bu.*

### [Isolation](#content)
1. **Isolation distributional kernel: A new tool for kernel based anomaly detection.** KDD, 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403062)

   *Kai Ming Ting, Bicun Xu, Takashi Washio, and Zhihua Zhou.* 

1. **AIDA: Analytic isolation and distance-based anomaly detection algorithm.** arXiv, 2022. [paper](https://arxiv.org/abs/2212.02645)

   *Luis Antonio Souto Arias, Cornelis W. Oosterlee, and Pasquale Cirillo.*

1. **OptIForest: Optimal isolation forest for anomaly detection.** IJCAI, 2023. [paper](https://arxiv.org/abs/2306.12703)

   *Haolong Xiang, Xuyun Zhang, Hongsheng Hu, Lianyong Qi, Wanchun Dou, Mark Dras, Amin Beheshti, and Xiaolong Xu.*

1. **Deep isolation forest for anomaly detection.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10108034)

   *Hongzuo Xu, Guansong Pang, Yijie Wang, and Yongjun Wang.* 

1. **Self-supervised random forest on transformed distribution for anomaly detection.** TNNLS, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10412688)

   *Jiabin Liu, Huadong Wang, Hanyuan Hang, Shumin Ma, Xin Shen, and Yong Shi.* 

### [Multi Modal](#content)
1. **Multimodal industrial anomaly detection via hybrid fusion.** CVPR, 2023. [paper](https://arxiv.org/abs/2303.00601)

   *Yue Wang, Jinlong Peng, Jiangning Zhang, Ran Yi, Yabiao Wang, and Chengjie Wang.*

1. **A multimodal anomaly detector for robot-assisted feeding using an LSTM-based variational autoencoder.** ICRA, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8279425)

   *Daehyung Park, Yuuna Hoshi, and Charles C. Kemp.* 

1. **EasyNet: An easy network for 3D industrial anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.13925)

   *Ruitao Chen, Guoyang Xie, Jiaqi Liu, Jinbao Wang, Ziqi Luo, Jinfan Wang, and Feng Zheng.* 

1. **ADMire++: Explainable anomaly detection in the human brain via inductive learning on temporal multiplex networks.** ICML, 2023. [paper](https://openreview.net/pdf?id=t4H8acYudJ)

   *Ali Behrouz and Margo Seltzer.* 

1. **Improving anomaly segmentation with multi-granularity cross-domain alignment.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.08696)

   *Ji Zhang, Xiao Wu, Zhi-Qi Cheng, Qi He, and Wei Li.* 

1. **SeMAnD: Self-supervised anomaly detection in multimodal geospatial datasets.** ACM SIGSPATIAL, 2023. [paper](https://arxiv.org/abs/2309.15245)

   *Daria Reshetova, Swetava Ganguli, C. V. Krishnakumar Iyer, and Vipul Pandey.* 

1. **Improving vision anomaly detection with the guidance of language modality.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.02821)

   *Dong Chen, Kaihang Pan, Guoming Wang, Yueting Zhuang, and Siliang Tang.*

1. **Debunking free fusion myth: Online multi-view anomaly detection with disentangled product-of-experts modeling.** MM, 2023. [paper](https://dl.acm.org/doi/abs/10.1145/3581783.3612487)

   *Hao Wang, Zhiqi Cheng, Jingdong Sun, Xin Yang, Xiao Wu, Hongyang Chen, and Yan Yang.*

1. **Multimodal industrial anomaly detection by crossmodal feature mapping.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.04521)

   *Alex Costanzino, Pierluigi Zama Ramirez, Giuseppe Lisanti, and Luigi Di Stefano.*

### [Optimal Transport](#content)
1. **Prototype-oriented unsupervised anomaly detection for multivariate time series.** ICML, 2023. [paper](https://openreview.net/forum?id=3vO4lS6PuF)

   *Yuxin Li, Wenchao Chen, Bo Chen, Dongsheng Wang, Long Tian, and Mingyuan Zhou.*

1. **Weakly supervised anomaly detection via knowledge-data alignment.** WWW, 2024. [paper](https://arxiv.org/abs/2402.03785)

   *Haihong Zhao, Chenyi Zi, Yang Liu, Chen Zhang, Yan Zhou, and Jia Li.*

### [Causal Inference](#content)
1. **Learning causality-inspired representation consistency for video anomaly detection.** ACM MM, 2023. [paper](https://arxiv.org/abs/2308.01537)

   *Yang Liu, Zhaoyang Xia, Mengyang Zhao, Donglai Wei, Yuzheng Wang, Liu Siao, Bobo Ju, Gaoyun Fang, Jing Liu, and Liang Song.*

### [Gaussian Process](#content)
1. **Deep anomaly detection with deviation networks.** KDD, 2019. [paper](https://dl.acm.org/doi/10.1145/3292500.3330871)

   *Guansong Pang, Chunhua Shen, and Anton van den Hengel.* 

1. **Video anomaly detection and localization using hierarchical feature representation and Gaussian process regression.** CVPR, 2015. [paper](https://ieeexplore.ieee.org/document/7298909)

   *Kai-Wen Cheng and Yie-Tarng Chen, and Wen-Hsien Fang.* 

1. **Multidimensional time series anomaly detection: A GRU-based Gaussian mixture variational autoencoder approach.** ACCV, 2018. [paper](http://proceedings.mlr.press/v95/guo18a.html)

   *Yifan Guo, Weixian Liao, Qianlong Wang, Lixing Yu, Tianxi Ji, and Pan Li.* 

1. **Gaussian process regression-based video anomaly detection and localization with hierarchical feature representation.** TIP, 2015. [paper](https://ieeexplore.ieee.org/abstract/document/7271067)

   *Kaiwen Cheng, Yie-Tarng Chen, and Wen-Hsien Fang.* 

1. **Adversarial anomaly detection using Gaussian priors and nonlinear anomaly scores.** ICDM, 2023. [paper](https://arxiv.org/abs/2310.18091)

   *Fiete Lüer, Tobias Weber, Maxim Dolgich, and Christian Böhm.*

1. **Invariant anomaly detection under distribution shifts: A causal perspective.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.14329)

   *João B. S. Carvalho, Mengtao Zhang, Robin Geyer, Carlos Cotrini, and Joachim M. Buhmann.*

### [Multi Task](#content)
1. **Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization.** IJCV, 2022. [paper](https://link.springer.com/article/10.1007/s11263-022-01578-9)

   *Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger.*

1. **Anomaly detection in video via self-supervised and multi-task learning.** CVPR, 2021. [paper](http://openaccess.thecvf.com/content/CVPR2021/html/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.html)

   *Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu, and Mubarak Shah.*

1. **Detecting semantic anomalies.** AAAI, 2020. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/5712)

   *Faruk Ahmed and Aaron Courville.*

1. **MGADN: A multi-task graph anomaly detection network for multivariate time series.** arXiv, 2022. [paper](https://arxiv.org/abs/2211.12141)

   *Weixuan Xiong and Xiaochen Sun.*

1. **Holistic representation learning for multitask trajectory anomaly detection.** WACV, 2023. [paper](https://arxiv.org/abs/2311.01851)

   *Alexandros Stergiou, Brent De Weerdt, and Nikos Deligiannis.*

1. **Multi-task learning based video anomaly detection with attention.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Baradaran_Multi-Task_Learning_Based_Video_Anomaly_Detection_With_Attention_CVPRW_2023_paper.html)

   *Mohammad Baradaran and Robert Bergevin.*

### [Interpretability](#content)
1. **Transparent anomaly detection via concept-based explanations.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.10702)

   *Laya Rafiee Sevyeri, Ivaxi Sheth, and Farhood Farahnak.*

1. **Towards self-interpretable graph-level anomaly detection.** NIPS, 2023. [paper](https://openreview.net/forum?id=SAzaC8f3cM)

   *Yixin Liu, Kaize Ding, Qinghua Lu, Fuyi Li, Leo Yu Zhang, and Shirui Pan.*

1. **Explainable anomaly detection using masked latent generative modeling.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.12550)

   *Daesoo Lee, Sara Malacarne, and Erlend Aune.*

1. **EVAL: Explainable video anomaly localization.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html)

   *Ashish Singh, Michael J. Jones, and Erik G. Learned-Miller.*

### [Open Set](#content)
1. **Anomaly heterogeneity learning for open-set supervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.12790)

   *Jiawen Zhu, Choubo Ding, Yu Tian, and Guansong Pang.*

1. **Open-set multivariate time-series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.12294)

   *Thomas Lai, Thi Kieu Khanh Ho, and Narges Armanfard.*

1. **SSB: Simple but strong baseline for boosting performance of open-set semi-supervised learning.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Fan_SSB_Simple_but_Strong_Baseline_for_Boosting_Performance_of_Open-Set_ICCV_2023_paper.html)

   *Yue Fan, Anna Kukleva, Dengxin Dai, and Bernt Schiele.*

### [Neural Process](#content)
1. **Semi-supervised anomaly detection via neural process.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10102264)

   *Fan Zhou, Guanyu Wang, Kunpeng Zhang, Siyuan Liu, and Ting Zhong.*

1. **Precursor-of-anomaly detection for irregular time series.** KDD, 2023. [paper](https://arxiv.org/abs/2306.15489)

   *Sheo Yon Jhin, Jaehoon Lee, and Noseong Park.*

1. **Pursuing feature separation based on neural collapse for out-of-distribution detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.17816)

   *Yingwen Wu, Ruiji Yu, Xinwen Cheng, Zhengbao He, and Xiaolin Huang.*

### [Nonparametric Approach](#content)
1. **Real-time nonparametric anomaly detection in high-dimensional settings.** TPAMI, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/8976215/)

   *Mehmet Necip Kurt, Yasin Yılmaz, and Xiaodong Wang.* 

1. **Neighborhood structure assisted non-negative matrix factorization and its application in unsupervised point anomaly detection.** JMLR, 2021. [paper](https://dl.acm.org/doi/abs/10.5555/3546258.3546292)

   *Imtiaz Ahmed, Xia Ben Hu, Mithun P. Acharya, and Yu Ding.* 

1. **Bayesian nonparametric submodular video partition for robust anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Sapkota_Bayesian_Nonparametric_Submodular_Video_Partition_for_Robust_Anomaly_Detection_CVPR_2022_paper.html)

   *Hitesh Sapkota and Qi Yu.* 

1. **Making parametric anomaly detection on tabular data non-parametric again.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.17052)

   *Hugo Thimonier, Fabrice Popineau, Arpad Rimmel, and Bich-Liên Doan.* 

### [Federated Learning](#content)
1. **FADngs: Federated learning for anomaly detection.** TNNLS, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10409269)

   *Boyu Dong, Dong Chen, Yu Wu, Siliang Tang, and Yueting Zhuang.*

1. **PeFAD: A parameter-efficient federated framework for time series anomaly detection.** SIGKDD, 2024. [paper](https://arxiv.org/abs/2406.02318)

   *Ronghui Xu, Hao Miao, Senzhang Wang, Philip S. Yu, and Jianxin Wang.*

1. **Weakly Supervised anomaly detection with privacy preservation under a bi-Level federated learning framework.** ESA, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417424013162)

   *Wei Guo and Pingyu Jiang.*


## [Application](#content)
### [Finance](#content)
1. **Antibenford subgraphs: Unsupervised anomaly detection in financial networks.** KDD, 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539100)

   *Tianyi Chen and E. Tsourakakis.* 

1. **Adversarial machine learning attacks against video anomaly detection systems.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022W/ArtOfRobust/html/Mumcu_Adversarial_Machine_Learning_Attacks_Against_Video_Anomaly_Detection_Systems_CVPRW_2022_paper.html)

   *Furkan Mumcu, Keval Doshi, and Yasin Yilmaz.* 

1. **Financial time series forecasting using CNN and Transformer.** AAAI, 2023. [paper](https://arxiv.org/abs/2304.04912)

   *Zhen Zeng, Rachneet Kaur, Suchetha Siddagangappa, Saba Rahimi, Tucker Balch, and Manuela Veloso.* 

1. **WAKE: A weakly supervised business process anomaly detection framework via a pre-trained autoencoder.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10285076)

   *Wei Guan, Jian Cao, Haiyan Zhao, Yang Gu, and Shiyou Qian.* 

1. **Probabilistic sampling-enhanced temporalspatial GCN: A scalable framework for transaction anomaly detection in Ethereum networks.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.00144)

   *Stefan Kambiz Behfar and Jon Crowcroft.* 

1. **Making the end-user a priority in benchmarking: OrionBench for unsupervised time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.17748)

   *Sarah Alnegheimish, Laure Berti-Equille, and Kalyan Veeramachaneni.* 

1. **Graph autoencoder anomaly detection for e-commerce application by contextual integrating contrast with reconstruction and complementarity.** TCE, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10388006)

   *Jinjie Wu, Zhipeng Qiu, Zhixia Zeng, Ruliang Xiao, Imad Rida, and Shi Zhang.* 

1. **Unsupervised anomaly detection on attributed networks with graph contrastive learning for consumer electronics security.** TCE, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10402014)

   *Bo Xu, Jinpeng Wang, Zhehuan Zhao, Hongfei Lin, and Feng Xia.* 

1. **Fin-Fed-OD: Federated outlier detection on financial tabular data.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.14933)

   *Dayananda Herurkar, Sebastian Palacio, Ahmed Anwar, Joern Hees, and Andreas Dengel.* 

1. **Advancing anomaly detection: Non-semantic financial data encoding with LLMs.** arXiv, 2024. [paper](https://arxiv.org/abs/2406.03614)

   *Alexander Bakumenko, Kateřina Hlaváčková-Schindler, Claudia Plant, and Nina C. Hubig.* 

1. **Automated financial time series anomaly detection via curiosity-guided exploration and self-imitation learning** EAAI, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197624008212)

   *Feifei Cao and Xitong Guo.* 


### [Point Cloud](#content)
1. **Teacher-student network for 3D point cloud anomaly detection with few normal samples.** arXiv, 2022. [paper](https://arxiv.org/abs/2210.17258)

   *Jianjian Qin, Chunzhi Gu, Jun Yu, and Chao Zhang.* 

1. **Teacher-student network for 3D point cloud anomaly detection with few normal samples.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Bergmann_Anomaly_Detection_in_3D_Point_Clouds_Using_Deep_Geometric_Descriptors_WACV_2023_paper.html)

   *Paul Bergmann and David Sattlegger.* 

1. **Anomaly detection in 3D point clouds using deep geometric descriptors.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Bergmann_Anomaly_Detection_in_3D_Point_Clouds_Using_Deep_Geometric_Descriptors_WACV_2023_paper.html)

   *Lokesh Veeramacheneni and Matias Valdenegro-Toro.* 

1. **Learning point-wise abstaining penalty for point cloud anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.10230)

   *Shaocong Xu, Pengfei Li, Xinyu Liu, Qianpu Sun, Yang Li, Shihui Guo, Zhen Wang, Bo Jiang, Rui Wang, Kehua Sheng, Bo Zhang, and Hao Zhao.* 

1. **Real3D-AD: A dataset of point cloud anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.13226)

   *Jiaqi Liu, Guoyang Xie, Ruitao Chen, Xinpeng Li, Jinbao Wang, Yong Liu, Chengjie Wang, and Feng Zheng.*

1. **Cheating depth: Enhancing 3D surface anomaly detection via depth simulation.** WACV, 2024. [paper](https://arxiv.org/abs/2311.01117)

   *Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj.*

1. **Image-pointcloud fusion based anomaly detection using PD-REAL dataset.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.04095)

   *Jianjian Qin, Chunzhi Gu, Jun Yu, and Chao Zhang.*

1. **Back to the feature: Classical 3D features are (almost) all you need for 3D anomaly detection.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Horwitz_Back_to_the_Feature_Classical_3D_Features_Are_Almost_All_CVPRW_2023_paper.html)

   *Eliahu Horwitz and Yedid Hoshen.*

1. **SplatPose & detect: Pose-agnostic 3D anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.06832)

   *Mathis Kruse, Marco Rudolph, Dominik Woiwode, and Bodo Rosenhahn.*

### [Autonomous Driving](#content)
1. **DeepSegmenter: Temporal action localization for detecting anomalies in untrimmed naturalistic driving videos.** arXiv, 2023. [paper](https://arxiv.org/abs/2304.08261)

   *Armstrong Aboah, Ulas Bagci, Abdul Rashid Mussah, Neema Jakisa Owor, and Yaw Adu-Gyamfi.*

1. **Multivariate time-series anomaly detection with temporal self-supervision and graphs: Application to vehicle failure prediction.** ECML PKDD, 2023. [paper](https://link.springer.com/chapter/10.1007/978-3-031-43430-3_15)

   *Hadi Hojjati, Mohammadreza Sadeghi, and Narges Armanfard.*

1. **Traffic anomaly detection: Exploiting temporal positioning of flow-density samples.** TITS, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10287217)

   *Iman Taheri Sarteshnizi, Saeed Asadi Bagloee, Majid Sarvi, and Neema Nassir.*

### [Medical Image](#content)
1. **SWSSL: Sliding window-based self-supervised learning for anomaly detection in high-resolution images.** TMI, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10247020)

   *Haoyu Dong, Yifan Zhang, Hanxue Gu, Nicholas Konz, Yixin Zhang, and Maciej A Mazurowskii.*

1. **A model-agnostic framework for universal anomaly detection of multi-organ and multi-modal images.** MICCAI, 2023. [paper](https://link.springer.com/chapter/10.1007/978-3-031-43898-1_23)

   *Yinghao Zhang, Donghuan Lu, Munan Ning, Liansheng Wang, Dong Wei, and Yefeng Zheng.*

1. **Dual conditioned diffusion models for out-of-distribution detection: Application to fetal ultrasound videos.** MICCAI, 2023. [paper](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_21)

   *Divyanshu Mishra, He Zhao, Pramit Saha, Aris T. Papageorghiou, and J. Alison Noble.*

1. **MAEDiff: Masked autoencoder-enhanced diffusion models for unsupervised anomaly detection in brain images.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.10561)

   *Rui Xu, Yunke Wang, and Bo Du.* 

1. **Domain adaptive and fine-grained anomaly detection for single-cell sequencing data and beyond.** IJCAI, 2024. [paper](https://arxiv.org/abs/2404.17454)

   *Kaichen Xu, Yueyang Ding, Suyang Hou, Weiqiang Zhan, Nisang Chen, Jun Wang, and Xiaobo Sun.* 

1. **Position-guided prompt learning for anomaly detection in chest X-rays.** MICCAI, 2024. [paper](https://arxiv.org/abs/2405.11976)

   *Zhichao Sun, Yuliang Gu, Yepeng Liu, Zerui Zhang, Zhou Zhao, and Yongchao Xu.* 

1. **MediCLIP: Adapting CLIP for few-shot medical image anomaly detection.** MICCAI, 2024. [paper](https://arxiv.org/abs/2405.11315)

   *Ximiao Zhang, Min Xu, Dehui Qiu, Ruixin Yan, Ning Lang, and Xiuzhuang Zhou.* 

1. **Spatial-aware attention generative adversarial network for semi-supervised anomaly detection in medical image.** MICCAI, 2024. [paper](https://arxiv.org/abs/2405.12872)

   *Zerui Zhang, Zhichao Sun, Zelong Liu, Bo Du, Rui Yu, Zhou Zhao, and Yongchao Xu.* 

### [Robotics](#content)
1. **Proactive anomaly detection for robot navigation with multi-sensor fusion.** RAL, 2023. [paper](https://ieeexplore.ieee.org/document/9720937)

   *Tianchen Ji, Arun Narenthiran Sivakumar, Girish Chowdhary, and Katherine Driggs-Campbell.* 

1. **Multi-channel anomaly detection for spacecraft time series using MAP estimation.** TAES, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10530457)

   *Tianyu Li, Sriram Baireddy, Mary Comer, Edward Delp, Sundip R. Desai, Richard H. Foster, and Moses W. Chan.* 

### [Cyber Intrusion](#content)
1. **Intrusion detection using convolutional neural networks for representation learning.** ICONIP, 2017. [paper](https://link.springer.com/chapter/10.1007/978-3-319-70139-4_87)

   *Hipeng Li, Zheng Qin, Kai Huang, Xiao Yang, and Shuxiong Ye.* 

1. **UMD: Unsupervised model detection for x2x backdoor attacks.** ICML, 2023. [paper](https://openreview.net/forum?id=t0ozPUGnBs)

   *Zhen Xiang, Zidi Xiong, and Bo Li.* 

1. **Kick bad guys out! Zero-knowledge-proof-based anomaly detection in federated learning.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.04055)

   *Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Yuhang Yao, Qifan Zhang, Salman Avestimehr, and Chaoyang He.* 

1. **Adaptive-correlation-aware unsupervised deep learning for anomaly detection in cyber-physical systems.** TDSC, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10265213)

   *Liang Xi, Dehua Miao, Menghan Li, Ruidong Wang, Han Liu, and Xunhua Huang.* 

1. **MTS-DVGAN: Anomaly detection in cyber-physical systems using a dual variational generative adversarial network.** Computers & Security, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167404823004807)

   *Haili Sun, Yan Huang, Lansheng Han, Cai Fu, Hongle Liu, and Xiang Long.* 

1. **Adversarial attacks against dynamic graph neural networks via node injection.** High-Confidence Computing, 2023. [paper](https://www.sciencedirect.com/science/article/pii/S2667295223000831)

   *Yanan Jiang and Hui Xia.* 

1. **Hybrid resampling and weighted majority voting for multi-class anomaly detection on imbalanced malware and network traffic data.** Engineering Applications of Artificial Intelligence, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197623017529)

   *Liang Xue and Tianqing Zhu.* 

### [Diagnosis](#content)
1. **Transformer-based normative modelling for anomaly detection of early schizophrenia.** NIPS, 2022. [paper](https://arxiv.org/abs/2212.04984)

   *Pedro F Da Costa, Jessica Dafflon, Sergio Leonardo Mendes, João Ricardo Sato, M. Jorge Cardoso, Robert Leech, Emily JH Jones, and Walter H.L. Pinaya.* 


### [High Performance Computing](#content)
1. **Anomaly detection using autoencoders in high performance computing systems.** IAAI, 2019. [paper](https://dl.acm.org/doi/10.1609/aaai.v33i01.33019428)

   *Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, and Luca Benini.* 

1. **MoniLog: An automated log-based anomaly detection system for cloud computing infrastructures.** ICDE, 2023. [paper](https://ieeexplore.ieee.org/document/9458872)

   *Arthur Vervaet.* 

1. **Self-supervised learning for anomaly detection in computational workflows.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.01247)

   *Hongwei Jin, Krishnan Raghavan, George Papadimitriou, Cong Wang, Anirban Mandal, Ewa Deelman, and Prasanna Balaprakash.* 

1. **Local outlier factor for anomaly detection in HPCC systems.** Journal of Parallel and Distributed Computing, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S074373152400087X)

   *Arya Adesh, Shobha G, Jyoti Shetty, and Lili Xu.* 

###[Physics](#content)
1. **Anomaly detection under coordinate transformations.** Physical Review D, 2023. [paper](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.015009)

   *Gregor Kasieczka, Radha Mastandrea, Vinicius Mikuni, Benjamin Nachman, Mariel Pettee, and David Shih.* 

1. **Back to the roots: Tree-based algorithms for weakly supervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.13111)

   *Thorben Finke, Marie Hein, Gregor Kasieczka, Michael Krämer, Alexander Mück, Parada Prangchaikul, Tobias Quadfasel, David Shih, and Manuel Sommerhalder.* 

1. **A physics-informed variational autoencoder for rapid galaxy inference and anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.16687)

   *Alexander Gagliano and V. Ashley Villar.* 

1. **Towards robust hyperspectral anomaly detection: Decomposing background, anomaly, and mixed noise via convex optimization.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.14814)

   *Koyo Sato and Shunsuke Ono.* 

1. **Detecting out-of-distribution earth observation images with diffusion models.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.12667)

   *Georges Le Bellier and Nicolas Audebert.* 

### [Industry Process](#content)
1. **In-situ anomaly detection in additive manufacturing with graph neural networks.** ICLR, 2023. [paper](https://arxiv.org/abs/2305.02695)

   *Sebastian Larsen and Paul A. Hooper.* 

1. **Knowledge distillation-empowered digital twin for anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.04616)

   *Qinghua Xu, Shaukat Ali, Tao Yue, Zaimovic Nedim, and Inderjeet Singh.* 

1. **Anomaly detection with memory-augmented adversarial autoencoder networks for industry 5.0.**TCE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10263623)

   *Huan Zhang, Neeraj Kumar, Sheng Wu, Chunlei Wu, Jian Wang, and Peiying Zhang.* 

1. **FDEPCA: A novel adaptive nonlinear feature extraction method via fruit fly olfactory neural network for iomt anomaly detection.** IEEE Journal of Biomedical and Health Informatics, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10262155)

   *Yihan Chen, Zhixia Zeng, Xinhong Lin, Xin Du, Imad Rida, and Ruliang Xiao.* 

1. **A discrepancy aware framework for robust anomaly detection.** TII, 2023. [paper](https://arxiv.org/abs/2310.07585)

   *Yuxuan Cai, Dingkang Liang, Dongliang Luo, Xinwei He, Xin Yang, and Xiang Bai.* 

1. **Anomaly detection with memory-augmented adversarial autoencoder networks for industry 5.0.** TCE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10263623)

   *Huan Zhang, Neeraj Kumar, Sheng Wu, Chunlei Wu, Jian Wang, and Peiying Zhang.* 

1. **Towards total online unsupervised anomaly detection and localization in industrial vision.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.15652)

   *Han Gao, Huiyuan Luo, Fei Shen, and Zhengtao Zhang.*

1. **Self-supervised variational graph autoencoder for system-level anomaly detection.** TIM, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10285620)

   *Le Zhang, Wei Cheng, Ji Xing, Xuefeng Chen, Zelin Nie, Shuo Zhang, Junying Hong, and Zhao Xu.*

1. **Distillation-based fabric anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.02287)

   *Simon Thomine and Hichem Snoussi.*

1. **Towards total online unsupervised anomaly detection and localization in industrial vision.** arXiv, 2024. [paper](https://arxiv.org/abs/2305.15652)

   *Han Gao, Huiyuan Luo, Fei Shen, and Zhengtao Zhang.*

1. **Adaptable and interpretable framework for anomaly detection in SCADA-based industrial systems.** ESA, 2024. [paper](https://www.sciencedirect.com/science/article/pii/S0957417424000654)

   *Marek Wadinger and Michal Kvasnica.*

1. **Graph structure change-based anomaly detection in multivariate time series of industrial processes.** TII, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10391270)

   *Zhen Zhang, Zhiqiang Geng, and Yongming Han.*

1. **A convolutional neural network approach for image-based anomaly detection in smart agriculture.** ESA, 2024. [paper](https://www.sciencedirect.com/science/article/pii/S0957417424000757)

   *José Mendoza-Bernal, Aurora González-Vidal, and Antonio F. Skarmeta.*

1. **Label-free anomaly detection in aerial agricultural images with masked image modeling.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.08931)

   *Sambal Shikhar and Anupam Sobti.*

1. **Prioritized local matching network for cross-category few-shot anomaly detection.** TAI, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10494062)

   *Huilin Deng, Hongchen Luo, Wei Zhai, Yang Cao, and Yu Kang.*

1. **Outlier-probability-based feature adaptation for robust unsupervised anomaly detection on contaminated training data.** TCSVT, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10542974)

   *Jianxiong Zhou and Ying Wu.*

### [Software](#content)
1. **GRAND: GAN-based software runtime anomaly detection method using trace information.** Neural Networks, 2023. [paper](https://www.sciencedirect.com/science/article/pii/S0893608023005919)

   *Shiyi Kong, Jun Ai, Minyan Lu, and Yiang Gong.*

1. **Log-based anomaly detection of enterprise software: An empirical study.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.20492)

   *Nadun Wijesinghe and Hadi Hemmati.*

1. **Efficiency of unsupervised anomaly detection methods on software logs.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01934)

   *Jesse Nyyssölä and Mika Mäntylä.*

1. **SpikeLog: Log-based anomaly detection via potential-assisted spiking neuron network.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10375739)

   *Jiaxing Qi, Zhongzhi Luan, Shaohan Huang, Carol Fung, Hailong Yang, and Depei Qian.*

1. **Hilogx: Noise-aware log-based anomaly detection with human feedback.**  The VLDB Journal, 2024. [paper](https://link.springer.com/article/10.1007/s00778-024-00843-2)

   *Tong Jia, Ying Li, Yong Yang, and Gang Huang.*

1. **Multivariate log-based anomaly detection for distributed database.**  KDD, 2024. [paper](https://arxiv.org/abs/2406.07976)

   *Lingzhe Zhang, Tong Jia, Mengxi Jia, Ying Li, Yong Yang, and Zhonghai Wu.*

### [Astronomy](#content)
1. **Multi-class deep SVDD: Anomaly detection approach in astronomy with distinct inlier categories.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.05011)

   *Pérez-Carrasco Manuel, Cabrera-Vives Guillermo, Hernández-García Lorena, Forster Francisco, Sánchez-Sáez Paula, Muñoz Arancibia Alejandra, Astorga Nicolás, Bauer Franz, Bayo Amelia, Cádiz-Leyton Martina, and Catelan Marcio.*

1. **GWAK: Gravitational-wave anomalous knowledge with recurrent autoencoders.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.11537)

   *Ryan Raikman, Eric A. Moreno, Ekaterina Govorkova, Ethan J Marx, Alec Gunny, William Benoit, Deep Chatterjee, Rafia Omer, Muhammed Saleem, Dylan S Rankin, Michael W Coughlin, Philip C Harris, and Erik Katsavounidis.* 