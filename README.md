# Deep-Learning-Based-Anomaly-Detection

***Anomaly Detection***: The process of detectingdata instances that ***significantly deviate*** from the majority of the whole dataset.

Contributed by Chunyang Zhang.

## [Content](#content)

<table>
<tr><td colspan="2"><a href="#survey-papers">1. Survey</a></td></tr> 
<tr><td colspan="2"><a href="#methods">2. Method</a></td></tr>
<tr>
    <td>&ensp;<a href="#autoencoder">2.1 AutoenEoder</a></td>
    <td>&ensp;<a href="#gan">2.2 GAN</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#flow">2.3 Flow</a></td>
    <td>&ensp;<a href="#diffusion-model">2.4 Diffusion Model</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#transformer">2.5 Transformer</a></td>
    <td>&ensp;<a href="#representation-learning">2.6 Representation Learninge</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#nonparametric-approach">2.7 Nonparametric Approach</a></td>
    <td>&ensp;<a href="#reinforcement-learning">2.8 Reinforcement Learning</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#cnn">2.9 CNN</a></td>
    <td>&ensp;<a href="#graph-network">2.10 Graph Neural Network</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#sparse-coding">2.11 Sparse Coding</a></td>
    <td>&ensp;<a href="#support-vector">2.12 Support Vector</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#ood">2.13 OOD</a></td>
    <td>&ensp;<a href="#novelty-detection">2.14 Novelty Detection</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#lstm">2.15 LSTM</a></td>
    <td>&ensp;<a href="#"></a></td>
</tr>
<tr><td colspan="2"><a href="#methods">3. Mechanism</a></td></tr>
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
    <td>&ensp;<a href="#lifelong-learning">3.6 Lifelong Learning</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#knowledge-distillation">3.7 Knowledge Distillation</a></td>
    <td>&ensp;<a href="#data-augmentation">3.8 Data Augmentation</a></td>
</tr>
<tr>
	<td>&ensp;<a href="#contrastive-learning">3.9 Contrastive Learning</a></td>
    <td>&ensp;<a href="#model-selection">3.10 Model-Selection</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#gaussian-process">3.11 Gaussian Process</a></td>
    <td>&ensp;<a href="#multi-task">3.12 Multi Task</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#outlier-exposure">3.13 Outlier Exposure</a></td>
    <td>&ensp;<a href="#density-estimation">3.14 Density Estimation</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#memory-bank">3.15 Memory Bank</a></td>
    <td>&ensp;<a href="#active-learning">3.16 Active Learning</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#cluster">3.17 Cluster</a></td>
    <td>&ensp;<a href="#isolation">3.18 Isolation</a></td>
</tr>
<tr><td colspan="2"><a href="#methods">4. Application</a></td></tr>
<tr>
    <td>&ensp;<a href="#finance">4.1 Finance</a></td>
    <td>&ensp;<a href="#point-cloud">4.2 Point Cloud</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#hpc">4.3 HPC</a></td>
    <td>&ensp;<a href="#intrusion">4.4 Intrusion</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#diagnosis">4.5 Diagnosis</a></td>
    <td>&ensp;<a href="#"></a></td>
</tr>
</table>







## [Survey Papers](#content)
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

1. **GAN-based anomaly detection: A review.** Neurocomputing, 2022. [paper](https://dl.acm.org/doi/10.1145/3453155)

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


## [Method](#content) 
### [AutoEncoder](#content)
1. **Graph regularized autoencoder and its application in unsupervised anomaly detection.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/document/9380495)

   *Imtiaz Ahmed, Travis Galoppo, Xia Hu, and Yu Ding.* 

1. **Innovations autoencoder and its application in one-class anomalous sequence detection.** JMLR, 2022. [paper](https://www.jmlr.org/papers/volume23/21-0735/21-0735.pdf)

   *Xinyi Wang and Lang Tong.* 

1. **Attention guided anomaly localization in images.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-58520-4_29)

   *Shashanka Venkataramanan, Kuan-Chuan Peng, Rajat Vikram Singh, and Abhijit Mahalanobis.* 

1. **Dynamic local aggregation network with adaptive clusterer for anomaly detection.** ECCV, 2022. [paper](https://dl.acm.org/doi/abs/10.1007/978-3-031-19772-7_24)

   *Zhiwei Yang, Peng Wu, Jing Liu, and Xiaotao Liu.* 

1. **Latent space autoregression for novelty detection.** CVPR, 2018. [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Abati_Latent_Space_Autoregression_for_Novelty_Detection_CVPR_2019_paper.html)

   *Davide Abati, Angelo Porrello, Simone Calderara, and Rita Cucchiara.*

1. **Anomaly detection in time series with robust variational quasi-recurrent autoencoders.** ICDM, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/9835268)

   *Tung Kieu, Bin Yang, Chenjuan Guo, Razvan-Gabriel Cirstea, Yan Zhao, Yale Song, and Christian S. Jensen.*

1. **Robust and explainable autoencoders for unsupervised time series outlier detection.** ICDE, 2022. [paper](https://ieeexplore.ieee.org/document/9835554)

   *Tung Kieu, Bin Yang, Chenjuan Guo, Christian S. Jensen, Yan Zhao, Feiteng Huang, and Kai Zheng.*

1. **Latent feature learning via autoencoder training for automatic classification configuration recommendation.** KBS, 2022. [paper](https://www.sciencedirect.com/science/article/pii/S0950705122013144)

   *Liping Deng and MingQing Xiao.*

1. **A multimodal anomaly detector for robot-assisted feeding using an LSTM-based variational autoencoder.** ICRA, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8279425)

   *Daehyung Park, Yuuna Hoshi, and Charles C. Kemp.* 

1. **Deep autoencoding Gaussian mixture model for unsupervised anomaly detection.** ICLR, 2018. [paper](https://openreview.net/forum?id=BJJLHbb0-)

   *Bo Zongy, Qi Songz, Martin Renqiang Miny, Wei Chengy, Cristian Lumezanuy, Daeki Choy, and Haifeng Chen.* 

1. **Anomaly detection with robust deep autoencoders.** KDD, 2017. [paper](https://dl.acm.org/doi/10.1145/3097983.3098052)

   *Chong Zhou and Randy C. Paffenroth.* 

1. **Unsupervised anomaly detection via variational auto-encoder for seasonal KPIs in web applications.** WWW, 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3178876.3185996)

   *Haowen Xu, Wenxiao Chen, Nengwen Zhao,Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan Pei, Yang Feng, Jie Chen, Zhaogang Wang, and Honglin Qiao.* 

1. **Spatio-temporal autoencoder for video anomaly detection.** MM, 2017. [paper](https://dl.acm.org/doi/abs/10.1145/3123266.3123451)

   *Yiru Zhao, Bing Deng, Chen Shen, Yao Liu, Hongtao Lu, and Xiansheng Hua.* 

1. **Deep structured energy based models for anomaly detection.** ICML, 2016. [paper](https://dl.acm.org/doi/10.5555/3045390.3045507)

   *Shuangfei Zhai, Yu Cheng, Weining Lu, and Zhongfei Zhang.* 

1. **Learning discriminative reconstructions for unsupervised outlier removal.** ICCV, 2015. [paper](https://ieeexplore.ieee.org/document/7410534)

   *Yan Xia, Xudong Cao, Fang Wen, Gang Hua, and Jian Sun.* 

1. **Outlier detection with autoencoder ensembles.** ICDM, 2017. [paper](https://research.ibm.com/publications/outlier-detection-with-autoencoder-ensembles)

   *Jinghui Chen, Saket Sathey, Charu Aggarwaly, and Deepak Turaga.*

1. **A study of deep convolutional auto-encoders for anomaly detection in videos.** Pattern Recognition Letters, 2018. [paper](https://www.sciencedirect.com/science/article/pii/S0167865517302489)

   *Manassés Ribeiro, AndréEugênio Lazzaretti, and Heitor Silvério Lopes.*

1. **Clustering and unsupervised anomaly detection with L2 normalized deep auto-encoder representations.** IJCNN, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8489068)

   *Caglar Aytekin, Xingyang Ni, Francesco Cricri, and Emre Aksu.*

1. **Classification-reconstruction learning for open-set recognition.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoshihashi_Classification-Reconstruction_Learning_for_Open-Set_Recognition_CVPR_2019_paper.pdf)

   *Ryota Yoshihashi, Shaodi You, Wen Shao, Makoto Iida, Rei Kawakami, and Takeshi Naemura.*

1. **Learning temporal regularity in video sequences.** CVPR, 2016. [paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Hasan_Learning_Temporal_Regularity_CVPR_2016_paper.html)

   *Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K. Roy-Chowdhury, and Larry S. Davis.*

1. **Clustering driven deep autoencoder for video anomaly detection.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_20)

   *Yunpeng Chang, Zhigang Tu, Wei Xie, and Junsong Yuan.*

1. **Making reconstruction-based method great again for video anomaly detection.** ICDM, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/10027694/)

   *Yizhou Wang, Can Qin, Yue Bai, Yi Xu, Xu Ma, and Yun Fu.*

1. **Two-stream decoder feature normality estimating network for industrial snomaly fetection.** ICASSP, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10027694/)

   *Chaewon Park, Minhyeok Lee, Suhwan Cho, Donghyeong Kim, and Sangyoun Lee.*

### [GAN](#content)
1. **Stabilizing adversarially learned one-class novelty detection using pseudo anomalies.** TIP, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9887825)

   *Muhammad Zaigham Zaheer, Jin-Ha Lee, Arif Mahmood, Marcella Astri, and Seung-Ik Lee.* 

1. **GAN ensemble for anomaly detection.** AAAI, 2021. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16530)

   *Han, Xu, Xiaohui Chen, and Li-Ping Liu.* 

1. **Generative cooperative learning for unsupervised video anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zaheer_Generative_Cooperative_Learning_for_Unsupervised_Video_Anomaly_Detection_CVPR_2022_paper.html)

   *Zaigham Zaheer, Arif Mahmood, M. Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee.* 

1. **GAN-based anomaly detection in imbalance problems.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-65414-6_11)

   *Junbong Kim, Kwanghee Jeong, Hyomin Choi, and Kisung Seo.* 

1. **Old is Gold: Redefining the adversarially learned one-class classifier training paradigm.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.html)

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

1. **GANomaly: Semi-supervised Anomaly Detection via Adversarial Training.** ACCV, 2019. [paper](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_39)

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

### [Diffusion Model](#content)
1. **AnoDDPM: Anomaly detection with denoising diffusion probabilistic models using simplex noise.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.html)

   *Julian Wyatt, Adam Leach, Sebastian M. Schmon, and Chris G. Willcocks.* 

1. **Diffusion models for medical anomaly detection.** MICCAI, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_4)

   *Julia Wolleb, Florentin Bieder, Robin Sandkühler, and Philippe C. Cattin.* 

### [Transformer](#content)
1. **Video anomaly detection via prediction network with enhanced spatio-temporal memory exchange.** ICASSP, 2022. [paper](https://ieeexplore.ieee.org/document/9747376)

   *Guodong Shen, Yuqi Ouyang, and Victor Sanchez.* 

1. **TranAD: Deep transformer networks for anomaly detection in multivariate time series data.** VLDB, 2022. [paper](https://dl.acm.org/doi/abs/10.14778/3514061.3514067)

   *Shreshth Tuli, Giuliano Casale, and Nicholas R. Jennings.* 

1. **Pixel-level anomaly detection via uncertainty-aware prototypical transformer.** MM, 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548082)

   *Chao Huang, Chengliang Liu, Zheng Zhang, Zhihao Wu, Jie Wen, Qiuping Jiang, and Yong Xu.* 

   **AddGraph: Anomaly detection in dynamic graph using attention-based temporal GCN.** IJCAI, 2019. [paper](https://www.ijcai.org/proceedings/2019/614)

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

   *Haiming Yao and Xue Wang,.* 

1. **VT-ADL: A vision transformer network for image anomaly detection and localization.** ISIE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9576231)

   *Pankaj Mishra, Riccardo Verk, Daniele Fornasier, Claudio Piciarelli, and Gian Luca Foresti.* 

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

### [Nonparametric Approach](#content)
1. **Real-time nonparametric anomaly detection in high-dimensional settings.** TPAMI, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/8976215/)

   *Mehmet Necip Kurt, Yasin Yılmaz, and Xiaodong Wang.* 

1. **Neighborhood structure assisted non-negative matrix factorization and its application in unsupervised point anomaly detection.** JMLR, 2021. [paper](https://dl.acm.org/doi/abs/10.5555/3546258.3546292)

   *Imtiaz Ahmed, Xia Ben Hu, Mithun P. Acharya, and Yu Ding.* 

1. **Bayesian nonparametric submodular video partition for robust anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Sapkota_Bayesian_Nonparametric_Submodular_Video_Partition_for_Robust_Anomaly_Detection_CVPR_2022_paper.html)

   *Hitesh Sapkota and Qi Yu.* 

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

### [CNN](#content)
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

### [Graph Neural Network](#content)
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

### [Sparse Coding](#content)
1. **Video anomaly detection with sparse coding inspired deep neural networks.** TPAMI, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/8851288/)

   *Weixin Luo, Wen Liu, Dongze Lian, Jinhui Tang, Lixin Duan, Xi Peng, and Shenghua Gao.* 

1. **Self-supervised sparse representation for video anomaly detection.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_42)

   *Jhihciang Wu, Heyen Hsieh, Dingjie Chen, Chioushann Fuh, and Tyngluh Liu.* 

1. **A revisit of sparse coding based anomaly detection in stacked RNN framework.** ICCV, 2017. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_42)

   *Weixin Luo, Wen Liu, and Shenghua Gao.* 

1. **HashNWalk: Hash and random walk based anomaly detection in hyperedge streams.** IJCAI, 2022. [paper](https://www.ijcai.org/proceedings/2022/0296.pdf)

   *Geon Lee, Minyoung Choe, and Kijung Shin.* 

1. **Fast abnormal event detection.** IJCV, 2019. [paper](https://link.springer.com/article/10.1007/s11263-018-1129-8)

   *Cewu Lu, Jianping Shi, Weiming Wang, and Jiaya Jia.* 

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

### [OOD](#content)
1. **Your out-of-distribution detection method is not robust!** NIPS, 2022. [paper](https://arxiv.org/abs/2209.15246)

   *Mohammad Azizmalayeri, Arshia Soltani Moakhar, Arman Zarei, Reihaneh Zohrabi, Mohammad Taghi Manzuri, and Mohammad Hossein Rohban.* 

1. **Exploiting mixed unlabeled data for detecting samples of seen and unseen out-of-distribution classes.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20814)

   *Yixuan Sun and Wei Wang.* 

1. **RankFeat: Rank-1 feature removal for out-of-distribution detection.** AAAI, 2022. [paper](https://arxiv.org/abs/2209.08590)

   *Yue Song, Nicu Sebe, and Wei Wang.* 

1. **Detect, distill and update: Learned DB systems facing out of distribution data.** SIGMOD, 2023. [paper](https://arxiv.org/abs/2210.05508)

   *Meghdad Kurmanji and Peter Triantafillou.* 

1. **Beyond mahalanobis distance for textual OOD detection.** NIPS, 2022. [paper](https://openreview.net/forum?id=ReB7CCByD6U)

   *Pierre Colombo, Eduardo Dadalto Câmara Gomes, Guillaume Staerman, Nathan Noiry, and Pablo Piantanida.* 

1. **Exploring the limits of out-of-distribution detection.** NIPS, 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/3941c4358616274ac2436eacf67fae05-Abstract.html)

   *Stanislav Fort, Jie Ren, and Balaji Lakshminarayanan.* 

1. **Is out-of-distribution detection learnable?** ICLR, 2022. [paper](https://openreview.net/forum?id=sde_7ZzGXOE)

   *Zhen Fang, Yixuan Li, Jie Lu, Jiahua Dong, Bo Han, and Feng Liu.* 

1. **Out-of-distribution detection is not all you need.** NIPS, 2022. [paper](https://openreview.net/forum?id=hxFth8JGGR4)

   *Joris Guerin, Kevin Delmas, Raul Sena Ferreira, and Jérémie Guiochet.* 

1. **iDECODe: In-distribution equivariance for conformal out-of-distribution detection.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20670)

   *Ramneet Kaur, Susmit Jha, Anirban Roy, Sangdon Park, Edgar Dobriban, Oleg Sokolsky, and Insup Lee.* 

1. **Out-of-distribution detection using an ensemble of self supervised leave-out classifiers.** ECCV, 2018. [paper](https://link.springer.com/chapter/10.1007/978-3-030-01237-3_34)

   *Apoorv Vyas, Nataraj Jammalamadaka, Xia Zhu, Dipankar Das, Bharat Kaul, and Theodore L. Willke.* 

1. **Self-supervised learning for generalizable out-of-distribution detection.** AAAI, 2020. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/5966)

   *Sina Mohseni, Mandar Pitale, JBS Yadawa, and Zhangyang ang.* 

1. **Energy-based out-of-distribution detection.** NIPS, 2020. [paper](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)

   *Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li.* 

1. **Augmenting softmax information for selective classification with out-of-distribution data.** ACCV, 2022. [paper](https://openaccess.thecvf.com/content/ACCV2022/html/Xia_Augmenting_Softmax_Information_for_Selective_Classification_with_Out-of-Distribution_Data_ACCV_2022_paper.html)

   *Guoxuan Xia and Christos-Savvas Bouganis.* 

1. **Robustness to spurious correlations improves semantic out-of-distribution detection.** AAAI, 2023. [paper](https://arxiv.org/abs/2302.04132)

   *Lily H. Zhang and Rajesh Ranganath.* 

### [Novelty Detection](#content)
1. **Semi-supervised novelty detection.** JMLR, 2010. [paper](https://www.jmlr.org/papers/v11/blanchard10a.html)

   *Gilles Blanchard, Gyemin Lee, and Clayton Scott.* 

### [LSTM](#content)
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


## [Mechanism](#content)
### [Dataset](#content)
1. **DoTA: Unsupervised detection of traffic anomaly in driving videos.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/document/9712446)

   *Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Yuchen Wang, Ella Atkins, and David Crandall.* 

1. **Revisiting time series outlier detection: Definitions and benchmarks.** NIPS, 2021. [paper](https://openreview.net/forum?id=r8IvOsnHchr)

   *Kwei-Herng Lai, Daochen Zha, Junjie Xu, Yue Zhao, Guanchu Wang, and Xia Hu.* 

1. **Street Scene: A new dataset and evaluation protocol for video anomaly detection.** WACV, 2020. [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf)

   *Bharathkumar Ramachandra and Michael J. Jones.*

1. **The eyecandies dataset for unsupervised multimodal anomaly detection and localization.** ACCV, 2020. [paper](https://openaccess.thecvf.com/content/ACCV2022/html/Bonfiglioli_The_Eyecandies_Dataset_for_Unsupervised_Multimodal_Anomaly_Detection_and_Localization_ACCV_2022_paper.html)

   *Luca Bonfiglioli, Marco Toschi, Davide Silvestri, Nicola Fioraio, and Daniele De Gregorio.*

1. **Not only look, but also listen: Learning multimodal violence detection under weak supervision.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_20)

   *Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang.* 

1. **A revisit of sparse coding based anomaly detection in stacked RNN framework.** ICCV, 2017. [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Luo_A_Revisit_of_ICCV_2017_paper.html)

   *Weixin Luo, Wen Liu, and Shenghua Gao.* 

1. **The MVTec anomaly detection dataset: A comprehensive real-world dataset for unsupervised anomaly detection.** IJCV, 2021. [paper](https://link.springer.com/article/10.1007/s11263-020-01400-4)

   *Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger.* 

1. **MVTec AD-A comprehensive real-world dataset for unsupervised anomaly detection.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html)

   *Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger.* 

1. **Anomaly detection in crowded scenes.** CVPR, 2010. [paper](https://ieeexplore.ieee.org/abstract/document/5539872/)

   *Vijay Mahadevan, Weixin Li, Viral Bhalodia, and Nuno Vasconcelos.* 

1. **Abnormal event detection at 150 FPS in MATLAB.** ICCV, 2013. [paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/html/Lu_Abnormal_Event_Detection_2013_ICCV_paper.html)

   *Cewu Lu, Jianping Shi, and Jiaya Jia.* 

1. **Surface defect saliency of magnetic tile.** The Visual Computer, 2020. [paper](https://link.springer.com/article/10.1007/s00371-018-1588-5)

   *Yibin Huang, Congying Qiu, and Kui Yuan.* 

### [Library](#content)
1. **ADBench: Anomaly detection benchmark.** NIPS, 2022. [paper](https://openreview.net/forum?id=foA_SFQ9zo0)

   *Songqiao Han, Xiyang Hu, Hailiang Huang, Minqi Jiang, and Yue Zhao.* 

1. **TSB-UAD: An end-to-end benchmark suite for univariate time-series anomaly detection.** VLDB, 2022. [paper](https://dl.acm.org/doi/abs/10.14778/3529337.3529354)

   *John Paparrizos, Yuhao Kang, Paul Boniol, Ruey S. Tsay, Themis Palpanas, and Michael J. Franklin.* 

1. **PyOD: A python toolbox for scalable outlier detection.** JMLR, 2019. [paper](https://www.jmlr.org/papers/v20/19-011.html)

   *Yue Zhao, Zain Nasrullah, and Zheng Li.* 

1. **OpenOOD: Benchmarking generalized out-of-distribution detection.** NIPS, 2022. [paper](https://arxiv.org/pdf/2210.07242.pdf)

   *Jingkang Yang, Pengyun Wang, Dejian Zou, Zitang Zhou, Kunyuan Ding, Wenxuan Peng, Haoqi Wang, Guangyao Chen, Bo Li, Yiyou Sun, Xuefeng Du,Kaiyang Zhou, Wayne Zhang, Dan Hendrycks, Yixuan Li, and Ziwei Liu.* 

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

### [Analysis](#content)
1. **Are we certain it’s anomalous?** arXiv, 2022. [paper](https://arxiv.org/pdf/2211.09224.pdf)

   *Alessandro Flaborea, Bardh Prenkaj, Bharti Munjal, Marco Aurelio Sterpa, Dario Aragona, Luca Podo, and Fabio Galasso.* 

1. **Understanding anomaly detection with deep invertible networks through hierarchies of distributions and features.** NIPS, 2020. [paper](https://proceedings.neurips.cc/paper/2020/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html)

   *Robin Schirrmeister, Yuxuan Zhou, Tonio Ball, and Dan Zhang.* 

1. **Further analysis of outlier detection with deep generative models.** NIPS, 2018. [paper](http://proceedings.mlr.press/v137/wang20a.html)

   *Ziyu Wang, Bin Dai, David Wipf, and Jun Zhu.* 

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

1. **FewSOME: Few shot anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2301.06957)

   *Niamh Belton, Misgina Tsighe Hagos, Aonghus Lawlor, and Kathleen M. Curran.* 

1. **Cross-domain video anomaly detection without target domain adaptation.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Aich_Cross-Domain_Video_Anomaly_Detection_Without_Target_Domain_Adaptation_WACV_2023_paper.html)

   *Abhishek Aich, Kuanchuan Peng, and Amit K. Roy-Chowdhury.* 

1. **Zero-shot anomaly detection without foundation models.** arXiv, 2023. [paper](https://arxiv.org/abs/2302.07849)

   *Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Maja Rudolph, and Stephan Mandt.* 

1. **Pushing the limits of fewshot anomaly detection in industry vision:  A Graphcore.** ICLR, 2023. [paper](https://openreview.net/forum?id=xzmqxHdZAwO)

   *Guoyang Xie, Jinbao Wang, Jiaqi Liu, Yaochu Jin, and Feng Zheng.* 

### [Loss Function](#content)
1. **Detecting regions of maximal divergence for spatio-temporal anomaly detection.** TPAMI, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8352745)

   *Björn Barz, Erik Rodner, Yanira Guanche Garcia, and Joachim Denzler.* 

1. **Convex formulation for learning from positive and unlabeled data.** ICML, 2015. [paper](https://dl.acm.org/doi/10.5555/3045118.3045266)

   *Marthinus Christoffel Du Plessis, Gang Niu, and Masashi Sugiyama.* 

### [Lifelong Learning](#content)
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

### [Knowledge Distillation](#content)
1. **Anomaly detection via reverse distillation from one-class embedding.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html)

   *Hanqiu Deng and Xingyu Li.* 

1. **Multiresolution knowledge distillation for anomaly detection.** CVPR, 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Salehi_Multiresolution_Knowledge_Distillation_for_Anomaly_Detection_CVPR_2021_paper.html)

   *Mohammadreza Salehi, Niousha Sadjadi, Soroosh Baselizadeh, Mohammad H. Rohban, and Hamid R. Rabiee.* 

1. **Uninformed students: Student-teacher anomaly detection with discriminative latent embeddings.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html)

   *Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger.* 

1. **Reconstructed student-teacher and discriminative networks for anomaly detection.** IROS, 2022. [paper](https://arxiv.org/abs/2210.07548)

   *Shinji Yamada, Satoshi Kamiya, and Kazuhiro Hotta.* 

1. **DeSTSeg: Segmentation guided denoising student-teacher for anomaly detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2211.11317)

   *Xuan Zhang, Shiyu Li, Xi Li, Ping Huang, Jiulong Shan, and Ting Chen.* 

1. **Asymmetric student-teacher networks for industrial anomaly detection.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.html)

   *Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt.* 

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

1. **No shifted augmentations (NSA): Compact distributions for robust self-supervised Anomaly Detection.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Yousef_No_Shifted_Augmentations_NSA_Compact_Distributions_for_Robust_Self-Supervised_Anomaly_WACV_2023_paper.html)

   *Mohamed Yousef, Marcel Ackermann, Unmesh Kurup, and Tom Bishop.*

### [Contrastive Learning](#content)
1. **Graph anomaly detection via multi-scale contrastive learning networks with augmented view.** AAAI, 2023. [paper](https://arxiv.org/abs/2212.00535)

   *Jingcan Duan, Siwei Wang, Pei Zhang, En Zhu, Jingtao Hu, Hu Jin, Yue Liu, and Zhibin Dong.* 

1. **Partial and asymmetric contrastive learning for out-of-distribution detection in long-tailed recognition.** ICML, 2022. [paper](https://proceedings.mlr.press/v162/wang22aq.html)

   *Haotao Wang, Aston Zhang, Yi Zhu, Shuai Zheng, Mu Li, Alex Smola, and Zhangyang Wang.* 

1. **Focus your distribution: Coarse-to-fine non-contrastive learning for anomaly detection and localization.** ICME, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9859925)

   *Ye Zheng, Xiang Wang, Rui Deng, Tianpeng Bao, Rui Zhao, and Liwei Wu.* 

1. **MGFN: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2211.15098)

   *Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu.* 

### [Model Selection](#content)
1. **Automatic unsupervised outlier model selection.** NIPS, 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html)

   *Yue Zhao, Ryan Rossi, and Leman Akoglu.*

1. **Toward unsupervised outlier model selection.** ICDM, 2022. [paper](https://www.andrew.cmu.edu/user/yuezhao2/papers/22-icdm-elect.pdf)

   *Yue Zhao, Sean Zhang, and Leman Akoglu.*

1. **Unsupervised model selection for time-series anomaly detection.** ICLR, 2023. [paper](https://openreview.net/forum?id=gOZ_pKANaPW)

   *Mononito Goswami, Cristian Ignacio Challu, Laurent Callot, Lenon Minorics, and Andrey Kan.* 

### [Gaussian Process](#content)
1. **Deep anomaly detection with deviation networks.** KDD, 2019. [paper](https://dl.acm.org/doi/10.1145/3292500.3330871)

   *Guansong Pang, Chunhua Shen, and Anton van den Hengel.* 

1. **Video anomaly detection and localization using hierarchical feature representation and Gaussian process regression.** CVPR, 2015. [paper](https://ieeexplore.ieee.org/document/7298909)

   *Kai-Wen Cheng and Yie-Tarng Chen, and Wen-Hsien Fang.* 

1. **Multidimensional time series anomaly detection: A GRU-based Gaussian mixture variational autoencoder approach.** ACCV, 2018. [paper](http://proceedings.mlr.press/v95/guo18a.html)

   *Yifan Guo, Weixian Liao, Qianlong Wang, Lixing Yu, Tianxi Ji, and Pan Li.* 

1. **Gaussian process regression-based video anomaly detection and localization with hierarchical feature representation.** TIP, 2015. [paper](https://ieeexplore.ieee.org/abstract/document/7271067)

   *Kaiwen Cheng, Yie-Tarng Chen, and Wen-Hsien Fang.* 

### [Multi Task](#content)
1. **Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization.** IJCV, 2022. [paper](https://link.springer.com/article/10.1007/s11263-022-01578-9)

   *Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger.*

1. **Anomaly detection in video via self-supervised and multi-task learning.** CVPR, 2021. [paper](http://openaccess.thecvf.com/content/CVPR2021/html/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.html)

   *Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu, and Mubarak Shah.*

1. **Detecting semantic anomalies.** AAAI, 2020. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/5712)

   *Faruk Ahmed and Aaron Courville.*

1. **MGADN: A multi-task graph anomaly detection network for multivariate time series.** arXiv, 2022. [paper](https://arxiv.org/abs/2211.12141)

   *Weixuan Xiong and Xiaochen Sun.*

### [Outlier Exposure](#content)
1. **Latent outlier exposure for anomaly detection with contaminated data.** ICML, 2022. [paper](https://arxiv.org/abs/2202.08088)

   *Chen Qiu, Aodong Li, Marius Kloft, Maja Rudolph, and Stephan Mandt.*

1. **Deep anomaly detection with outlier exposure.** ICLR, 2019. [paper](https://openreview.net/forum?id=HyxCxhRcY7)

   *Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich.*

1. **A simple and effective baseline for out-of-distribution detection using abstention.** ICLR, 2021. [paper](https://openreview.net/forum?id=q_Q9MMGwSQu)

   *Sunil Thulasidasan, Sushil Thapa, Sayera Dhaubhadel, Gopinath Chennupati, Tanmoy Bhattacharya, and Jeff Bilmes.*

1. **Does your dermatology classifier know what it doesn’t know? Detecting the long-tail of unseen conditions.** Medical Image Analysis, 2022. [paper](https://www.sciencedirect.com/science/article/pii/S1361841521003194)

   *Abhijit Guha Roy, Jie Ren, Shekoofeh Azizi, Aaron Loh, Vivek Natarajan, Basil Mustafa, Nick Pawlowski, Jan Freyberg, Yuan Liu, Zach Beaver, Nam Vo, Peggy Bui, Samantha Winter, Patricia MacWilliams, Greg S. Corrado, Umesh Telang, Yun Liu, Taylan Cemgil, Alan Karthikesalingam, Balaji Lakshminarayanan, and Jim Winkens.*

### [Statistics](#content)
1. **(1+ε)-class Classification: An anomaly detection method for highly imbalanced or incomplete data sets.** JMLR, 2021. [paper](https://dl.acm.org/doi/10.5555/3455716.3455788)

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

### [Memory Bank](#content)
1. **Towards total recall in industrial anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)

   *Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler.* 

1. **Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection.** ICCV, 2019. [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html)

   *Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel.* 

### [Active Learning](#content)
1. **DADMoE: Anomaly detection with mixture-of-experts from noisy labels.** AAAI, 2023. [paper](https://arxiv.org/abs/2208.11290)

   *Yue Zhao, Guoqing Zheng, Subhabrata Mukherjee, Robert McCann, and Ahmed Awadallah.* 

1. **Incorporating expert feedback into active anomaly discovery.** ICDM, 2016. [paper](https://ieeexplore.ieee.org/document/7837915)

   *Shubhomoy Das, Weng-Keen Wong, Thomas Dietterich, Alan Fern, and Andrew Emmott.* 

### [Cluster](#content)
1. **MIDAS: Microcluster-based detector of anomalies in edge streams.** AAAI, 2020. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/5724)

   *Siddharth Bhatia, Bryan Hooi, Minji Yoon, Kijung Shin, and Christos Faloutsos.* 

1. **Multiple dynamic outlier-detection from a data stream by exploiting duality of data and queries.** SIGMOD, 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3448016.3452810)

   *Susik Yoon, Yooju Shin, Jae-Gil Lee, and Byung Suk Lee.* 

### [Isolation](#content)
1. **Isolation distributional kernel: A new tool for kernel based anomaly detection.** KDD, 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403062)

   *Kai Ming Ting, Bicun Xu, Takashi Washio, and Zhihua Zhou.* 

1. **AIDA: Analytic isolation and distance-based anomaly detection algorithm.** arXiv, 2022. [paper](https://arxiv.org/abs/2212.02645)

   *Luis Antonio Souto Arias, Cornelis W. Oosterlee, and Pasquale Cirillo.* 


## [Application](#content)
### [Finance](#content)
1. **Antibenford subgraphs: Unsupervised anomaly detection in financial networks.** KDD, 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539100)

   *Tianyi Chen and E. Tsourakakis* 

### [Point Cloud](#content)
1. **Teacher-student network for 3D point cloud anomaly detection with few normal samples.** arXiv, 2022. [paper](https://arxiv.org/abs/2210.17258)

   *Jianjian Qin, Chunzhi Gu, Jun Yu, and Chao Zhang.* 

1. **Teacher-student network for 3D point cloud anomaly detection with few normal samples.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Bergmann_Anomaly_Detection_in_3D_Point_Clouds_Using_Deep_Geometric_Descriptors_WACV_2023_paper.html)

   *Paul Bergmann and David Sattlegger.* 

1. **Anomaly Detection in 3D Point Clouds Using Deep Geometric Descriptors.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Bergmann_Anomaly_Detection_in_3D_Point_Clouds_Using_Deep_Geometric_Descriptors_WACV_2023_paper.html)

   *Lokesh Veeramacheneni and Matias Valdenegro-Toro.* 

### [HPC](#content)
1. **Anomaly detection using autoencoders in high performance computing systems.** IAAI, 2019. [paper](https://dl.acm.org/doi/10.1609/aaai.v33i01.33019428)

   *Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, and Luca Benini.* 

### [Intrusion](#content)
1. **Intrusion detection using convolutional neural networks for representation learning.** ICONIP, 2017. [paper](https://link.springer.com/chapter/10.1007/978-3-319-70139-4_87)

   *Hipeng Li, Zheng Qin, Kai Huang, Xiao Yang, and Shuxiong Ye.* 

### [Diagnosis](#content)
1. **Transformer-based normative modelling for anomaly detection of early schizophrenia.** NIPS, 2022. [paper](https://arxiv.org/abs/2212.04984)

   *Pedro F Da Costa, Jessica Dafflon, Sergio Leonardo Mendes, João Ricardo Sato, M. Jorge Cardoso, Robert Leech, Emily JH Jones, and Walter H.L. Pinaya.* 