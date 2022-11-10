# Anomaly Detection and Localization

Anomaly: Data instances that significantly deviate from the majority of data instances

Contributed by Chunyang Zhang.

## [Content](#content)

<table>
<tr><td colspan="2"><a href="#survey-papers">1. Survey</a></td></tr> 
<tr><td colspan="2"><a href="#methods">2. Methods</a></td></tr>
<tr>
    <td>&ensp;<a href="#pinn">2.1 PINN</a></td>
    <td>&ensp;<a href="#deeponet">2.2 DeepONet</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#fourier-operator">2.3 Fourier Operator</a></td>
    <td>&ensp;<a href="#graph-networks">2.4 Graph Networks</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#machine-learning">2.5 Machine Learning</a></td>
    <td>&ensp;<a href="#identification">2.6 Identification</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#finite-element">2.7 Finite Element</a></td>
    <td>&ensp;<a href="#convolutional-filter">2.8 Convolutional Filter</a></td>
</tr>
</table>




## [Survey papers](#content)

1. **Deep learning for anomaly detection: A review.** ACM Computing Surveys, 2022. [paper](https://dl.acm.org/doi/10.1145/3439950)

   *Guansong Pang, Chunhua Shen, Longbing Cao, and Anton Van Den Hengel.*

1. **A unifying review of deep and shallow anomaly detection.** Proceedings of the IEEE, 2020. [paper](https://ieeexplore.ieee.org/document/9347460)

   *Ruff, Lukas and Kauffmann, Jacob R. and Vandermeulen, Robert A. and Montavon, Grégoire and Samek, Wojciech and Kloft, Marius and Dietterich, Thomas G., and Müller, Klaus-Robert.*

1. **A review on outlier/anomaly detection in time series data.** ACM Computing Surveys, 2022. [paper](https://dl.acm.org/doi/10.1145/3444690)

   *Ane Blázquez-García, Angel Conde, Usue Mori, Jose A. Lozano.* 

1. **A comprehensive survey on graph anomaly detection with deep learning.** IEEE Transactions on Knowledge and Data Engineering, 2021. [paper](https://ieeexplore.ieee.org/document/9565320)

   *Ma, Xiaoxiao and Wu, Jia and Xue, Shan and Yang, Jian and Zhou, Chuan and Sheng, Quan Z. and Xiong, Hui, and Akoglu, Leman.* 

1. **Transformers in time series: A survey.** arXiv, 2022. [paper](https://arxiv.org/abs/2202.07125)

   *Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun.*

1. **Deep learning approaches for anomaly-based intrusion detection systems: A survey, taxonomy, and open issues.** Knowledge-Based Systems, 2020. [paper](https://www.sciencedirect.com/science/article/pii/S0950705119304897)

   *Arwa Aldweesh, Abdelouahid Derhab, and Ahmed Z.Emam.* 

1. **A survey of deep learning-based network anomaly detection.** Cluster Computing, 2019. [paper](https://link.springer.com/article/10.1007/s10586-017-1117-8)

   *Donghwoon Kwon, Hyunjoo Kim, Jinoh Kim, Sang C. Suh, Ikkyun Kim, and Kuinam J. Kim.* 

1. **Survey on anomaly detection using data mining techniques.** Procedia Computer Science, 2015. [paper](https://www.sciencedirect.com/science/article/pii/S1877050915023479)

   *Shikha Agrawal and Jitendra Agrawal.* 

1. **Graph based anomaly detection and description: A survey.** Data Mining and Knowledge Discovery, 2015. [paper](https://link.springer.com/article/10.1007/s10618-014-0365-y)

   *Leman Akoglu, Hanghang Tong, and Danai Koutra.* 

1. **Real-time big data processing for anomaly detection: A Survey.** International Journal of Information Management, 2019. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0268401218301658)

   *Riyaz Ahamed Ariyaluran Habeeb, Fariza Nasaruddin, Abdullah Gani, Ibrahim AbakerTargio Hashem, Ejaz Ahmed, Muhammad Imran.* 

1. **A survey of network anomaly detection techniques.** Journal of Network and Computer Applications, 2016. [paper](https://www.sciencedirect.com/science/article/abs/pii/S1084804515002891)

   *Mohiuddin Ahmed, Abdun Naser Mahmood, and JiankunHu.* 

## [Methods](#content) 

### [PINN](#content)