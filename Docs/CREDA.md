

_Article_ 

# **Conditional Domain Adaptation with** **_α_ -Rényi Entropy Regularization and Noise-Aware Label Weighting** 

**Diego Armando Pérez-Rosero * , Andrés Marino Álvarez-Meza and German Castellanos-Dominguez** 

Signal Processing and Recognition Group, Universidad Nacional de Colombia, Manizales 170003, Colombia; amalvarezme@unal.edu.co (A.M.Á.-M.); cgcastellanosd@unal.edu.co (G.C.-D.) 

***** Correspondence: dieaperezros@unal.edu.co 

### **Abstract** 

Academic Editor: Dongping Zhu Received: 10 July 2025 Revised: 8 August 2025 Accepted: 12 August 2025 Published: 14 August 2025 **Citation:** Pérez-Rosero, D.A.; Álvarez-Meza, A.M.; CastellanosDominguez, G. Conditional Domain Adaptation with _α_ -Rényi Entropy Regularization and Noise-Aware Label Weighting. _Mathematics_ **2025** , _13_ , 2602. https://doi.org/10.3390/ math13162602 

**Copyright:** © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/). 

Domain adaptation is a key approach to ensure that artificial intelligence models maintain reliable performance when facing distributional shifts between training (source) and testing (target) domains. However, existing methods often struggle to simultaneously preserve domain-invariant representations and discriminative class structures, particularly in the presence of complex covariate shifts and noisy pseudo-labels in the target domain. In this work, we introduce Conditional Rényi _α_ -Entropy Domain Adaptation, named CREDA, a novel deep learning framework for domain adaptation that integrates kernel-based conditional alignment with a differentiable, matrix-based formulation of Rényi’s quadratic entropy. The proposed method comprises three main components: (i) a deep feature extractor that learns domain-invariant representations from labeled source and unlabeled target data; (ii) an entropy-weighted approach that down-weights low-confidence pseudo-labels, enhancing stability in uncertain regions; and (iii) a class-conditional alignment loss, formulated as a Rényi-based entropy kernel estimator, that enforces semantic consistency in the latent space. We validate CREDA on standard benchmark datasets for image classification, including Digits, ImageCLEF-DA, and Office-31, showing competitive performance against both classical and deep learning-based approaches. Furthermore, we employ nonlinear dimensionality reduction and class activation maps visualizations to provide interpretability, revealing meaningful alignment in feature space and offering insights into the relevance of individual samples and attributes. Experimental results confirm that CREDA improves cross-domain generalization while promoting accuracy, robustness, and interpretability. 

**Keywords:** domain adaptation; image classification; Rényi’s entropy; class-conditional alignment; noisy labels 

**MSC:** 68T05 

## **1. Introduction** 

A primary challenge in the development of artificial intelligence systems is ensuring that models maintain reliable performance under conditions that differ from those observed during training [1]. Such discrepancies may arise due to changes in the operational environment, variations in acquisition devices, or differences in user population characteristics [2]. These shifts, though often subtle, can significantly impact model behavior and compromise generalization capabilities, even when the underlying task remains unchanged [3]. This vulnerability becomes particularly critical in real-world applications, where it is infeasible to anticipate all possible future scenarios, thus limiting the scalability 

_Mathematics_ **2025** , _13_ , 2602 

https://doi.org/10.3390/math13162602 

_Mathematics_ **2025** , _13_ , 2602 

2 of 29 

and trustworthiness of deployed solutions [4]. In this context, validation within the source domain alone proves insufficient to guarantee consistent performance in heterogeneous settings, prompting the development of strategies to mitigate such discrepancies. Among these, domain adaptation has emerged as a key approach, enabling the reuse of pretrained models in new environments by aligning distributions across domains, thereby reducing the need for extensive data collection and annotation in the target domain [5]. The latter not only enhances the efficiency of knowledge transfer, but also supports the creation of more robust and sustainable systems in dynamic and uncertain environments. 

Despite the progress achieved through domain adaptation, the problem of generalizing to unseen domains remains only partially resolved. Domain shifts can take complex forms that go beyond marginal discrepancies, affecting the internal structure of learned representations and leading to systematic performance degradation in the target domain [6]. Consequently, adapted models frequently exhibit degraded or inconsistent performance when deployed in unfamiliar environments, especially under shifts in input distributions that are structural and semantic in nature [7]. This limitation arises primarily from the inability to preserve domain-invariant features under covariate shifts, where noise in input features, biased samples, or insufficient representations can degrade the alignment across domains and compromise the stability of the learned models [8]. Second, generalization is further hindered when the learned features lack discriminative power, particularly in the presence of concept shift and noisy labels. These factors distort latent representations and decision boundaries, making it difficult to maintain semantic clarity in the target domain [9]. Third, the absence of interpretability mechanisms impedes the reliable evaluation of whether predictions are based on meaningful semantic signals or on spurious correlations inherited from the source domain [10]. Collectively, these challenges hinder the development of domain-adaptive systems that are accurate, robust, and interpretable. 

In response to the challenges inherent in domain adaptation, numerous classical approaches have been proposed, most of which rely on linear transformations to align source and target distributions. These strategies aim to mitigate distributional discrepancies through statistical alignment techniques. Methods such as Correlation Alignment (CORAL) and Subspace Alignment (SA) reduce marginal discrepancy by aligning covariance matrices or projecting data onto orthonormal subspaces [11,12]. Despite their effectiveness under controlled conditions, their reliance on original feature spaces or linear projections makes them susceptible to distortions, noise, and domain-specific biases, hindering the extraction of invariant representations [13]. To address these limitations, geometrically inspired extensions such as Geometric Transfer Learning (GTL) have been developed, incorporating structural constraints between domains [14]. Nonetheless, they depend on linear subspace representations, which fail to adequately preserve the support of the target domain in the presence of data heterogeneity or limited representational capacity [15]. In addition, techniques such as Transfer Joint Matching (TJM), Transfer Component Analysis (TCA), and Maximum Independence Domain Adaptation (MIDA) seek to align both marginal and conditional distributions via linear projections [16–18]. Yet, they do not guarantee class separability in the latent space, particularly under concept shift or class imbalance, resulting in ambiguous decision boundaries and diminished discriminative performance [19]. A comparable deficiency is noted in Joint Distribution Adaptation (JDA), which, despite modeling joint alignment, assumes uniform relevance across classes and lacks adaptive mechanisms to address intra-class heterogeneity or instance-level significance [20]. 

Due to the structural constraints of traditional domain adaptation techniques, particularly the decoupling of feature transformation and prediction phases, deep learning methods have emerged as a more cohesive solution for preserving domain-invariant features across the representation space [21]. These approaches leverage the expressive 

_Mathematics_ **2025** , _13_ , 2602 

3 of 29 

capabilities of deep neural networks to jointly optimize feature extraction and domain alignment, enhancing adaptability under covariate shift [22]. Adversarial training-based models, including Domain-Adversarial Neural Networks (DANNs) and their extensions, have demonstrated considerable effectiveness in aligning marginal distributions within a shared latent space [23,24]. Still, while these methods reduce global disparities, they often struggle to maintain class separability, as they do not explicitly model conditional structures or discriminative boundaries [25]. To overcome these limitations, hybrid models have emerged that integrate deep learning architectures with statistical alignment objectives, enabling end-to-end optimization for improved domain adaptation performance [26]. These approaches aim to preserve both predictive accuracy and domain invariance by combining supervised losses with the minimization of statistical discrepancies across multiple network layers [27,28]. However, hybrid methods also face challenges, such as gradient conflicts between classification and alignment objectives and semantic misalignment caused by noisy pseudo-labels [29]. In parallel, self-supervised learning (SSL) has been introduced into domain adaptation pipelines to alleviate the dependence on labeled target data, typically by leveraging contrastive objectives to learn transferable features without explicit supervision [30–32]. More recently, foundation models—large-scale pretrained architectures with broad generalization capacity—have opened new avenues for adaptation by employing mechanisms such as prompt tuning, adapter modules, or domain-specific fine-tuning [33,34]. While these strategies show promise, their deployment in the presence of domain shift remains constrained by semantic misalignment and high computational cost [35]. Although deep learning has significantly advanced the extraction of domaininvariant features, ensuring discriminative consistency and semantic alignment in the target domain remains a critical challenge [36]. 

Despite notable advances in deep learning techniques designed to extract domaininvariant features, many of these methods struggle to maintain a discriminative class structure within the target domain [21,22]. To address this, transfer-based strategies—such as finetuning, teacher–student models, meta-learning frameworks, and asymmetric architectures like Adversarial Discriminative Domain Adaptation (ADDA)—have been introduced to enhance inter-class separation through adaptive training or auxiliary supervision [25,37–39]. However, these methods often suffer from limitations including degradation of pretrained representations and sensitivity to noise [40,41] and the absence of explicit modeling of class boundaries, particularly in ADDA variants [42]. Conditional alignment techniques, such as Conditional Adversarial Domain Adaptation (CDAN), address part of this shortcoming by incorporating classifier outputs into the discriminator, thereby capturing class-conditional dependencies [43]. Nonetheless, they remain vulnerable to class imbalance and low-confidence predictions, which can lead to distorted decision boundaries [36]. In response to these challenges, information-theoretic approaches have emerged as a complementary paradigm, optimizing transfer through objectives based on mutual information or entropy [44,45]. By leveraging strategies such as entropy minimization and the information bottleneck principle, these methods regularize latent representations, thereby mitigating overfitting on the source domain and improving generalization under target shift [46–48]. 

In addition to generalization and discriminability, interpretability has become a pivotal aspect of domain adaptation, especially in high-stakes applications where understanding model behavior is essential for fostering trust, transparency, and accountability [49]. In this context, latent space analysis has proven valuable for examining the structure of learned representations. Linear techniques such as Principal Component Analysis (PCA) offer computational efficiency but fall short in capturing the nonlinear relationships relevant across multiple domains [50]. In contrast, nonlinear methods like t-distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold Approximation and Projection 

_Mathematics_ **2025** , _13_ , 2602 

4 of 29 

(UMAP) are more effective in representing complex inter-domain structures [51]. UMAP, in particular, stands out for its ability to preserve both local and global structures, maintain stability under parameter variation, and scale efficiently—making it especially useful for visualizing semantic alignment across domains [52,53]. Moreover, interpretability is especially crucial in sensitive applications. Among post hoc methods, Gradient-weighted Class Activation Mapping (Grad-CAM) generates attention maps that highlight regions influencing model predictions, while its extension, Grad-CAM++, improves spatial resolution through higher-order derivatives, though it remains limited by nonlinear activation functions [54–56]. In domain adaptation, Grad-CAM++ has proven effective not only as an explainability tool but also for visually assessing semantic consistency across domains [57]. Other approaches, such as Layer-wise Relevance Propagation (LRP) and SHapley Additive exPlanations (SHAP), provide quantitative insights by assigning relevance scores to input features, aiding the identification of spurious patterns or conflicting decision rules [58]. The lack of interpretability methods specifically designed for transfer learning and domain adaptation remains a significant limitation, highlighting the need for more robust explanatory tools tailored to cross-domain scenarios [59]. 

Here, we propose Conditional Rényi _α_ -Entropy Domain Adaptation (CREDA), a novel domain adaptation framework designed to simultaneously preserve domain-invariant representations, enforce class-conditional alignment, and mitigate the effect of noisy pseudolabels. The core idea of CREDA is to regularize deep feature alignment using a differentiable, matrix-based formulation of Rényi’s quadratic entropy, which provides a non-parametric and robust estimate of class-wise distributional similarity. CREDA is implemented as an end-to-end trainable architecture comprising three key stages: 

- Deep Feature Extraction: A shared ResNet-18 backbone encodes samples from both source and target domains into a latent representation space. 

- Noise-Aware Label Weighting: An entropy-derived confidence score is used to downweight low-confidence pseudo-labels in the target domain, improving robustness against noisy or ambiguous predictions. 

- Class-Conditional Alignment via Rényi-based entropy: A novel entropy-based regularization term is applied over kernel Gram matrices to minimize divergence between class-wise source and target feature distributions. 

We evaluate CREDA on three widely used visual domain adaptation benchmarks for image classification: Digits, ImageCLEF-DA, and Office-31. Additionally, we compare its performance against state-of-the-art methods—including DANN, ADDA, and CDAN+E— across various backbone architectures such as ResNet-18, ResNet-50, and Vision Transformers (ViT). The results consistently demonstrate that CREDA achieves superior performance in terms of classification accuracy, semantic alignment, and interpretability, with improvements of average accuracy across benchmarks. Qualitative analyses using UMAP and Grad-CAM++ further confirm that CREDA maintains both inter-class separability and cross-domain semantic coherence, highlighting its potential for deployment in real-world, label-scarce environments. 

The remainder of this paper is organized as follows: Section 2 introduces the materials and methods. Sections 3 and 4 discuss the experiments and results. Finally, Section 5 outlines the concluding remarks. 

## **2. Materials and Methods** 

### _2.1. Kernel Methods Fundamentals_ 

Kernel methods provide a powerful framework for developing nonlinear algorithms. The core idea is to implicitly map the input data from its original space _X_ into a high- 

_Mathematics_ **2025** , _13_ , 2602 

5 of 29 

dimensional, or even infinite-dimensional, feature space _H_ via a nonlinear mapping Φ : _X →H_ . The space _H_ is a special type of Hilbert space known as a Reproducing Kernel Hilbert Space (RKHS), and the mapping Φ is chosen such that complex patterns in the data may become simpler, e.g., linearly separable _H_ [60]. 

Explicitly computing the coordinates of the mapped data points Φ( _x_ ) is often computationally expensive or infeasible. Then, the kernel trick allows us to bypass this by defining a kernel function _κ_ : _X × X →_ R that computes the inner product between two points in the feature space: 



Then, we work directly with the kernel function without ever needing to know the explicit form of Φ or the structure of _H_ . Indeed, an RKHS is uniquely defined by this property, ensuring that all computations can be performed using the kernel [61]. In practice, a common choice for the kernel function is the Gaussian kernel: 



which corresponds to an infinite-dimensional feature space, with _σ ∈_ R<sup>+</sup> . Still, its mathematical tractability and intuitive notion of similarity make it a commonly used approach [62]. 

### _2.2. Kernel-Based α-Rényi’s Entropy Estimation_ 

Let _X_ be a continuous random variable with a probability density function (PDF) _f_ ( _x_ ), _x ∈ X_ , the Rényi’s _α_ -order entropy is defined as follows [63]: 



where _α >_ 0, and _α̸_ = 1. A primary challenge in applying this definition is that in most practical scenarios, especially with high-dimensional data like deep features, the underlying PDF _f_ ( _x_ ) is unknown [64]. To circumvent this, a Parzen-window method, also known as Kernel Density Estimation (KDE) can be employed. Namely, given a finite set of _N_ samples _{xi ∈ X}i_<sup>_N_</sup> =1<sup>, the PDF at any point</sup><sup>_x_can be estimated as the average of kernel functions</sup> centered at each sample [65]: 



where the Gaussian kernel is selected for its mathematical simplicity and desirable smoothing behavior (see Equation (2)). In particular, when _α_ = 2 in Equation (3), we focus on the special case of Rényi’s entropy, known as quadratic entropy. Indeed, the integral term in Equation (3),<sup>�</sup> _f_ ( _x_ )<sup>2</sup> _dx_ , is known as the Information Potential (IP) [66], a measure of the average information contained in the distribution. Substituting the KDE estimator _f_<sup>ˆ</sup> ( _x_ ) into the IP integral, yields the following: 



_Mathematics_ **2025** , _13_ , 2602 

6 of 29 

A significant advantage of using a Gaussian kernel is that the integral in Equation (5) has a closed-form solution based on the convolution property of Gaussians [67]: 



The latter simplifies the IP estimator to a practical, sample-based formula that depends only on pairwise interactions between samples, completely bypassing the need for explicit PDF estimation: 



Next, let **K** _∈_ R<sup>_N×N_</sup> be a Gram matrix whose elements are the pairwise kernel evaluations, **K** _ij_ = _κ_<sup>_√_</sup> 2 _σ_<sup>(</sup><sup>_xi_,</sup><sup>_xj_).The sum of all elements in this matrix can be computed as</sup> **1**<sup>_T_</sup> **K1** , where **1** is a column vector of ones. This gives a matrix-based estimator for the IP: 



Recently, a _α_ -Rényi matrix-based operator extracts from the IP expression in Equation (8). More generally, Rényi’s entropy can be defined directly over the eigenspectrum of a normalized Gram matrix. If we define a normalized Gram matrix **A** = **K** /tr( **K** ), where tr( _·_ ) is the trace operator, the entropy is given by [68]: 



where _λ_<sup>˘</sup> _i_ ( **A** ) are the eigenvalues of **A** . For our work with _α_ = 2, we use a computationally stable form based on the Frobenius norm: _H_ 2( **A** ) = _−_ log(tr( **A**<sup>˘</sup><sup>_⊤_</sup> **A**<sup>˘</sup> )), where **A**<sup>˘</sup> = **A** /tr( **A** ), tr( **A**<sup>˘</sup> ) = 1, and _∥_ **A** _∥_<sup>2</sup> _F_<sup>=tr(</sup><sup>**A**</sup><sup>_⊤_</sup><sup>**A**).Thismatrix-basedformulationisessentialfordeep</sup> learning due to several key properties: 

- Non-parametric: It makes no prior assumptions about the underlying data distribution, making it highly suitable for the complex and high-dimensional feature spaces learned by neural networks. 

- Differentiable: The entropy loss is a function of the Gram matrix elements, which are themselves differentiable functions of the feature vectors produced by a given network. This allows gradients to be backpropagated through the kernel computations to the network’s parameters, enabling end-to-end training. 

- Robust: The entropy is calculated based on the collective geometric structure of the data, as captured by all pairwise interactions in the Gram matrix. This makes the measure inherently robust to outliers, which would have a limited impact on the overall sum of kernel values. 

The matrix-based entropy framework in Equation (9) can be extended to measure relationships between two random variables, _X_ and _Y_ , represented by paired feature vectors _{_ **f** _X_ , _i_ , **f** _Y_ , _i}i_<sup>_N_</sup> =1<sup>.This is achieved by defining a joint Gram matrix using the Hadamard</sup> (element-wise) product as follows: 

- Joint Entropy—(JE). Let **K** _X ∈_ R<sup>_N×N_</sup> and **K** _Y ∈_ R<sup>_N×N_</sup> be the Gram matrices computed from the feature sets of _X_ and _Y_ , respectively. The joint entropy based on the _α_ -Rényi estimator is defined as follows [69]: 



_Mathematics_ **2025** , _13_ , 2602 

7 of 29 

where **K** _XY_ = **K** _X ⊙_ **K** _Y_ , **K**<sup>˘</sup> _X_ , _Y_ = **K** _X_ , _Y_ /tr( **K** _X_ , _Y_ ), and _⊙_ denotes the Hadamard product. Of note, the joint matrix **K** _XY_ captures the similarity between pairs of samples in the joint feature space. 

- Mutual Information—(MI). It quantifies the statistical dependence between two variables. In the matrix-based framework, it is defined in analogy to its classic informationtheoretic definition: 



where each entropy term is computed from its respective (normalized) Gram matrix. Maximizing MI is a common objective in representation learning, as it encourages a representation to retain information about a relevant variable. 

– Conditional Entropy—(CE). It measures the remaining uncertainty in a variable _X_ given that _Y_ is known. It is defined as follows: 



Minimizing conditional entropy is equivalent to making _X_ more predictable from _Y_ . 

_2.3. Domain Adaptation with α-Rényi Entropy-Based Label Weighting and Regularization_ 

Our proposed method, Conditional _α_ -Rényi’s Entropy Regularization (CREDA), is designed for end-to-end training in unsupervised domain adaptation. The framework leverages a deep feature extractor _F_ : _X →_ R<sup>_d_</sup> that maps an input image **x** _∈_ R<sup>_H_˘</sup><sup>_×W_˘</sup><sup>_×_˘</sup><sup>_C_</sup> , with _X ⊆_ R<sup>_p′_</sup> , _p_<sup>_′_</sup> = _H_<sup>˘</sup> _× W_<sup>˘</sup> _× C_<sup>˘</sup> , to a _d_ -dimensional feature vector **f** _∈_ R<sup>_d_</sup> , as follows: 



where _f_<sup>˘</sup> _l_ ( _·_ ) stands for the _l_ -th feature extractor layer ( _l ∈{_ 1, . . . , _L}_ ), and _◦_ is the function composition operator. Moreover, a classifier _G_ : R<sup>_d_</sup> _→_ [0, 1]<sup>_C_</sup> that predicts class-probability vector **g** _∈_ [0, 1]<sup>_C_</sup> , is defined as follows: 



with _g_ ˘ _l′_ ( _·_ ) as a given classifier layer ( _l_<sup>_′_</sup> _∈ L_<sup>˘</sup> ), ∑<sup>_C_</sup> _c_ =1<sup>_gc_= 1, and</sup><sup>_gc∈_</sup><sup>**g**.</sup> In practice, we are given a labeled source domain _D_<sup>_s_</sup> = _{_ **x** _i_<sup>_s∈_R</sup><sup>_p′_,</sup><sup>**y**</sup> _i_<sup>_s∈{_0, 1</sup><sup>_}C}_</sup> _i_<sup>_N_</sup> =<sup>_s_</sup> 1<sup>,</sup> with ∑<sup>_C_</sup> _c_ =1<sup>_y_</sup> _i_<sup>_s_</sup> , _c_<sup>= 1,</sup><sup>_y_</sup> _i_<sup>_s_</sup> , _c̸_<sup>=</sup><sup>_y_</sup> _i_<sup>_s_</sup> , _c_<sup>_′_,</sup><sup>_c_,</sup><sup>_c′∈C_, and</sup><sup>_y_</sup> _i_<sup>_s_</sup> , _c_<sup>,</sup><sup>_y_</sup> _i_<sup>_s_</sup> , _c_<sup>_′∈_</sup><sup>**y**</sup> _i_<sup>_s_.Also, an unlabeled target domain</sup> is provided as _D_<sup>_t_</sup> = _{_ **x**<sup>_t_</sup> _j_<sup>_∈_R</sup><sup>_p′}N_</sup> _j_ =<sup>_t_</sup> 1<sup>.Foreachclass</sup><sup>_c_,wecomputethesource,target,</sup> and source-target kernel-based matrices **K** _c_<sup>_s∈_R</sup><sup>_nsc×nsc_,</sup><sup>**K**</sup><sup>_t_</sup> _c_<sup>_∈_R</sup><sup>_ntc×ntc_,and</sup><sup>**K**</sup><sup>_st_</sup> _c_<sup>_∈_R</sup><sup>_ntc×nsc_,</sup> as follows: 





where _g_<sup>_t_</sup> _j_ , _c_<sup>_∈_</sup><sup>**g**</sup><sup>_t_</sup> _j_<sup>,and</sup><sup>**g**</sup><sup>_t_</sup> _j_<sup>=</sup><sup>_G_(</sup><sup>**f**</sup><sup>_t_</sup> _j_<sup>).Moreover,</sup><sup>_n_</sup> _c_<sup>_s_isthenumberofsamplesin</sup><sup>_Ds_,where</sup> _yi_<sup>_s_</sup> , _c_<sup>= 1.Likewise,</sup><sup>_n_</sup> _c_<sup>_t_holds the number of target inputs satisfying arg max</sup> _c_<sup>_′gt_</sup> _j_ , _c_<sup>_′_=</sup><sup>_c_.</sup> Here, to enhance robustness against noisy pseudo-labels in the target set, we introduce a confidence weighting scheme derived from a principled, entropy-based measure of prediction uncertainty. The core idea is to quantify the uncertainty of a classifier’s 

_Mathematics_ **2025** , _13_ , 2602 

8 of 29 

output probability vector, **g** _j ∈_ [0, 1]<sup>_C_</sup> , using its Rényi’s quadratic entropy in Equation (3), as follows: 



In turn, to create a universally comparable score, this entropy value is normalized by its theoretical maximum, which occurs for a uniform distribution and is equal to _H_ 2,max = _−_ log ∑<sup>_C_</sup> = log( _C_ ). This yields a normalized uncertainty score � _c_ =1<sup>(1/</sup><sup>_C_)2�</sup> _U_ ˆ ( **g**<sup>_t_</sup> _j_<sup>)=</sup><sup>_H_ˆ2(</sup><sup>**g**</sup><sup>_t_</sup> _j_<sup>)/ log(</sup><sup>_C_),whichisboundedin[0, 1].Therefore,weproposeincorporat-</sup> ing a confidence weighting vector **w**<sup>_t_</sup> _∈_ R<sup>_Nt_</sup> , derived from the normalized uncertainty score _U_<sup>ˆ</sup> ( **g**<sup>_t_</sup> _j_<sup>):</sup> 



where _w_<sup>_t_</sup> _j_<sup>_∈_</sup><sup>**w**</sup><sup>_t_.The latter provides a theoretically grounded mechanism to down-weight</sup> ambiguous predictions, a strategy that has proven effective in related contexts for handling label uncertainty [70]. 

Afterward, a target weighting matrix **W**<sup>˜</sup><sup>_c_</sup> _t_<sup>_∈_R</sup><sup>_ntc×ntc_can be computed, yielding the fol-</sup> lowing: 



where **w** ˜<sup>_t_</sup> _c_<sup>=</sup><sup>_{wt_</sup> _j_<sup>: arg max</sup><sup>_c′gt_</sup> _j_ , _c_<sup>_′_=</sup><sup>_c} ∈_R</sup><sup>_ntc_.</sup> 

Now, our CREDA method lies in a novel regularization term that enforces alignment between the class-conditional distributions of the source and target domains. So, we employ a kernel-based quadratic Rényi entropy mutual information estimator (see Section 2.2) and the confidence weighting scheme in Equation (19), as follows: 



where **K**<sup>˜</sup><sup>_t_</sup> _c_<sup>=</sup><sup>**K**</sup><sup>_t_</sup> _c_<sup>_⊙_</sup><sup>**W**˜</sup><sup>_t_</sup> _c_<sup>, and</sup> 



which enables the computation of our MI estimator in Equation (21) even when the source and target sample sizes differ, namely _n_<sup>_t_</sup> _c̸_<sup>=</sup><sup>_n_</sup> _c_<sup>_s_.</sup> 

Finally, the complete CREDA loss integrates the standard supervised cross-entropy on labeled source data with our proposed mutual information regularizer, based on the quadratic Rényi entropy formulation, as follows: 



where _λ ∈_ R<sup>+</sup> is a hyperparameter controlling the strength of the domain alignment. 

In practice, the computation of the kernel matrices in Equations (15)–(17) in our CREDA loss is performed within each training mini-batch. For a given mini-batch of source and target samples, features are first extracted, and pseudo-labels for the target samples are generated. Subsequently, for each class _c_ , the corresponding feature vectors from the source batch (with ground-truth label _c_ ) and the target batch (with pseudo-label _c_ ) are filtered. The cross-domain kernel matrix **K**<sup>_st_</sup> _c_<sup>is then computed by evaluating the Gaussian kernel</sup> between every filtered source feature and every filtered target feature from the batch. The intra-domain matrices, **K**<sup>_s_</sup> _c_<sup>and</sup><sup>**K**</sup><sup>_c_</sup> _t_<sup>, are computed similarly among the respective filtered</sup> features. If a class is not present in a given mini-batch, its contribution to the regularization 

_Mathematics_ **2025** , _13_ , 2602 

9 of 29 

loss for that training step is zero. This batch-wise, class-conditional procedure allows for an efficient and scalable implementation of our proposed alignment objective. 

Remarkably, the selection of Rényi’s quadratic entropy ( _α_ = 2) is motivated by its direct connection to the IP in Equation (5), which, under a Gaussian kernel, translates the alignment objective into a geometrically intuitive goal [63]. Specifically, the sample-based estimator in Equation (7) becomes a sum of pairwise similarities, meaning that minimizing our class-conditional loss in Equation (23) is equivalent to encouraging feature vectors of the same class to form tight, pure clusters in the feature space, directly promoting class separability. Furthermore, our approach is sensitive to higher-order statistics; thereby, CREDA-based loss captures the overall structure of the distributions, such as their dispersion and modality, which is critical for aligning complex, multi-modal classes often found in real-world datasets. Finally, the estimator’s formulation as an average over all pairwise interactions provides a robust estimate of class-wise distributional similarity. This inherent averaging makes the gradient estimates stable by mitigating the influence of individual outliers or noisy pseudo-labels, a common challenge in unsupervised settings. 

Moreover, in discussing the convergence properties of our CREDA loss, it is crucial to distinguish between the statistical consistency of the estimator and the empirical convergence of the deep learning model during training. The mutual information estimator in Equation (21) inherits strong theoretical properties from its foundation in Parzen-window kernel estimation (see Equation (4)). As established in non-parametric statistics, KDE provides a consistent estimator, meaning the estimated probability density converges to the true underlying density as the number of samples approaches infinity [65]. Consequently, the IP at the core of our approach, and by extension our full mutual information estimator, are also statistically consistent estimators of the true quadratic Rényi’s mutual information between the class-conditional distributions. Now, from an optimization perspective, the complete CREDA loss is non-convex due to the highly nonlinear nature of deep approaches. Therefore, formal guarantees of convergence to a global minimum are not feasible, a common characteristic of deep learning systems. Still, our method is designed to facilitate stable empirical convergence. The use of an infinitely differentiable Gaussian kernel ensures our regularization term is smooth, contributing to a well-behaved loss landscape that is conducive to gradient-based optimization. 

Figure 1 summarizes the core components and training pipeline of our proposed CRERDA model for conditional domain adaptation. 



<!-- Start of picture text -->
Source Domain<br>Extractor Classification  Classifcation Classification<br> Model Model Labels Loss<br>Target Domain<br>Total Loss<br>-Rényi-Based Loss<br>Label -Rényi-Based<br>Weighting Regularization<br><!-- End of picture text -->

**Figure 1.** CREDA framework for domain adaptation, incorporating classification loss and _α_ -Rényi Entropy-based label weighting and regularization to attain domain alignment with a class-aware structure. **Blue** : source, **Red** : target, **Purple** : shared. 

## **3. Experimental Set-Up** 

To rigorously evaluate the effectiveness of the proposed CREDA framework for domain adaptation in image classification tasks, we present a comprehensive analysis that includes descriptions of the benchmark datasets, training protocols, comparative baselines, and quantitative and qualitative performance assessments. 

_Mathematics_ **2025** , _13_ , 2602 

10 of 29 

### _3.1. Tested Datasets_ 

To assess the effectiveness and robustness of the proposed domain adaptation method, we conducted extensive experiments on three widely recognized benchmark datasets commonly used in domain adaptation research. Each dataset encompasses visual domains exhibiting substantial distribution shifts, thereby providing a challenging setting for learning domain-invariant representations, as detailed below: 

- _Digits:_ This benchmark suite is designed for evaluating domain adaptation on digit recognition tasks, spanning both handwritten and natural-scene digits. It comprises three standard datasets: MNIST (M), a large database of handwritten digits; USPS (U), another handwritten digit set characterized by its lower resolution; and SVHN (S), which contains house numbers cropped from real-world street-level images [71]. Notably, the S domain is particularly challenging due to its significant variability in lighting, background clutter, and visual styles compared to M and U (see Figure 2). 

- _ImageCLEF-DA:_ This is a standard benchmark for unsupervised domain adaptation, organized as part of the ImageCLEF evaluation campaign. It comprises 12 common object classes shared across three distinct visual domains: Caltech-256 (C), ImageNet ILSVRC 2012 (I), and Pascal VOC 2012 (P), see Figure 3. Each domain contains 600 images, with a balanced distribution of 50 images per class [72]. All images are resized to 224 _×_ 224 pixels. 

- _Office-31:_ It consists of 4110 images across 31 object classes, sourced from three domains with distinct visual characteristics: Amazon (A), which features centered objects on a clean, white background under controlled lighting; Webcam (W), containing lowresolution images with typical noise and color artifacts; and DSLR (D), which includes high-resolution images with varying focus and lighting conditions [73]. Here, we selected a subset of ten shared classes (see Figure 4). 

Together, these benchmarks allows evaluating the capacity of domain adaptation methods to generalize across diverse and challenging visual domains. 



<!-- Start of picture text -->
M<br>U<br>S<br>0 1 2 3 4 5 6 7 8 9<br><!-- End of picture text -->

**Figure 2.** Representative input images for each digit class across source and target domains. 



<!-- Start of picture text -->
P<br>I<br>C<br>Aeroplane Bike Motorbike People Bird Boat Bottle Bus Car Dog Horse Monitor<br><!-- End of picture text -->

**Figure 3.** Representative input images for each object class across source and target domains in the ImageCLEF-DA dataset. 

_Mathematics_ **2025** , _13_ , 2602 

11 of 29 



<!-- Start of picture text -->
A<br>W<br>D<br>Backpack Bike Calculator Headphones Keyboard Laptop Monitor Mouse Mug Projector<br><!-- End of picture text -->

**Figure 4.** Representative input images for each object class across source and target domains in the Office-31 dataset. 

### _3.2. Assessment and Method Comparison_ 

To comprehensively evaluate the impact of the feature extractor’s architecture on model performance, we experimented with three distinct backbones: a standard ResNet-18, its deeper counterpart ResNet-50, and a ViT. Each backbone is adapted for feature extraction in domain transfer tasks by removing its final classification layer. The primary baseline is a ResNet-18 convolutional backbone pretrained on ImageNet [74]. To tailor the architecture for our tasks, the final fully connected layer is removed, while all preceding convolutional and residual blocks are retained. This modification enables the extraction of high-level spatial representations that are robust and transferable across domains [75]. A comprehensive description of the ResNet-18 feature extractor’s architecture is provided in Table 1. 

**Table 1.** Architectural details of the ResNet-18 feature extractor. 

|**Layer Name**|**Type**|**Input Shape**|**Output Shape**|**Param. #**|
|---|---|---|---|---|
|Input|InputLayer|(3, <sup>˘</sup>_H_, <sup>˘</sup>_W_)|(3, <sup>˘</sup>_H_, <sup>˘</sup>_W_)|0|
|Conv1|Conv2D + BN + ReLU|(3, <sup>˘</sup>_H_, <sup>˘</sup>_W_)|(64, <sup>˘</sup>_H_/2, <sup>˘</sup>_W_/2)|9408|
|MaxPool|MaxPooling|(64, <sup>˘</sup>_H_/2, <sup>˘</sup>_W_/2)|(64, <sup>˘</sup>_H_/4, <sup>˘</sup>_W_/4)|0|
|Layer1|Residual Block_×_2|(64, <sup>˘</sup>_H_/4, <sup>˘</sup>_W_/4)|(64, <sup>˘</sup>_H_/4, <sup>˘</sup>_W_/4)|73,728|
|Layer2|Residual Block_×_2|(64, <sup>˘</sup>_H_/4, <sup>˘</sup>_W_/4)|(128, <sup>˘</sup>_H_/8, <sup>˘</sup>_W_/8)|230,144|
|Layer3|Residual Block_×_2|(128, <sup>˘</sup>_H_/8, <sup>˘</sup>_W_/8)|(256, <sup>˘</sup>_H_/16, <sup>˘</sup>_W_/16)|919,040|
|Layer4|Residual Block_×_2|(256, <sup>˘</sup>_H_/16, <sup>˘</sup>_W_/16)|(512, <sup>˘</sup>_H_/32, <sup>˘</sup>_W_/32)|3,674,112|
|AvgPool|GlobalAvgPooling|(512, <sup>˘</sup>_H_/32, <sup>˘</sup>_W_/32)|(512, 1, 1)|0|
|Flatten|Flatten|(512, 1, 1)|(512)|0|



Afterward, to investigate the effect of network depth, we also employed a ResNet-50 backbone, a deeper and more powerful variant within the ResNet family [74]. ResNet-50 utilizes bottleneck residual blocks, which are more computationally efficient for deeper networks [76]. Similar to the ResNet-18 configuration, the model is pretrained on ImageNet, and its final fully connected layer is removed to serve as a feature extractor. This results in a 2048-dimensional feature vector. The detailed architecture is presented in Table 2. 

Also, to explore an alternative architectural paradigm beyond convolutional networks, we incorporated a ViT-based model, specifically the `vit_tiny_patch16_224` variant (termed ViT-Tiny) [77]. Unlike CNNs, ViT-Tiny processes images by splitting them into a sequence of fixed-size patches, which are then linearly embedded and fed into a standard Transformer encoder. For this study, we use a ViT-Tiny pretrained on ImageNet with an input resolution of 224 _×_ 224. The classification head is discarded, and the output embedding of the special `[CLS]` token from the final Transformer block is used as the feature representation, yielding a 192-dimensional vector. The architecture is detailed in Table 3. 

_Mathematics_ **2025** , _13_ , 2602 

12 of 29 

**Table 2.** Architectural details of the ResNet-50 feature extractor. 

|**Layer Name**|**Type**|**Input Shape**|**Output Shape**|**Param. #**|
|---|---|---|---|---|
|Input|InputLayer|(3, <sup>˘</sup>_H_, <sup>˘</sup>_W_)|(3, <sup>˘</sup>_H_, <sup>˘</sup>_W_)|0|
|Conv1|Conv2D + BN + ReLU|(3, <sup>˘</sup>_H_, <sup>˘</sup>_W_)|(64, <sup>˘</sup>_H_/2, <sup>˘</sup>_W_/2)|9408|
|MaxPool|MaxPooling|(64, <sup>˘</sup>_H_/2, <sup>˘</sup>_W_/2)|(64, <sup>˘</sup>_H_/4, <sup>˘</sup>_W_/4)|0|
|Layer1|Bottleneck Block_×_3|(64, <sup>˘</sup>_H_/4, <sup>˘</sup>_W_/4)|(256, <sup>˘</sup>_H_/4, <sup>˘</sup>_W_/4)|214,528|
|Layer2|Bottleneck Block_×_4|(256, <sup>˘</sup>_H_/4, <sup>˘</sup>_W_/4)|(512, <sup>˘</sup>_H_/8, <sup>˘</sup>_W_/8)|1,182,720|
|Layer3|Bottleneck Block_×_6|(512, <sup>˘</sup>_H_/8, <sup>˘</sup>_W_/8)|(1024, <sup>˘</sup>_H_/16, <sup>˘</sup>_W_/16)|7,084,032|
|Layer4|Bottleneck Block_×_3|(1024, <sup>˘</sup>_H_/16, <sup>˘</sup>_W_/16)|(2048, <sup>˘</sup>_H_/32, <sup>˘</sup>_W_/32)|15,085,568|
|AvgPool|GlobalAvgPooling|(2048, <sup>˘</sup>_H_/32, <sup>˘</sup>_W_/32)|(2048, 1, 1)|0|
|Flatten|Flatten|(2048, 1, 1)|(2048)|0|



**Table 3.** Architectural details of the ViT-Tiny feature extractor. 

|**Layer Name**|**Type**|**Input Shape**|**Output Shape**|**Param. #**|
|---|---|---|---|---|
|Input|InputLayer|(3, 224, 224)|(3, 224, 224)|0|
|Patch Embedding|Conv2D (Patching)|(3, 224, 224)|(196, 192)|147,648|
|Add CLS Token|Concatenation|(196, 192)|(197, 192)|192|
|Add Pos. Embedding|Parameter Add|(197, 192)|(197, 192)|37,824|
|Transformer Encoder|Encoder Block_×_12|(197, 192)|(197, 192)|5,529,792|
|Extract CLS Token|Indexing|(197, 192)|(1, 192)|0|
|LayerNorm|LayerNorm|(1, 192)|(1, 192)|384|
|Flatten|Flatten|(1, 192)|(192)|0|



Moreover, the following domain adaptation strategies are considered for comparison: 

– Baseline: A thenceforward approach is trained exclusively on the source domain without any adaptation mechanism, see Figure 5. The optimization objective is to minimize the conventional supervised cross-entropy loss, which serves as a lower bound for performance evaluation under domain shift: 



– DANN: The Domain-Adversarial Neural Network (DANN) [78] introduces a domain discriminator, _G_<sup>˜</sup> : R<sup>_d_</sup> _→_ [0, 1], which is trained to distinguish source features from target ones, see Figure 6. The discriminator is implemented as a multi-layer neural network, where a predicted label of 1 indicates source domain membership, and 0 indicates target domain membership. Moreover, the feature extractor is simultaneously trained to produce features that fool the discriminator, thereby learning domaininvariant representations via a Gradient Reversal Layer (GRL). The overall objective is a minimax game: 



where _λ_<sup>˘</sup> _∈_ R<sup>+</sup> represents a trade-off hyperparameter. The domain adversarial loss _LAdv_ is the binary cross-entropy for domain classification, where source samples are assigned domain label 0, and target samples label 1. 

- ADDA: The Adversarial Discriminative Domain Adaptation (ADDA) framework [79] separates the training into two distinct stages, see Figure 7. First, a source feature extractor _F_<sup>_s_</sup> ( _·_ ) and the classifier _G_ ( _·_ ) are trained using the supervised loss _L_ Baseline (see Equation (24)). In the second stage, the parameters of _F_<sup>_s_</sup> ( _·_ ) and _G_ ( _·_ ) are frozen. Then, a new target feature extractor, _F_<sup>_t_</sup> ( _·_ ) (initialized with the weights in _F_<sup>_s_</sup> ( _·_ )), is then trained to fool the domain discriminator in a minimax game (see Equation (25)). 

_Mathematics_ **2025** , _13_ , 2602 

13 of 29 

The objective is to align the target feature distribution with the fixed source feature distribution. 

- CDAN+E: The Conditional Domain Adversarial Network (CDAN) [80] enhances adversarial alignment by using a multilinear feature representation, **h** = _F_ ( **x** ) _⊗G_ ( _F_ ( **x** )), as input to the domain discriminator _G_<sup>˜</sup> . The CDAN+E variant, as implemented in standard benchmarks, employs a sophisticated entropy-based mechanism that serves a dual purpose: it implements entropy minimization for the target domain while simultaneously weighting the adversarial loss to focus on more reliable samples, as seen in Figure 8. 

Specifically, the Shannon entropy _H_ ( **g** ) is computed for the predictions **g** of all samples in a batch. This entropy value is then used in two ways. First, it is passed through a GRL, which implicitly creates an entropy minimization objective for the feature extractor, encouraging it to produce more confident (low-entropy) predictions. Second, the entropy is transformed into a sample-wise weight, as follows: 



This weighting scheme gives greater importance to samples with confident predictions (low entropy), thereby focusing the adversarial alignment on well-structured regions of the feature space. The resulting weighted conditional adversarial loss, _LAdv_ , is then defined as follows: 



where both _w_ ˘ _i_<sup>_s_and</sup><sup>_w_˘</sup><sup>_t_</sup> _j_<sup>are calculated according to Equation (26).The total loss for the</sup> CDAN+E framework can thus be expressed as the combination of the supervised loss and this integrated adversarial and entropy-regularized objective (see Equation (25)). 



<!-- Start of picture text -->
Source Domain<br>Extractor Classification  Classifcation<br>Baseline Loss<br> Model Model Labels<br><!-- End of picture text -->

**Figure 5.** Baseline model for supervised training on the source domain without adaptation. 



<!-- Start of picture text -->
Source Domain<br>Extractor Classification  Classifcation<br>Baseline Loss<br> Model Model Labels<br>Target Domain<br>DANN Loss<br>Discriminator  Domain  Adversarial<br>Model Labels Loss<br><!-- End of picture text -->

**Figure 6.** DANN framework for unsupervised domain adaptation. **Blue** : source, **Red** : target, **Purple** : shared. 

_Mathematics_ **2025** , _13_ , 2602 

14 of 29 



<!-- Start of picture text -->
Source Domain<br>Extractor Classification  Classifcation<br>Baseline Loss<br> Model Model Labels<br>ADDA Loss<br>Target Domain<br>Extractor Discriminator  Domain  Adversarial<br> Model Model Labels Loss<br><!-- End of picture text -->

**Figure 7.** ADDA framework for unsupervised domain adaptation. **Blue** : source; **Red** : target; **Purple** : shared. 



<!-- Start of picture text -->
Source Domain<br>Extractor Classification  Classifcation<br>Baseline Loss<br> Model Model Labels<br>Target Domain<br>Entropy<br>Regularization<br>CDAN+E Loss<br>Conditional Discriminator  Domain  Adversarial<br>Feature Fusion Model Labels Loss<br><!-- End of picture text -->

**Figure 8.** CDAN+E framework for unsupervised domain adaptation. **Blue** : source; **Red** : target; **Purple** : shared. 

Overall, two main components are employed depending on the training objective: a label classifier for supervised task learning and a domain discriminator for adversarial domain adaptation. Namely, the label classifier transforms the feature vector of dimension _d_ , produced by the backbone, into a vector of _C_ class logits. The value of _d_ depends on the specific feature extractor employed (e.g., 512 for ResNet-18, 2048 for ResNet-50, and 192 for ViT-Tiny). The corresponding architecture is presented in Table 4. 

**Table 4.** Architecture of the generic label classifier. 

|**Layer Name**|**Type**|**Input Shape**|**Output Shape**|**Param. #**|
|---|---|---|---|---|
|Input|InputLayer|(_d_,)|(_d_,)|0|
|FC1|Dense|(_d_,)|(_d_/2,)|(_d × d_/2) +_d_/2|
|BN1|BatchNorm1d|(_d_/2,)<br>|(_d_/2,)<br>|_d_|
|ReLU1|Activation|(_d_/2,)|(_d_/2,)|0|
|FC2|Dense|(_d_/2,)<br>|(_d_/4,)<br>|(_d_/2_× d_/4) +_d_/4|
|BN2|BatchNorm1d|(_d_/4,)<br>|(_d_/4,)<br>|_d_/2|
|ReLU2|Activation|(_d_/4,)|(_d_/4,)|0|
|Output|Dense|(_d_/4,)|(_C_,)|(_d_/4_× C_) +_C_|



In adversarial training, a domain discriminator is employed to differentiate between source and target samples, thereby promoting domain-invariant feature extraction. Its input dimension _d_ is determined by the underlying method. For instance, DANN and ADDA use the feature vector directly, while CDAN+E utilizes the outer product between 

_Mathematics_ **2025** , _13_ , 2602 

15 of 29 

features and class predictions, yielding an input dimension _d × C_ . The architecture, which mirrors the general structure of the label classifier, is detailed in Table 5. 

**Table 5.** Architecture of the generic domain discriminator. 

|**Layer Name**|**Type**|**Input Shape**|**Output Shape**|**Param. #**|
|---|---|---|---|---|
|Input|InputLayer|(_d_,)|(_d_,)|0|
|FC1|Dense|(_d_,)|(_d_/2,)|(_d × d_/2) +_d_/2|
|BN1|BatchNorm1d|(_d_/2,)|(_d_/2,)|_d_|
|ReLU1|Activation|(_d_/2,)|(_d_/2,)|0|
|FC2|Dense|(_d_/2,)|(_d_/4,)|(_d_/2_× d_/4) +_d_/4|
|BN2|BatchNorm1d|(_d_/4,)|(_d_/4,)|_d_/2|
|ReLU2|Activation|(_d_/4,)|(_d_/4,)|0|
|Output|Dense|(_d_/4,)|(1,)|(_d_/4) +1|



In all experimental scenarios, we report the classification accuracy and its associated standard deviation in the test set of the target domain. Moreover, during training, model performance is periodically evaluated on validation subsets drawn from both source and target domains to monitor intermediate generalization behavior. In this sense, the Accuracy (ACC) measure is defined as follows: 



where _y_ ˆ _i ∈_ **_y_** ˆ and _yi ∈_ **_y_** denote the predicted and ground truth labels, respectively. I( _·_ ) is the indicator function that returns 1 if the condition is true and 0 otherwise. The standard deviation is estimated from the batch-wise accuracies, serving as a proxy for model stability during inference. The Baseline model is trained solely on labeled samples from the source domain and is directly evaluated in the target domain without any adaptation mechanisms. This setting establishes a lower bound for performance under domain shift conditions. 

In addition to quantitative measures, we assess the discriminative quality of the learned feature representations using qualitative techniques. Specifically, we employ the well-known Uniform Manifold Approximation and Projection (UMAP) [52], a nonlinear dimensionality reduction technique to project high-dimensional features into a two-dimensional latent space, enabling visual inspection of inter-domain and inter-class separability [81]. This technique facilitates an empirical evaluation of how well the feature extractor captures semantically consistent structures across domains. To further complement this analysis, we apply the GradCAM++ method to the classifier module in order to visualize spatial attention regions associated with individual predictions [82]. These attention maps provide insight into the decision-making process of the model and support a comparative interpretation of class activation patterns across source and target domains. 

### _3.3. Training Details_ 

The training procedure follows the standard protocol for unsupervised domain adaptation: all labeled data from the source domain are used along with the entire set of unlabeled data from the target domain. The latter approach aims to learn domain-invariant representations without requiring explicit supervision in the target domain. 

All models are trained using the Adam optimizer. For the non-adaptive baseline, models are trained with a fixed learning rate (10<sup>_−_3</sup> for ResNet architectures and 10<sup>_−_4</sup> for ViT-Tiny) and no weight decay. For all domain adaptation methods, a dynamic scheduling scheme is employed for both the learning rate and the adversarial weighting parameter to promote stable convergence and mitigate early overfitting of the discriminator. Both 

_Mathematics_ **2025** , _13_ , 2602 

16 of 29 

epoch hyperparameters are updated according to the relative training progress _p_ ˘ = total_epochs<sup>,</sup> according to the following expressions: 



where the schedule hyper-hyperparameters are updated to _α_ = 20, _β_ = 0.75, and _δ_ = 20 (see Figure 9). 



<!-- Start of picture text -->
10 3 1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>10 4 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0<br>Relative Training Progress - p<br>Learning Rate<br>Adversarial Weight<br><!-- End of picture text -->

**Figure 9.** Dynamic scheduling of learning rate and adversarial weighting factor as functions of the relative training progress _p_ ˘ (horizontal axis, dimensionless, 0–1) and their values in logarithmic scale (vertical axis). **Blue:** learning rate _η_ ( ˘ _p_ ). **Orange:** adversarial weight _λ_ ( ˘ _p_ ), see Equation (29). 

In addition to stratified sampling, the batch size is dynamically adjusted based on the size of the training set ( _N_ ) in each domain, according to the following empirical rule: 



The initial learning rate _η_ 0 was empirically tuned for each model, method, and dataset, typically ranging from 10<sup>_−_3</sup> to 10<sup>_−_5</sup> . Notably, the first stage of ADDA was trained with a fixed learning rate of 10<sup>_−_4</sup> . Furthermore, to adapt the pretrained ViT-Tiny architecture for the lower-resolution Digits dataset (32 _×_ 32), we applied bicubic interpolation to its positional embeddings. This step was necessary to align the spatial dimensions of the pretrained weights (originally for 224 _×_ 224 inputs) with the target image size, enabling effective knowledge transfer. 

Next, to maintain class balance during model training and evaluation, an initial partition is performed into training (70%), validation (15%), and test (15%) subsets. This process is conducted independently for both the source and target domains. To ensure representative subsets, stratified sampling is applied within each partition, preserving the internal class distributions of each domain. In particular, the independent construction of the validation sets enables consistent and comparable evaluation conditions across domains, which is essential in domain adaptation scenarios where distributional shifts may introduce evaluation bias. 

The lower and upper bounds were established empirically. The lower bound ensures the existence of at least 10 mini-batches per epoch, contributing to optimization stability and preventing prohibitively long training times on small datasets. Conversely, the upper bound 

_Mathematics_ **2025** , _13_ , 2602 

17 of 29 

avoids excessively large batches that could destabilize learning or exceed GPU memory capacity. This configuration strikes an effective trade-off between gradient stability and computational efficiency, especially when handling domains of different sizes. 

It is important to note that, since both dataset partitioning and batch size are determined by the number of available samples in each domain, the number of training instances per epoch is not the same across domains. This asymmetry reflects the inherent scale differences between datasets and allows each domain to contribute proportionally to the learning process without enforcing artificial uniformity. 

For all experiments, the kernel bandwidth parameter _σ_ used in the estimation of Rényi’s quadratic entropy was adaptively determined for each training batch using the median heuristic. This common practice involves setting _σ_ as the square root of the median of all pairwise squared Euclidean distances within the combined source and target feature batch, as follows [83]: 



This data-driven approach automates a critical hyperparameter, ensuring that the kernel’s scale is appropriately tailored to the feature distribution, which enhances the stability and effectiveness of the alignment process across domains. 

Moreover, to qualitatively assess the discriminative capacity of the learned features, we apply dimensionality reduction using UMAP, leveraging the GPU-accelerated `cuML` implementation. Unless otherwise stated, the default parameters are set as follows: `n_components` = 2, `n_neighbors` = 80, and `random_state` = 42. Prior to projection, features are normalized with `MinMaxScaler` , which facilitates visual inspection of inter-class and inter-domain separability in the latent space. Also, we employ the GradCAM++ technique via the `torchcam` library to visualize class-specific attention regions within the input images. Representative samples for each class are selected from both source and target domains, and the last convolutional layer of the feature extractor is designated as the target layer. The resulting attention masks are normalized and overlaid on the corresponding images, offering a qualitative perspective on the spatial focus of the model during classification. 

Our experiments were conducted on the Google Colab platform, leveraging a highperformance instance equipped with a NVIDIA (Santa Clara, CA, USA) A100 GPU (40.0 GB of VRAM), 83.5 GB of system RAM, and 235.7 GB of disk storage. For full reproducibility, we set a global random seed of 42 across Python, NumPy 2.0.2, and PyTorch (for both CPU and CUDA) and configured the cuDNN backend to use deterministic algorithms, ensuring consistent results from GPU computations. The development environment was based on `Python 3.11.11` , using `PyTorch 2.1.2` for model training, `cuML 25.02.01` for GPUaccelerated UMAP visualization, and `torchcam 0.4.0` for `GradCAM++` . All source code and datasets are publicly available at: https://github.com/Daprosero/Domain_Adaptation (accessed on 4 July 2025). 

## **4. Results and Discussion** 

### _4.1. Domain Adaption Results_ 

A fundamental objective in domain adaptation is to learn representations that remain invariant under distributional shifts between domains, commonly referred to as covariate shift. A model’s ability to mitigate this challenge is directly reflected in its accuracy on the target domain. To evaluate CREDA’s performance quantitatively, we conducted experiments on three widely adopted benchmark datasets using various backbone architectures. 

_Mathematics_ **2025** , _13_ , 2602 

18 of 29 

In the digit adaptation tasks (see Table 6), CREDA demonstrates state-of-the-art performance, achieving the highest average accuracy with both ResNet-18 (62.65%) and ResNet-50 (64.07%) backbones. It performs exceptionally well in challenging tasks such as M _→_ U (achieving up to 91.77% with ResNet-50), characterized by significant visual disparities. A noteworthy observation arises with the ViT-Tiny backbone. Here, conventional adversarial methods like DANN, ADDA, and CDAN+E experience a significant performance collapse, falling well below the source-only Baseline. This suggests that the dynamics of adversarial training may be unstable or less compatible with the global, patch-based feature space learned by Transformers, in contrast to the hierarchical features of CNNs. Nevertheless, CREDA is markedly less affected by this architectural shift. While the Baseline achieves the top rank in this specific instance, CREDA’s performance (47.23%) remains highly competitive and substantially surpasses other adaptation methods, highlighting its greater architectural robustness. 

Similarly, on the ImageCLEF-DA dataset (see Table 7), CREDA’s superiority is even more pronounced. It consistently achieves top-tier results, securing the highest average accuracy across all three backbones. Critically, with the ViT-Tiny backbone, CREDA (82.41%) is the only adaptation method to decisively outperform the strong Baseline model (80.19%). This again contrasts sharply with other adversarial methods, which either lag or perform on par with the Baseline. This reinforces the hypothesis that CREDA’s Rényi entropybased regularization offers a more stable and effective path to domain alignment than the adversarial objectives of its counterparts, particularly when paired with Transformer architectures. These results suggest that our method more effectively balances domain alignment and the preservation of class discriminability. 

Lastly, on the Office-31 benchmark (see Table 8), CREDA confirms its superiority by achieving the highest average accuracy across all backbones, peaking at 92.96% with ResNet50. The trend of architectural robustness continues, as CREDA again outperforms all other methods with ViT-Tiny, achieving an average accuracy of 89.31% against the Baseline’s 85.46%. The fragility of other methods is particularly evident here, with DANN, ADDA, and CDAN+E suffering catastrophic performance drops (e.g., 20.14% for DANN on D _→_ A), rendering them less effective than a simple no-adaptation approach. This consistently demonstrates CREDA’s ability not only to adapt effectively but also to generalize its mechanism across fundamentally different architectural paradigms, from convolutional to attention-based models. 

**Table 6.** Accuracy (%) on Digits for unsupervised domain adaptation using different backbone architectures. 

|**Model**|**Method**|**M****_→_U**|**M****_→_S**|**U****_→_M**|**U****_→_S**|**S****_→_M**|**S****_→_U**|**Avg**|
|---|---|---|---|---|---|---|---|---|
||Baseline|56.73_±_20.93|22.04_±_14.77|76.64_±_15.49|9.68_±_10.60|69.49_±_16.82|**74.69**_±_**15.88**|51.55_±_15.75|
||DANN|86.20_±_10.64|19.28_±_13.61|80.84_±_13.65|**28.65**_±_**16.12**|72.64_±_15.72|70.66_±_16.33|59.71_±_14.35|
|ResNet-18|ADDA|7.68_±_9.13|30.99_±_16.93|83.30_±_12.67|28.27_±_16.22|**74.32**_±_**15.31**|66.27_±_15.85|54.75_±_13.49|
||CDAN+E|81.99_±_11.92|15.08_±_12.80|25.12_±_15.75|14.19_±_12.72|56.42_±_17.01|66.45_±_17.51|43.21_±_14.62|
||CREDA (ours)|**88.39**_±_**11.20**|**32.92**_±_**17.14**|**86.08**_±_**12.08**|26.29_±_15.79|71.09_±_16.01|71.12_±_15.95|**62.65**_±_**14.69**|
||Baseline|84.45_±_14.08|19.61_±_14.37|64.97_±_17.51|7.99_±_9.56|12.91_±_11.83|66.88_±_18.16|42.80_±_14.25|
||DANN|90.77_±_10.20|36.38_±_16.90|**90.64**_±_**10.38**|**21.01**_±_**14.46**|73.10_±_15.59|69.65_±_15.92|63.59_±_13.91|
|ResNet-50|ADDA|84.00_±_11.52|11.66_±_11.31|39.03_±_17.22|14.72_±_12.28|61.32_±_16.72|63.16_±_15.78|45.65_±_14.14|
||CDAN+E|54.20_±_17.52|17.33_±_13.40|12.73_±_12.04|10.96_±_10.74|30.69_±_16.69|43.88_±_19.33|28.30_±_14.95|
||CREDA (ours)|**91.77**_±_**9.27**|**37.36**_±_**17.78**|80.84_±_14.01|20.52_±_14.15|**76.16**_±_**14.93**|**77.79**_±_**13.54**|**64.07**_±_**13.95**|
||Baseline|67.56_±_18.44|**27.38**_±_**16.10**|**69.82**_±_**17.17**|14.11_±_12.63|**66.28**_±_**17.13**|62.50_±_18.11|**51.28**_±_**16.60**|
||DANN|26.97_±_15.90|7.87_±_9.69|24.84_±_15.65|9.53_±_10.36|7.67_±_9.44|17.28_±_13.45|15.69_±_12.42|
|ViT-Tiny|ADDA|8.96_±_10.40|10.19_±_10.79|13.28_±_12.51|11.19_±_11.18|10.08_±_10.57|2.83_±_5.67|9.42_±_10.19|
||CDAN+E|16.36_±_14.27|10.17_±_10.80|9.74_±_10.50|9.40_±_10.30|9.74_±_10.50|9.14_±_10.67|10.76_±_11.17|
||CREDA (ours)|**75.69**_±_**15.44**|21.51_±_14.37|47.23_±_17.26|**17.56**_±_**13.50**|54.57_±_17.67|**66.82**_±_**17.84**|47.23_±_16.01|



_Mathematics_ **2025** , _13_ , 2602 

19 of 29 

**Table 7.** Accuracy (%) on ImageCLEF-DA for unsupervised domain adaptation using different backbone architectures. 

|**Model**|**Method**|**I****_→_P**|**I****_→_C**|**P****_→_I**|**P****_→_C**|**C****_→_I**|**C**_→_**P**|**Avg**|
|---|---|---|---|---|---|---|---|---|
||Baseline|58.00_±_21.71|76.83_±_21.52|68.00_±_19.30|76.50_±_19.05|49.83_±_24.44|38.67_±_25.60|61.31_±_21.94|
||DANN|60.00_±_25.00|**85.56**_±_**13.55**|66.67_±_16.43|78.89_±_18.04|**76.67**_±_**17.78**|57.78_±_23.74|70.93_±_19.09|
|ResNet-18|ADDA|**68.89**_±_**20.87**|77.78_±_11.10|71.11_±_19.09|**81.11**_±_**16.39**|74.44_±_14.56|58.89_±_21.62|72.04_±_17.27|
||CDAN+E|56.67_±_23.31|62.22_±_15.50|65.56_±_16.71|58.89_±_16.28|68.89_±_21.62|47.78_±_22.90|60.00_±_19.39|
||CREDA (ours)|66.67_±_19.31|82.22_±_17.24|**77.78**_±_**16.28**|80.00_±_18.56|73.33_±_19.22|**62.22**_±_**18.81**|**73.70**_±_**18.24**|
||Baseline|46.50_±_26.20|56.83_±_33.03|69.67_±_20.05|80.17_±_16.71|55.50_±_20.47|43.00_±_23.72|58.61_±_23.36|
||DANN|**77.78**_±_**17.13**|90.00_±_9.42|**82.22**_±_**13.41**|86.67_±_16.10|**86.67**_±_**10.66**|**75.56**_±_**11.72**|**83.15**_±_**13.07**|
|ResNet-50|ADDA|80.00_±_14.60|80.00_±_18.84|82.22_±_17.24|82.22_±_18.04|85.56_±_11.25|73.33_±_13.06|80.56_±_15.51|
||CDAN+E|48.89_±_25.19|60.00_±_23.31|58.89_±_14.43|52.22_±_24.11|56.67_±_13.59|42.22_±_23.13|53.15_±_20.63|
||CREDA (ours)|72.22_±_18.04|**92.22**_±_**9.91**|**82.22**_±_**13.41**|**90.00**_±_**12.07**|85.56_±_12.45|67.78_±_13.41|81.67_±_13.22|
||Baseline|73.00_±_24.32|92.50_±_11.44|**85.67**_±_**15.47**|**88.00**_±_**12.57**|77.17_±_25.62|64.83_±_29.69|80.19_±_19.85|
||DANN|70.00_±_22.06|91.11_±_6.15|81.11_±_15.50|86.67_±_11.92|83.33_±_16.96|62.22_±_15.50|79.07_±_14.68|
|ViT-Tiny|ADDA|68.89_±_18.04|90.00_±_7.77|78.89_±_12.45|85.56_±_12.45|7.78_±_9.91|48.89_±_22.19|63.33_±_13.80|
||CDAN+E|70.00_±_18.07|87.78_±_9.91|72.22_±_15.50|78.89_±_15.84|36.67_±_20.64|24.44_±_13.93|61.67_±_15.65|
||CREDA (ours)|**77.78**_±_**18.72**|**93.33**_±_**8.43**|80.00_±_13.59|86.67_±_15.19|**87.78**_±_**11.25**|**68.89**_±_**18.04**|**82.41**_±_**14.20**|



To provide a robust statistical assessment of our method’s consistency and superiority, we conducted a Friedman test on the accuracy ranks across all nine experimental configurations (three datasets _×_ three backbones). The test revealed a statistically significant difference among the methods’ performances ( _χ_<sup>2</sup> (4) = 23.21, _p <_ 1.15 _×_ 10<sup>_−_4</sup> ), thus allowing us to reject the null hypothesis that all approaches perform equally. This result provides strong evidence that the observed differences in performance are not due to random chance. 

Table 9 shows that CREDA achieves the best (lowest) average rank of 1.22. Furthermore, its performance stability is underscored by a remarkably low standard deviation ( _±_ 0.44), the lowest among all evaluated methods. This indicates that CREDA consistently ranked at or near the top, irrespective of the dataset or backbone architecture. In contrast, methods like the Baseline (3.44 _±_ 1.42) exhibit much higher variance, suggesting their performance is less stable across different settings. This statistical validation robustly confirms that CREDA’s leading performance is not an artifact of specific experimental conditions but rather a consistent and significant advantage across a diverse range of domains and model architectures, including the challenging Transformer-based setups where other adaptation techniques falter. 

**Table 8.** Accuracy (%) on Office-31 for unsupervised domain adaptation using different backbone architectures. 

|**Model**|**Method**|**A****_→_W**|**A****_→_D**|**W****_→_A**|**W****_→_D**|**D****_→_A**|**D****_→_W**|**Avg**|
|---|---|---|---|---|---|---|---|---|
||Baseline|50.51_±_29.45|55.41_±_25.37|54.91_±_34.50|96.82_±_7.98|46.56_±_34.31|78.98_±_28.90|63.86_±_26.75|
||DANN|73.33_±_15.81|87.50_±_12.50|67.36_±_16.68|**100.00**_±_**0.00**|51.39_±_17.09|84.44_±_12.29|77.34_±_12.40|
|ResNet-18|ADDA|62.22_±_26.71|87.50_±_12.50|54.17_±_19.65|**100.00**_±_**0.00**|59.72_±_17.45|84.44_±_9.41|74.68_±_14.29|
||CDAN+E|64.44_±_14.72|75.00_±_12.50|50.69_±_21.64|**100.00**_±_**0.00**|51.39_±_16.54|88.89_±_12.29|71.74_±_12.95|
||CREDA (ours)|**82.22**_±_**18.82**|**91.67**_±_**14.43**|**74.31**_±_**17.40**|**100.00**_±_**0.00**|**74.31**_±_**17.40**|**93.33**_±_**10.46**|**85.97**_±_**13.09**|
||Baseline|48.14_±_29.65|46.50_±_31.67|53.44_±_29.99|76.43_±_24.05|53.86_±_36.86|90.51_±_21.12|61.48_±_28.89|
||DANN|**88.89**_±_**9.41**|95.83_±_7.22|**91.67**_±_**8.57**|**100.00**_±_**0.00**|81.94_±_13.71|91.11_±_10.21|91.57_±_8.19|
|ResNet-50|ADDA|**88.89**_±_**9.41**|95.83_±_7.22|88.89_±_9.48|**100.00**_±_**0.00**|84.72_±_13.93|91.11_±_10.21|91.57_±_8.37|
||CDAN+E|73.33_±_15.81|75.00_±_12.50|70.14_±_19.71|95.83_±_7.22|52.08_±_15.01|68.89_±_20.41|72.55_±_15.11|
||CREDA (ours)|86.67_±_11.18|**100.00**_±_**0.00**|**91.67**_±_**8.57**|**100.00**_±_**0.00**|**86.11**_±_**12.04**|**93.33**_±_**10.46**|**92.96**_±_**7.04**|
||Baseline|85.42_±_21.61|86.62_±_14.96|80.27_±_22.52|**100.00**_±_**0.00**|66.18_±_28.50|**94.24**_±_**12.00**|85.46_±_16.60|
||DANN|53.33_±_18.62|75.00_±_25.00|19.44_±_12.29|91.67_±_7.22|20.14_±_12.23|91.11_±_11.23|58.45_±_14.43|
|ViT-Tiny|ADDA|80.00_±_15.31|**95.83**_±_**7.22**|12.50_±_10.50|**100.00**_±_**0.00**|15.28_±_13.93|55.56_±_16.61|59.86_±_10.60|
||CDAN+E|17.78_±_24.58|62.50_±_12.50|20.14_±_12.96|**100.00**_±_**0.00**|21.53_±_12.72|8.89_±_10.21|38.47_±_12.16|
||CREDA (ours)|**93.33**_±_**6.85**|91.67_±_7.22|**88.19**_±_**16.86**|**100.00**_±_**0.00**|**71.53**_±_**16.50**|91.11_±_20.41|**89.31**_±_**11.31**|



_Mathematics_ **2025** , _13_ , 2602 

20 of 29 

**Table 9.** Average classification rank of all methods across datasets and model architectures. Ranks are assigned per block (row) based on average accuracy. The final row presents the mean rank ± standard deviation for each method. The Friedman test confirms a significant difference in performance ( _χ_<sup>2</sup> = 23.21, _p <_ 1.15 _×_ 10<sup>_−_4</sup> ). 

|**Dataset**|**Backbone**|**Baseline**|**DANN**|**ADDA**|**CDAN+E**|**CREDA (Ours)**|
|---|---|---|---|---|---|---|
||ResNet-18|4.0|2.0|3.0|5.0|1.0|
|Digits|ResNet-50|4.0|2.0|3.0|5.0|1.0|
||ViT-Tiny|1.0|3.0|5.0|4.0|2.0|
||ResNet-18|4.0|3.0|2.0|5.0|1.0|
|ImageCLEF-DA|ResNet-50|4.0|1.0|3.0|5.0|2.0|
||ViT-Tiny|2.0|3.0|4.0|5.0|1.0|
||ResNet-18|5.0|2.0|3.0|4.0|1.0|
|Office-31|ResNet-50|5.0|2.0|3.0|4.0|1.0|
||ViT-Tiny|2.0|4.0|3.0|5.0|1.0|
|**Mean Rank ± Std**|–|3.4_±_1.4|2.4_±_0.9|3.2_±_0.8|4.7_±_0.5|**1.2**_±_**0.4**|



### _4.2. Interpretability Results_ 

To clarify the reasons for these performance disparities, it is crucial to first examine the inherent complexity of the data domains. Figure 10 presents the 2D UMAP projections of the original feature space, visualized independently for each domain. These plots reveal a fundamental challenge that extends beyond domain shift: the limited class separability within individual domains. This limitation is particularly pronounced in complex datasets such as ImageCLEF-DA and Office-31, where class instances (depicted by distinct colors) exhibit significant entanglement, forming dense and unstructured distributions. Such inherent visual similarity among categories not only complicates classification within the source domain but also serves as a principal source of noisy pseudo-labels in the target domain during unsupervised adaptation. Consequently, a robust domain adaptation strategy must not only align cross-domain distributions but also construct feature representations that enhance inter-class discrimination. 



<!-- Start of picture text -->
MNIST USPS SVHN<br>Imagenet Photo Caltech<br>Amazon Webcam DSLR<br>Class 0 Class 2 Class 4 Class 6 Class 8 Class 10<br>Class 1 Class 3 Class 5 Class 7 Class 9 Class 11<br>Digits<br>ImageCLEF-DA<br>Office-31<br><!-- End of picture text -->

**Figure 10.** Two-dimensional UMAP projections of original feature representations before domain adaptation. **Rows:** evaluated benchmarks. **Columns:** domains within each benchmark. 

_Mathematics_ **2025** , _13_ , 2602 

21 of 29 

Building upon this analysis, Figure 11 illustrates how different adaptation techniques address these structural challenges, visualized through UMAP projections of the learned latent spaces. The first column depicts the initial state prior to training, highlighting both the pronounced domain gap (e.g., M _→_ U) and the poor semantic organization (e.g., I _→_ C). The Baseline model, trained exclusively on source data, fails to bridge this gap, maintaining a clear division between domains. In contrast, adversarial methods like DANN and ADDA achieve some domain alignment, but often at the expense of class coherence, resulting in fragmented (as seen in M _→_ U) and disordered representations across all tasks. While CDAN+E introduces a modest improvement in structural consistency, significant inter-class dispersion remains. Ultimately, CREDA yields a markedly superior configuration: it not only facilitates seamless domain integration—evidenced by the homogeneous blending of source and target samples—but also preserves (M _→_ U) and, notably, enhances (I _→_ C, W _→_ D) class-wise separability, as demonstrated by the emergence of compact, well-defined clusters from initially unstructured feature spaces. This outcome provides a visual explanation for CREDA’s superior quantitative performance, indicating its ability to balance the removal of spurious domain-specific cues with the preservation and recovery of underlying semantic structure. 

Having established CREDA’s capacity to address covariate shift, we next assess whether the learned representations preserve semantic coherence under concept shift, where object appearance changes substantially across domains. In this context, Figure 12 presents UMAP projections with embedded images to qualitatively examine the model’s ability to cluster semantically related concepts. 

The results indicate that CREDA learns a semantically rich feature space that transcends superficial variability. For instance, in the M _→_ U task, the model accurately groups digits despite substantial stylistic differences, as seen in the clusters corresponding to digits 6, 0, and 4. In the I _→_ C task, it successfully groups semantically similar but visually diverse objects, forming distinct clusters for categories like airplanes and bottles despite variations in perspective and background. Similarly, in the W _→_ D task, objects such as keyboards and mugs are grouped according to their semantic identity, overcoming differences in image quality. Altogether, these visualizations demonstrate that CREDA not only aligns domains but also constructs a feature space in which proximity reflects conceptual similarity—an essential attribute for robust generalization in real-world applications. 



<!-- Start of picture text -->
Original Baseline DANN ADDA CDAN+E CREDA<br>U<br>M<br>C<br>I<br>D<br>W<br><!-- End of picture text -->

**Figure 11.** UMAP projections of the learned feature representations across domain adaptation methods, with the source domain shown in blue and the target domain in orange. **Rows:** datasets used in the evaluation. **Columns:** compared adaptation models. 

_Mathematics_ **2025** , _13_ , 2602 

22 of 29 



<!-- Start of picture text -->
M U I C W D<br><!-- End of picture text -->

**Figure 12.** UMAP projections of learned feature representations under the CREDA model, with input images overlaid, where source domain samples appear in blue and target domain samples in orange. **Left** : Digits. **Middle** : ImageCLEF-DA. **Right** : Office-31. 

Finally, to reinforce the model’s reliability, it is essential not only to demonstrate high accuracy and semantic coherence, but also to ensure that its predictions are grounded in interpretable reasoning. In other words, it must be verified that decisions are driven by relevant visual cues rather than spurious correlations. 

To address this, we employ Grad-CAM++, with the results shown in Figure 13. The heatmaps reveal strong semantic consistency: regardless of the domain, the model focuses attention on canonical and representative regions of the object, such as the face in a portrait or the main structural components of a vehicle. This confirms that CREDA does not rely on superficial distribution alignment, but rather performs deep and meaningful semantic knowledge transfer. These findings not only enhance trust in the model’s predictions but also establish CREDA as a transparent and robust solution for domain adaptation, strengthening the interpretability and reliability of its outputs. 



<!-- Start of picture text -->
Clase 0 Clase 1 Clase 2<br>ImageCLEF-DA<br>Office-31<br><!-- End of picture text -->

**Figure 13.** Class-wise visual explanations under the CREDA model. Each pair of images shows the source domain on the **left** and the corresponding target domain on the **right** . Heatmaps highlight the most salient regions contributing to the predicted class. 

### _4.3. Training and Inference Time Analysis_ 

To assess practical viability, we measured training and inference time on the M _→_ S task, selected for being the most extensive dataset combination. ResNet-50 was used as the feature extractor due to its higher computational demand relative to the other backbones, 

_Mathematics_ **2025** , _13_ , 2602 

23 of 29 

offering a conservative estimate of resource requirements. As shown in Figure 14, ADDA incurs the highest training cost due to its two-phase architecture, whereas single-stage methods (DANN, CDAN+E, and CREDA) introduce only a marginal overhead compared to the Baseline. Regarding inference, all adapted models were highly efficient. Notably, DANN, CDAN+E, and CREDA demonstrated slightly faster inference than the Baseline and ADDA, potentially because the learned domain-invariant features streamline the forward pass. This analysis confirms that CREDA offers a compelling trade-off, delivering superior accuracy with a manageable training cost while maintaining efficient inference speeds suitable for real-world deployment. 



<!-- Start of picture text -->
140<br>0.020<br>120<br>0.019<br>100<br>80 0.018<br>60<br>0.017<br>40<br>0.016<br>20<br>0.015<br>0<br>Baseline DANN ADDA CDAN+E CREDA<br>Training Time (seconds) Inference Time (seconds)<br><!-- End of picture text -->

**Figure 14.** Training and inference time comparison across domain adaptation methods. The left axis shows training time per epoch, while the right axis shows average inference time per sample. 

### _4.4. Limitations_ 

Despite the robust performance of the CREDA framework on unsupervised domain adaptation tasks, several limitations must be acknowledged. These, in turn, present pertinent avenues for future research. While CREDA demonstrates superior performance even when implemented on deeper CNNs or ViT-based architectures, a comprehensive investigation is required to fully characterize its scaling properties in large-scale or multiresolution contexts. Secondly, a singular hyperparameter tuning strategy was employed across all tasks, thereby precluding domain-pair-specific optimization. The incorporation of automated search schemes for adaptation could potentially enhance performance and generalization, albeit at an increased computational cost [84]. Thirdly, the combination of the classification loss and the Rényi divergence-based regularization relies on a static weighting coefficient. Exploring an adaptive normalization method for the loss functions could foster more stable training dynamics by balancing the magnitudes of the gradients. Moreover, as the regularizer is contingent upon kernel-based estimations, the model’s performance exhibits sensitivity to the kernel bandwidth. Although the median heuristic was employed to set this bandwidth at each training step, such a data-driven strategy may not generalize optimally across all domain pairs or distributions, warranting further exploration of adaptive kernel selection schemes. 

In particular, CREDA’s performance reveals its limitations, particularly in scenarios with extreme domain shifts. Quantitatively, the method’s effectiveness degrades most significantly on adaptation tasks involving the SVHN (S) dataset, such as M→S and U→S, where it achieves its lowest absolute accuracies (see Table 6). The severe performance drop of the source-only Baseline on these tasks confirms that the domain gap—transitioning 

_Mathematics_ **2025** , _13_ , 2602 

24 of 29 

from clean, centered digits to cluttered, real-world house numbers—is exceptionally large. This suggests that CREDA, while robust, struggles when the target domain introduces fundamental changes in image composition, including complex backgrounds, color variations, and distracting neighboring elements, which are not present in the source domain. Qualitatively, this failure mode can be attributed to the quality of the initial pseudo-labels. In extreme-shift scenarios, the classifier, trained only on source data, produces target pseudo-labels that are either confidently wrong or universally low-confidence. For instance, an SVHN digit ‘1’ with artifacts may be confidently misclassified as a ‘7’, or a ‘3’ with poor lighting as an ‘8’. While our entropy-based weighting is designed to mitigate noise, it cannot overcome a situation where the initial class-conditional signal is systematically corrupted. Consequently, CREDA’s primary limitation arises when the domain gap is so vast that it prevents the model from forming a reasonably accurate initial estimate of the target domain’s semantic structure, thereby undermining the effectiveness of the class-conditional alignment mechanism. 

## **5. Conclusions** 

This work introduced a novel domain adaptation framework, termed Conditional Rényi _α_ -Entropy Domain Adaptation (CREDA), a deep learning-based strategy integrating kernel-based conditional alignment from a matrix-based formulation of Rényi’s quadratic entropy. CREDA is structured around three key components. First, a deep feature extractor is used to learn domain-invariant representations by leveraging labeled source data and unlabeled target data. Second, an entropy-weighted strategy attenuates the influence of low-confidence pseudo-labels, thereby enhancing robustness in ambiguous regions. Third, a class-conditional alignment loss, expressed as a Rényi divergence, is introduced to promote semantic consistency across domains within the latent representation space. In contrast to supervised or semi-supervised approaches, the proposed method does not require labels in the target domain, making it particularly suitable for scenarios where annotation is costly or unavailable. Moreover, our class-wise alignment is formulated in a non-parametric and differentiable manner by leveraging kernel-based information potentials, enabling the preservation of semantic structure across domains. 

Experimental results across diverse visual adaptation scenarios demonstrate that CREDA consistently outperforms conventional methods such as DANN, ADDA, and CDAN+E in terms of predictive accuracy, representational quality, and interpretability. In particular, CREDA achieves the highest average accuracy across all datasets and architectures, with noticeable improvements when using deeper CNNs (ResNet-50) and attention-based models (ViT-Tiny). While most adversarial approaches experience performance degradation in these settings, CREDA remains robust and effective, as evidenced by the results presented in this study. Notably, CREDA maintains class separability even under complex distribution shifts and when the predicted labels in the target domain exhibit low confidence. The integration of UMAP- and GradCAM++-based visualizations offers valuable insights into the learned representations, reinforcing its applicability in real-world settings where traceability and semantic coherence are critical. From an implementation standpoint, CREDA does not require modifications to the classification loss function. Its confidence-aware weighting scheme and class-conditional regularization enhance robustness to pseudo-label noise and class imbalance. Moreover, its modular architecture facilitates seamless integration into existing deep learning pipelines. 

As future work, we aim to test CREDA on larger-scale datasets. Also, we plan to extend CREDA to multi-source and continual domain adaptation settings, where domain shifts occur either simultaneously or sequentially. Attention-based class-conditioned alignment across multiple source domains has been shown to mitigate negative transfer and effec- 

_Mathematics_ **2025** , _13_ , 2602 

25 of 29 

tively address class imbalance [85]. Second, we plan to incorporate class-conditional kernel alignment and attention-guided feature disentanglement to improve both interpretability and discriminative alignment, particularly in contexts characterized by subtle inter-class distinctions or limited labeled data. Additionally, exploring temporal or streaming variants of CREDA could prove beneficial in online adaptation scenarios, where data arrives sequentially and models must adapt incrementally. Recent advances in attention-aware class-conditioned alignment suggest that these mechanisms yield robust feature representations and highlight relevant discriminative regions in multi-source adaptation [86]. Finally, while CREDA was conceived for the standard unsupervised adaptation setting, its extension to more challenging scenarios, such as few-shot or source-free adaptation, remains uninvestigated [87]. Addressing these limitations would not only enhance the robustness of the proposed framework but also broaden its applicability to more complex transfer learning problems. 

**Author Contributions:** Conceptualization, D.A.P.-R., A.M.Á.-M. and G.C.-D.; data curation, D.A.P.-R.; methodology, D.A.P.-R., A.M.Á.-M. and G.C.-D.; project administration, A.M.Á.-M.; supervision, A.M.Á.-M. and G.C.-D.; resources, D.A.P.-R. and A.M.Á.-M. All authors have read and agreed to the published version of the manuscript. 

**Funding:** Under grants provived by the project: “Prototipo funcional de lengua electrónica para la identificación de sabores en cacao fino de origen colombiano”, funded by Minciencias-82729ICETEX 2022-0740 and Casa Luker. Also, A.M. Alvarez thanks the following project: “Aprendizaje de máquina cuántico utilizando espines electrónicos”, Hermes-62836, funded by Universidad Nacional de Colombia and Universidad de Caldas. 

**Data Availability Statement:** The publicly available dataset analyzed in this study and our Python codes can be found at https://github.com/Daprosero/Domain_Adaptation (accessed on 4 July 2025). 

**Conflicts of Interest:** The authors declare no conflicts of interest. 

## **References** 

1. Lu, X.; Yao, X.; Jiang, Q.; Shen, Y.; Xu, F.; Zhu, Q. Remaining useful life prediction model of cross-domain rolling bearing via dynamic hybrid domain adaptation and attention contrastive learning. _Comput. Ind._ **2025** , _164_ , 104172. [CrossRef] 

2. Wu, H.; Shi, C.; Yue, S.; Zhu, F.; Jin, Z. Domain Adaptation Network Based on Multi-Level Feature Alignment Constraints for Cross Scene Hyperspectral Image Classification. _Knowl.-Based Syst._ **2025** , 113972. [CrossRef] 

3. Huang, X.Y.; Chen, S.Y.; Wei, C.S. Enhancing Low-Density EEG-Based Brain-Computer Interfacing With Similarity-Keeping Knowledge Distillation. _IEEE Trans. Emerg. Top. Comput. Intell._ **2023** , _8_ , 1156–1166. [CrossRef] 

4. Jiang, J.; Zhao, S.; Zhu, J.; Tang, W.; Xu, Z.; Yang, J.; Liu, G.; Xing, T.; Xu, P.; Yao, H. Multi-source domain adaptation for panoramic semantic segmentation. _Inf. Fusion_ **2025** , _117_ , 102909. [CrossRef] 

5. Imtiaz, M.N.; Khan, N. Towards Practical Emotion Recognition: An Unsupervised Source-Free Approach for EEG Domain Adaptation. _arXiv_ **2025** , arXiv:2504.03707. 

6. Wang, J.; Lan, C.; Liu, C.; Ouyang, Y.; Qin, T.; Lu, W.; Chen, Y.; Zeng, W.; Yu, P.S. Generalizing to unseen domains: A survey on domain generalization. _IEEE Trans. Knowl. Data Eng._ **2022** , _35_ , 8052–8072. [CrossRef] 

7. Galappaththige, C.J.; Baliah, S.; Gunawardhana, M.; Khan, M.H. Towards generalizing to unseen domains with few labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Seattle, WA, USA, 16–22 June 2024; pp. 23691–23700. 

8. Zhu, H.; Bai, J.; Li, N.; Li, X.; Liu, D.; Buckeridge, D.L.; Li, Y. FedWeight: Mitigating covariate shift of federated learning on electronic health records data through patients re-weighting. _npj Digit. Med._ **2025** , _8_ , 286. [CrossRef] 

9. Li, L.; Zhang, X.; Liang, J.; Chen, T. Addressing Domain Shift via Imbalance-Aware Domain Adaptation in Embryo Development Assessment. _arXiv_ **2025** , arXiv:2501.04958. [CrossRef] 

10. Yuksel, G.; Kamps, J. Interpretability Analysis of Domain Adapted Dense Retrievers. _arXiv_ **2025** , arXiv:2501.14459. 

11. Adachi, K.; Yamaguchi, S.; Kumagai, A.; Hamagami, T. Test-time Adaptation for Regression by Subspace Alignment. _arXiv_ **2024** , arXiv:2410.03263. [CrossRef] 

_Mathematics_ **2025** , _13_ , 2602 

26 of 29 

12. Zhang, G.; Zhou, T.; Cai, Y. CORAL-based Domain Adaptation Algorithm for Improving the Applicability of Machine Learning Models in Detecting Motor Bearing Failures. _J. Comput. Methods Eng. Appl._ **2023** , _3_ , 1–17. [CrossRef] 

13. Wang, J.; Feng, W.; Chen, Y.; Yu, H.; Huang, M.; Yu, P.S. Visual domain adaptation with manifold embedded distribution alignment. In Proceedings of the 26th ACM International Conference on Multimedia, Seoul, Republic of Korea, 22–26 October 2018; pp. 402–410. 

14. Yun, K.; Satou, H. GAMA++: Disentangled Geometric Alignment with Adaptive Contrastive Perturbation for Reliable Domain Transfer. _arXiv_ **2025** , arXiv:2505.15241. 

15. Sanodiya, R.K.; Yao, L. A subspace based transfer joint matching with Laplacian regularization for visual domain adaptation. _Sensors_ **2020** , _20_ , 4367. [CrossRef] 

16. Wei, F.; Xu, X.; Jia, T.; Zhang, D.; Wu, X. A multi-source transfer joint matching method for inter-subject motor imagery decoding. _IEEE Trans. Neural Syst. Rehabil. Eng._ **2023** , _31_ , 1258–1267. [CrossRef] 

17. Battu, R.S.; Agathos, K.; Monsalve, J.M.L.; Worden, K.; Papatheou, E. Combining transfer learning and numerical modelling to deal with the lack of training data in data-based SHM. _J. Sound Vib._ **2025** , _595_ , 118710. [CrossRef] 

18. Yano, M.O.; Figueiredo, E.; da Silva, S.; Cury, A. Foundations and applicability of transfer learning for structural health monitoring of bridges. _Mech. Syst. Signal Process._ **2023** , _204_ , 110766. [CrossRef] 

19. Liang, S.; Li, L.; Zu, W.; Feng, W.; Hang, W. Adaptive deep feature representation learning for cross-subject EEG decoding. _BMC Bioinform._ **2024** , _25_ , 393. [CrossRef] 

20. Chen, G.; Xiang, D.; Liu, T.; Xu, F.; Fang, K. Deep discriminative domain adaptation network considering sampling frequency for cross-domain mechanical fault diagnosis. _Expert Syst. Appl._ **2025** , _280_ , 127296. [CrossRef] 

21. Wei, G.; Lan, C.; Zeng, W.; Chen, Z. Metaalign: Coordinating domain alignment and classification for unsupervised domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Nashville, TN, USA, 20–25 June 2021; pp. 16643–16653. 

22. Zhang, Y.; Wang, X.; Liang, J.; Zhang, Z.; Wang, L.; Jin, R.; Tan, T. Free lunch for domain adversarial training: Environment label smoothing. _arXiv_ **2023** , arXiv:2302.00194. [CrossRef] 

23. Lu, M.; Huang, Z.; Zhao, Y.; Tian, Z.; Liu, Y.; Li, D. DaMSTF: Domain adversarial learning enhanced meta self-training for domain adaptation. _arXiv_ **2023** , arXiv:2308.02753. 

24. Wu, Y.; Spathis, D.; Jia, H.; Perez-Pozuelo, I.; Gonzales, T.I.; Brage, S.; Wareham, N.; Mascolo, C. Udama: Unsupervised domain adaptation through multi-discriminator adversarial training with noisy labels improves cardio-fitness prediction. In Proceedings of the Machine Learning for Healthcare Conference, New York, NY, USA, 11–12 August 2023; PMLR: Cambridge, MA, USA, 2023; pp. 863–883. 

25. Mehra, A.; Kailkhura, B.; Chen, P.Y.; Hamm, J. Understanding the limits of unsupervised domain adaptation via data poisoning. _Adv. Neural Inf. Process. Syst._ **2021** , _34_ , 17347–17359. 

26. Zhu, Y.; Zhuang, F.; Wang, J.; Chen, J.; Shi, Z.; Wu, W.; He, Q. Multi-representation adaptation network for cross-domain image classification. _Neural Netw._ **2019** , _119_ , 214–221. [CrossRef] [PubMed] 

27. Madadi, Y.; Seydi, V.; Sun, J.; Chaum, E.; Yousefi, S. Stacking Ensemble Learning in Deep Domain Adaptation for Ophthalmic Image Classification. In _Ophthalmic Medical Image Analysis: Proceedings of the 8th International Workshop, OMIA 2021, Held in Conjunction with MICCAI 2021, Strasbourg, France, 27 September 2021, Proceedings 8_ ; Springer: Berlin/Heidelberg, Germany, 2021; pp. 168–178. 

28. Zhu, Y.; Zhuang, F.; Wang, J.; Ke, G.; Chen, J.; Bian, J.; Xiong, H.; He, Q. Deep subdomain adaptation network for image classification. _IEEE Trans. Neural Netw. Learn. Syst._ **2020** , _32_ , 1713–1722. [CrossRef] 

29. Li, X.; Chen, H.; Li, S.; Wei, D.; Zou, X.; Si, L.; Shao, H. Multi-kernel weighted joint domain adaptation network for cross-condition fault diagnosis of rolling bearings. _Reliab. Eng. Syst. Saf._ **2025** , _261_ , 111109. [CrossRef] 

30. Xiao, L.; Xu, J.; Zhao, D.; Wang, Z.; Wang, L.; Nie, Y.; Dai, B. Self-supervised domain adaptation with consistency training. In Proceedings of the 2020 25th International Conference on Pattern Recognition (ICPR), Milan, Italy, 10–15 January 2021; IEEE: Piscataway, NJ, USA, 2021; pp. 6874–6880. 

31. Wang, R.; Wu, Z.; Weng, Z.; Chen, J.; Qi, G.J.; Jiang, Y.G. Cross-domain contrastive learning for unsupervised domain adaptation. _IEEE Trans. Multimed._ **2022** , _25_ , 1665–1673. [CrossRef] 

32. Kang, G.; Jiang, L.; Yang, Y.; Hauptmann, A.G. Contrastive adaptation network for unsupervised domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Long Beach, CA, USA, 15–20 June 2019; pp. 4893–4902. 

33. Jia, M.; Tang, L.; Chen, B.C.; Cardie, C.; Belongie, S.; Hariharan, B.; Lim, S.N. Visual prompt tuning. In Proceedings of the European Conference on Computer Vision, Tel Aviv, Israel, 23–27 October 2022; Springer: Cham, Switzerland, 2022; pp. 709–727. 

34. Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A.C.; Lo, W.Y.; et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Paris, France, 1–6 October 2023; pp. 4015–4026. 

_Mathematics_ **2025** , _13_ , 2602 

27 of 29 

35. Chen, H.; Chen, H.; Zhao, Z.; Han, K.; Zhu, G.; Zhao, Y.; Du, Y.; Xu, W.; Shi, Q. An overview of domain-specific foundation model: Key technologies, applications and challenges. _arXiv_ **2024** , arXiv:2409.04267. [CrossRef] 

36. Chen, L.; Chen, H.; Wei, Z.; Jin, X.; Tan, X.; Jin, Y.; Chen, E. Reusing the task-specific classifier as a discriminator: Discriminator-free adversarial domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New Orleans, LA, USA, 18–24 June 2022; pp. 7181–7190. 

37. Xiao, R.; Liu, Z.; Wu, B. Teacher-student competition for unsupervised domain adaptation. In Proceedings of the 2020 25th International Conference on Pattern Recognition (ICPR), Milan, Italy, 10–15 January 2021; IEEE: Piscataway, NJ, USA, 2021; pp. 8291–8298. 

38. Choi, E.; Rodriguez, J.; Young, E. An In-Depth Analysis of Adversarial Discriminative Domain Adaptation for Digit Classification. _arXiv_ **2024** , arXiv:2412.19391. [CrossRef] 

39. Lu, W.; Luu, R.K.; Buehler, M.J. Fine-tuning large language models for domain adaptation: Exploration of training strategies, scaling, model merging and synergistic capabilities. _npj Comput. Mater._ **2025** , _11_ , 84. [CrossRef] 

40. Kumar, A.; Raghunathan, A.; Jones, R.; Ma, T.; Liang, P. Fine-tuning can distort pretrained features and underperform out-ofdistribution. _arXiv_ **2022** , arXiv:2202.10054. [CrossRef] 

41. Liu, Y.; Wong, W.; Liu, C.; Luo, X.; Xu, Y.; Wang, J. Mutual Learning for SAM Adaptation: A Dual Collaborative Network Framework for Source-Free Domain Transfer. In Proceedings of the 42nd International Conference on Machine Learning (ICML), Vancouver, BC, Canada, 13–19 July 2025; Poster presentation. 

42. Gao, Y.; Baucom, B.; Rose, K.; Gordon, K.; Wang, H.; Stankovic, J.A. E-ADDA: Unsupervised Adversarial Domain Adaptation Enhanced by a New Mahalanobis Distance Loss for Smart Computing. In Proceedings of the 2023 IEEE International Conference on Smart Computing (SMARTCOMP), Nashville, TN, USA, 26–30 June 2023; IEEE: Piscataway, NJ, USA, 2023; pp. 172–179. 

43. Dan, J.; Jin, T.; Chi, H.; Dong, S.; Xie, H.; Cao, K.; Yang, X. Trust-aware conditional adversarial domain adaptation with feature norm alignment. _Neural Netw._ **2023** , _168_ , 518–530. [CrossRef] [PubMed] 

44. Rao, K.; Harris, C.; Irpan, A.; Levine, S.; Ibarz, J.; Khansari, M. Rl-cyclegan: Reinforcement learning aware simulation-to-real. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Seattle, WA, USA, 13–19 June 2020; pp. 11157–11166. 

45. Tang, P.; Peng, L.; Yan, R.; Shi, H.; Yao, G.; Liu, C.; Li, J.; Zhang, Y. Domain adaptation via mutual information maximization for handwriting recognition. In Proceedings of the ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, 23–27 May 2022; IEEE: Piscataway, NJ, USA, 2022; pp. 2300–2304. 

46. Saito, K.; Kim, D.; Sclaroff, S.; Darrell, T.; Saenko, K. Semi-supervised domain adaptation via minimax entropy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Seoul, Republic of Korea, 27 October–2 November 2019; pp. 8050–8058. 

47. Chen, J.; Zhang, Z.; Xie, X.; Li, Y.; Xu, T.; Ma, K.; Zheng, Y. Beyond mutual information: Generative adversarial network for domain adaptation using information bottleneck constraint. _IEEE Trans. Med. Imaging_ **2021** , _41_ , 595–607. [CrossRef] 

48. Chang, W.G.; You, T.; Seo, S.; Kwak, S.; Han, B. Domain-specific batch normalization for unsupervised domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Long Beach, CA, USA, 15–20 June 2019; pp. 7354–7362. 

49. Wang, H.; Naidu, R.; Michael, J.; Kundu, S.S. Ss-cam: Smoothed score-cam for sharper visual feature localization. _arXiv_ **2020** , arXiv:2006.14255. 

50. Mirkes, E.M.; Bac, J.; Fouché, A.; Stasenko, S.V.; Zinovyev, A.; Gorban, A.N. Domain adaptation principal component analysis: Base linear method for learning with out-of-distribution data. _Entropy_ **2022** , _25_ , 33. [CrossRef] [PubMed] 

51. Jeon, H.; Park, J.; Shin, S.; Seo, J. Stop Misusing t-SNE and UMAP for Visual Analytics. _arXiv_ **2025** , arXiv:2506.08725. [CrossRef] 

52. McInnes, L.; Healy, J.; Melville, J. Umap: Uniform manifold approximation and projection for dimension reduction. _arXiv_ **2018** , arXiv:1802.03426. 

53. Huang, H.; Wang, Y.; Rudin, C.; Browne, E.P. Towards a comprehensive evaluation of dimension reduction methods for transcriptomic data visualization. _Commun. Biol._ **2022** , _5_ , 719. [CrossRef] 

54. Wei, G.; Lan, C.; Zeng, W.; Zhang, Z.; Chen, Z. Toalign: Task-oriented alignment for unsupervised domain adaptation. _Adv. Neural Inf. Process. Syst._ **2021** , _34_ , 13834–13846. 

55. Langbein, S.H.; Koenen, N.; Wright, M.N. Gradient-based Explanations for Deep Learning Survival Models. _arXiv_ **2025** , arXiv:2502.04970. 

56. Santos, R.; Pedrosa, J.; Mendonça, A.M.; Campilho, A. Grad-CAM: The impact of large receptive fields and other caveats. _Comput. Vis. Image Underst._ **2025** , _258_ , 104383. [CrossRef] 

57. Singh, A.K.; Chaudhuri, D.; Singh, M.P.; Chattopadhyay, S. Integrative CAM: Adaptive Layer Fusion for Comprehensive Interpretation of CNNs. _arXiv_ **2024** , arXiv:2412.01354. [CrossRef] 

58. Ahmad, J.; Rehman, M.I.U.; ul Islam, M.S.; Rashid, A.; Khalid, M.Z.; Rashid, A. Layer-Wise Relevance Propagation in Large-Scale Neural Networks for Medical Diagnosis. _Res. Med. Sci. Rev._ **2025** , _3_ , 6–18. 

_Mathematics_ **2025** , _13_ , 2602 

28 of 29 

59. Ding, R.; Liu, J.; Hua, K.; Wang, X.; Zhang, X.; Shao, M.; Chen, Y.; Chen, J. Leveraging data mining, active learning, and domain adaptation for efficient discovery of advanced oxygen evolution electrocatalysts. _Sci. Adv._ **2025** , _11_ , eadr9038. [CrossRef] 

60. Murphy, K.P. _Probabilistic Machine Learning: An Introduction_ ; MIT Press: Cambridge, MA, USA, 2022. 

61. Scholkopf, B.; Smola, A.J. _Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond_ ; MIT Press: Cambridge, MA, USA, 2018. 

62. Wilson, A.; Adams, R. Gaussian process kernels for pattern discovery and extrapolation. In Proceedings of the International Conference on Machine Learning, Atlanta, GA, USA, 16–21 June 2013; PMLR: Cambridge, MA, USA, 2013; pp. 1067–1075. 

63. Principe, J.C. _Information Theoretic Learning: Renyi’s Entropy and Kernel Perspectives_ ; Springer Science & Business Media: New York, NY, USA, 2010. 

64. Bishop, C.M.; Nasrabadi, N.M. _Pattern Recognition and Machine Learning_ ; Springer: New York, NY, USA, 2006; Volume 4. 

65. Silverman, B.W. _Density Estimation for Statistics and Data Analysis_ ; Routledge: London, UK, 2018. 

66. Xu, J.W.; Paiva, A.R.; Park, I.; Principe, J.C. A reproducing kernel Hilbert space framework for information-theoretic learning. _IEEE Trans. Signal Process._ **2008** , _56_ , 5891–5902. [CrossRef] 

67. Bromiley, P. Products and convolutions of Gaussian probability density functions. _Tina-Vis. Memo_ **2003** , _3_ , 1. 

68. Giraldo, L.G.S.; Rao, M.; Principe, J.C. Measures of entropy from data using infinitely divisible kernels. _IEEE Trans. Inf. Theory_ **2014** , _61_ , 535–548. [CrossRef] 

69. Giraldo, L.G.S.; Principe, J.C. Information theoretic learning with infinitely divisible kernels. _arXiv_ **2013** , arXiv:1301.3551. [CrossRef] 

70. Hatefi, E.; Karshenas, H.; Adibi, P. Probabilistic similarity preservation for distribution discrepancy reduction in domain adaptation. _Eng. Appl. Artif. Intell._ **2025** , _158_ , 111426. [CrossRef] 

71. Sankaranarayanan, S.; Balaji, Y.; Castillo, C.D.; Chellappa, R. Generate to adapt: Aligning domains using generative adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 18–23 June 2018; pp. 8503–8512. 

72. Cheng, J.; Liu, L.; Liu, B.; Zhou, K.; Da, Q.; Yang, Y. Foreground object structure transfer for unsupervised domain adaptation. _Int. J. Intell. Syst._ **2022** , _37_ , 8968–8987. [CrossRef] 

73. Murez, Z.; Kolouri, S.; Kriegman, D.; Ramamoorthi, R.; Kim, K. Image to image translation for domain adaptation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 18–23 June 2018; pp. 4500–4509. 

74. He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Las Vegas, NV, USA, 27–30 June 2016; pp. 770–778. 

75. Odusami, M.; Maskeliunas, R.; Damaševiˇcius, R.; Krilaviˇcius, T.¯ Analysis of Features of Alzheimer’s Disease: Detection of Early Stage from Functional Brain Changes in Magnetic Resonance Images Using a Finetuned ResNet18 Network. _Diagnostics_ **2021** , _11_ , 1071. [CrossRef] [PubMed] 

76. Mascarenhas, S.; Agarwal, M. A comparison between VGG16, VGG19 and ResNet50 architecture frameworks for Image Classification. In Proceedings of the 2021 International Conference on Disruptive Technologies for Multi-Disciplinary Research and Applications (CENTCON), Bengaluru, India, 19–21 November 2021; IEEE: Piscataway, NJ, USA, 2021; Volume 1, pp. 96–99. 

77. Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv_ **2020** , arXiv:2010.11929. 

78. Jin, Y.; Song, X.; Yang, Y.; Hei, X.; Feng, N.; Yang, X. An improved multi-channel and multi-scale domain adversarial neural network for fault diagnosis of the rolling bearing. _Control Eng. Pract._ **2025** , _154_ , 106120. [CrossRef] 

79. Li, B.; Liu, H.; Ma, N.; Zhu, S. Cross working conditions manufacturing process monitoring using deep convolutional adversarial discriminative domain adaptation network. _Proc. Inst. Mech. Eng. Part B J. Eng. Manuf._ **2025** , 09544054251324677. [CrossRef] 

80. Deng, M.; Zhou, D.; Ao, J.; Xu, X.; Li, Z. Bearing fault diagnosis of variable working conditions based on conditional domain adversarial-joint maximum mean discrepancy. _Int. J. Adv. Manuf. Technol._ **2025** , 1–18. [CrossRef] 

81. Qiao, D.; Ma, X.; Fan, J. Federated t-sne and umap for distributed data visualization. In Proceedings of the AAAI Conference on Artificial Intelligence, Philadelphia, PA, USA, 25 February–4 March 2025; Volume 39, pp. 20014–20023. 

82. Raveenthini, M.; Lavanya, R.; Benitez, R. Grad-CAM based explanations for multiocular disease detection using Xception net. _Image Vis. Comput._ **2025** , _154_ , 105419. [CrossRef] 

83. Chung, Y.; Eu, P.; Lee, J.; Choi, K.; Nam, J.; Chon, B.S. KAD: No More FAD! An Effective and Efficient Evaluation Metric for Audio Generation. _arXiv_ **2025** , arXiv:2502.15602. [CrossRef] 

84. Saito, K.; Kim, D.; Teterwak, P.; Sclaroff, S.; Darrell, T.; Saenko, K. Tune it the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density. _arXiv_ **2021** , arXiv:2108.10860. [CrossRef] 

85. Deng, Z.; Zhou, K.; Yang, Y.; Xiang, T. Domain Attention Consistency for Multi-Source Domain Adaptation. In Proceedings of the International Conference on Computer Vision (ICCV), Montreal, BC, Canada, 11–17 October 2021. 

_Mathematics_ **2025** , _13_ , 2602 

29 of 29 

86. Belal, A.; Meethal, A.; Romero, F.P.; Pedersoli, M.; Granger, E. Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptation of Object Detectors. _arXiv_ **2024** , arXiv:2403.09918. 

87. Xu, Y.; Men, A.; Liu, Y.; Zhuang, X.; Chen, Q. Incorporating Pre-Training Data Matters in Unsupervised Domain Adaptation. _IEEE Trans. Pattern Anal. Mach. Intell._ **2025** , _47_ , 7930–7943. [CrossRef] [PubMed] 

**Disclaimer/Publisher’s Note:** The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content. 

