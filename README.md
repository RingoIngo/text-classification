# Text Classification on SMS Dataset

The goal of this project was to perform automatic text categorization (TC) on a data set collected by SMSGuru.

- We evaluat some feature extraction methods in the context of TC. The methods used include
tokenization, stemming, spell correction, bag of words, term frequency inverse document frequency. The extracted features are evaluated with a multinomial
Naive Bayes classifier and cross-validation.
- We evaluate the performance of three classification algorithms. A challenge of this data set is the
imbalanced distribution of the class frequencies. The following two plots display the frequency distribution of the main/sub-categories.
<p align="center"">
  <img src="/images/plot1.png" width="400" />
  <img src="/images/plot2.png" width="400" /> 
</p>
                                          
We consider a **k-Nearest Neighbor** classifier (kNN: vanilla, kNN-B: adapted to imbalanced class distribtion), a **Support Vector Machine** with different pre-processing steps and **Linear Discriminant Analysis**. The next two plots show a) LDA projection of a test split of the SMSGuru data set onto a two-dimensional discriminative subspace b) Estimate of the generalization error for the different classification methods. As a performance measure a macro-averaged F1 score is used. Estimates were computed using a 3-fold nested cross-validation.
<p align="center">
  <img src="/images/lda_dim_2.png" width="400" />
  <img src="/images/gen_error.png" width="400" />
</p>

- In milestone three, we evaluated the use of combining different classifiers by averaging and multiplying.
Based on those results, we proposed a final prediction method and evaluated its usefulness for the SMSGuru
business.

