# Multi-model Classification App

## a. Problem Statement

Bob has started his own mobile company and wants to compete with big companies like Apple and Samsung. He needs to estimate the price of mobiles his company creates but doesn't know how to do this effectively. In this competitive mobile phone market, assumptions are not enough.

To solve this problem, Bob collects sales data of mobile phones from various companies. He wants to find the relationship between features of a mobile phone (e.g., RAM, Internal Memory, etc.) and its selling price. However, he is not proficient in Machine Learning and needs help to solve this problem.

**Objective**: Instead of predicting the actual price, the goal is to predict a price range indicating how high the price is. This is a **multi-class classification problem** with 4 price range categories (0: low cost, 1: medium cost, 2: high cost, 3: very high cost).

## b. Dataset Description

**Dataset**: Mobile Price Classification Dataset  
**Source**: Kaggle (iabhishekofficial/mobile-price-classification)

### Dataset Characteristics:
- **Total Samples**: 2000
- **Training Set**: 1600 samples (80%)
- **Test Set**: 400 samples (20%)
- **Train/Test Split**: 0.8

### Target Variable:
- **price_range**: Mobile phone price category
  - 0: Low cost
  - 1: Medium cost
  - 2: High cost
  - 3: Very high cost

### Features (20 total):

| Feature | Description |
|---------|-------------|
| **battery_power** | Total energy a battery can store in one time measured in mAh |
| **blue** | Has Bluetooth or not |
| **clock_speed** | Speed at which microprocessor executes instructions |
| **dual_sim** | Has dual sim support or not |
| **fc** | Front Camera mega pixels |
| **four_g** | Has 4G or not |
| **int_memory** | Internal Memory in Gigabytes |
| **m_dep** | Mobile Depth in cm |
| **mobile_wt** | Weight of mobile phone |
| **n_cores** | Number of cores of processor |
| **pc** | Primary Camera mega pixels |
| **px_height** | Pixel Resolution Height |
| **px_width** | Pixel Resolution Width |
| **ram** | Random Access Memory in Mega Bytes |
| **sc_h** | Screen Height of mobile in cm |
| **sc_w** | Screen Width of mobile in cm |
| **talk_time** | Longest time that a single battery charge will last when you are talking |
| **three_g** | Has 3G or not |
| **touch_screen** | Has touch screen or not |
| **wifi** | Has wifi or not |

## c. Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9825 | 0.9992 | 0.9826 | 0.9825 | 0.9825 | 0.9767 |
| Decision Tree | 0.8675 | 0.9631 | 0.8673 | 0.8675 | 0.8671 | 0.8235 |
| kNN | 0.6500 | 0.8608 | 0.6496 | 0.6500 | 0.6497 | 0.5334 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest (Ensemble) | 0.8875 | 0.9817 | 0.8870 | 0.8875 | 0.8872 | 0.8500 |
| XGBoost (Ensemble) | 0.9400 | 0.9927 | 0.9399 | 0.9400 | 0.9397 | 0.9202 |

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | **Best overall performance** with 98.25% accuracy and highest AUC (0.9992). Exceptional balance across all metrics with near-perfect precision, recall, and F1 scores. Despite being a simple linear model, it achieves outstanding results on this dataset, indicating strong linear separability between price ranges. The very high MCC (0.9767) demonstrates extremely reliable predictions across all classes. This model is the clear winner for this classification task. |
| Decision Tree | Good performance with 86.75% accuracy and strong AUC (0.9631). The model shows improved generalization compared to basic configurations with MCC of 0.8235. While it captures non-linear patterns effectively, it still shows some overfitting tendencies. The decent performance across all metrics makes it a solid choice, though it's outperformed by ensemble methods and logistic regression. |
| kNN | **Moderate performance** with 65% accuracy and the weakest performer. The AUC of 0.8608 suggests reasonable ranking ability despite lower classification accuracy. The MCC of 0.5334 indicates moderate correlation between predictions and actual values. The distance-based approach struggles with the high-dimensional feature space (20 features), and even with optimized hyperparameters (K=7, distance weighting), it cannot match other algorithms. Feature scaling helps but isn't sufficient to overcome the curse of dimensionality. |
| Naive Bayes | Reasonable performance with 81% accuracy and strong AUC (0.9506). The high AUC relative to accuracy indicates excellent probability calibration despite moderate classification performance. Consistent performance across metrics (precision, recall, F1 all ~0.81) shows balanced predictions across all price range classes. |
| Random Forest (Ensemble) | Strong performance with 88.75% accuracy and excellent AUC (0.9817). The ensemble approach provides robust generalization with high MCC (0.8500), indicating reliable predictions. By combining multiple decision trees, it reduces overfitting and improves stability compared to a single decision tree. The balanced metrics across precision, recall, and F1 demonstrate consistent performance across all price categories. A solid choice offering good accuracy with interpretability through feature importance. |
| XGBoost (Ensemble) | **Second-best performance** with 94% accuracy and very high AUC (0.9927). Excellent MCC (0.9202) demonstrates strong predictive power across all classes. The gradient boosting approach effectively captures complex non-linear patterns while maintaining good generalization. Balanced metrics (precision, recall, F1 all ~0.94) show consistent performance. While slightly below Logistic Regression, it demonstrates the power of advanced ensemble methods and would be the top choice if the data had more complex non-linear relationships. |

