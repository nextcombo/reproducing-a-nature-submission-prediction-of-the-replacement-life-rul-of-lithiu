# Reproducing a Nature Submission! Prediction of the Replacement Life (RUL) of Lithium Batteries Based on Signal Decomposition and Machine Learning

1. Introduction
This issue is based on the experimental data of lithium-ion batteries provided by NASA (National Aeronautics and Space Administration) and reproduces the model experiment proposed in a Nature sub-journal paper. Through an in-depth analysis and experimental verification of this advanced deep learning model, we aim to provide a methodological path and data support for subsequent research on health management systems for lithium-ion batteries, offering significant learning and reference value.

2. Paper Title: "A hybrid approach for predicting the remaining useful life of lithium-ion batteries using signal decomposition and machine learning"

3. Paper Link:
https://www.nature.com/articles/s41598-025-92262-8

1. Reproduction Overview
1.1 Model Evaluation:
1.2 Prediction Visualization:
We also provide detailed materials and video explanations, including tutorials on parameter adjustments and replacements for prediction tasks, with code commented line by line and detailed introductions to parameters:
● Dataset: NASA Lithium Ion Battery Life Test Public Dataset
● Environment Framework: Python 3.11, PyTorch 2.1 or above are required to run
● Usage Object: Those who need the requirements of the paper or their graduation project
● Code Guarantee: The code is commented in detail and can be run successfully without modification.
Click the bottom of the article to read the original text and get the code!

2. Reproduction of the Paper Brief
2.1 Abstract
2.2 Introduction to the Paper Model

Lithium-ion batteries have widespread applications in many fields, and accurate prediction of their remaining useful life (RUL) is crucial for effective management and safety assurance of these batteries. To address the issue of decreased RUL prediction accuracy due to local capacity regeneration during battery capacity degradation, a new RUL prediction method has been proposed. This method integrates the full integration of empirical mode decomposition and adaptive noise (CEEMDAN) technology with a hybrid prediction strategy that combines Support Vector Regression (SVR) and Long Short-Term Memory (LSTM) networks.

Firstly, CEEMDAN is used to decompose the battery capacity data into high-frequency components and low-frequency components, thereby reducing the impact of capacity regeneration. Subsequently, a Support Vector Machine model is employed to predict the low-frequency components that represent the main degradation trends, while an LSTM network optimized by the Sparrow Search Algorithm (SSA) is applied to predict the high-frequency components that contain capacity regeneration characteristics. Finally, the predictions from both models are combined to obtain the final RUL prediction. Experimental results on the NASA public dataset demonstrate that the hybrid method achieves root mean square error less than 0.0086 Ah, average mean square error less than 0.0060 Ah, and an R2 value greater than 0.96, ensuring RUL prediction errors within a cycle. This method fully leverages the complementary advantages of Support Vector Machines and LSTMs, providing a precise and reliable solution for the RUL prediction of lithium-ion batteries.
2.3 Introduction to the Paper's Dataset

This paper proposes a novel method for predicting the remaining useful life (RUL) of lithium batteries using CEEMDAN, SVR, and LSTM. CEEMDAN is employed to decompose the capacity data into multiple Intrinsic Mode Functions (IMFs) and a residue term. Each IMF represents different frequency components in the original data, while the residue captures the long-term trend component of the data. The SVR is utilized to capture the low-frequency residual part of the data, while LSTM is used to capture the nonlinearity and long-term and short-term correlations of high-frequency components. To avoid the impact of manual adjustment of LSTM hyperparameters on prediction accuracy, the Supervised Search Algorithm (SSA) is employed for hyperparameter optimization. This method was experimentally validated on the public dataset from NASA Ames Predictive Excellence Center (PCOE). The results demonstrate that this method achieves higher prediction accuracy than other mainstream methods, validating its effectiveness in accurately predicting the RUL of batteries.
This article obtained the original dataset of lithium battery aging from NASA's Ames Research Center. Four batteries, labeled B0005, B0006, B0007, and B0018, were selected to validate the effectiveness of this method. All battery tests were conducted under a constant environmental temperature of 24°C. The specific charging and discharging schemes are as follows: during the charging phase, a constant current (CC) of 1.5A was initially applied until the battery voltage reached 4.2V, then switched to constant voltage (CV) mode until the charging current decreased to 20 mA, indicating the end of charging. The discharge process included a constant current (CC) of 2 A until the battery voltage dropped to each battery's specific cut-off voltage: B0005 at 2.7V, B0006 and B0018 at 2.5V, B0007 at 2.2V. Table 1 also lists other key parameters, including discharge current (DC), discharge voltage (DV), and battery failure threshold (TS). This study utilized the capacity data from the original dataset for experimental verification. The termination of battery life (EOL) is defined as when the battery capacity drops to 70% of its rated capacity (1.4ah). However, for battery B0007, since it did not reach 1.4ah, EOL was set at 72% of the rated capacity (1.45ah). As shown in the figure, the capacity degradation trend of NASA lithium batteries is displayed.
3 Data Preprocessing
3.1 CEEMDAN Decomposition
This paper applies the CEEMDAN algorithm directly to the decomposition of lithium-ion battery capacity data, resulting in multiple IMFs. Compared to the decomposition of lithium-ion battery current and voltage data, this method offers faster processing speed, higher accuracy, and lower computational complexity. The selection of mode number k significantly impacts the decomposition performance. As shown in the figure, the IMFs obtained by decomposing NASA lithium-ion battery capacity data using the CEEMDAN algorithm are displayed. In the figure, each IMF displays different characteristics of the lithium-ion battery data. High-frequency IMFs (IMF1 to IMF3) mainly capture short-term changes and rapid fluctuations, although similar to random noise, they can still provide insights into subtle variations in battery performance. IMF4 represents the main trend, reflecting the overall decline over time. This decomposition allows for clear analysis, distinguishing between instantaneous fluctuations, periodic characteristics, and the main degradation trends.
3.2 Parameter Settings, Optimization of the Peach Optimization Algorithm
The Sparrow Optimization Algorithm (SSA), also known as the Sparrow Search Algorithm, is an optimization algorithm based on the behavioral characteristics of sparrows in nature. It simulates the optimization strategies used by sparrows in activities such as foraging, migratory movements, and social interactions. This algorithm has demonstrated excellent performance in solving a variety of optimization problems.
\t  
This paper employs the SSA algorithm to optimize the main hyperparameters of the network structure. The process of optimizing the hyperparameters of the LSTM network using SSA is illustrated in the figure. The SSA parameters used are shown in Table 2, and the range of optimization for the LSTM hyperparameters is shown in Table 3.
\t  
4. Prediction Model Based on CEEMDAN+SSA-LSTM+SVR
4.1 Import Data
Taking the B0005 dataset as an example, perform data preprocessing:
4.2 Set parameters and train the model
4.3 Model Evaluation
(1) Model Metrics Evaluation
(2) Predictive Effect Display
SVR predicts the effect of IMF4 components:
\t  
The overall combination prediction effect:
Note: The original paper does not provide open source code, and the training parameters are not fully detailed. We strive to reproduce the effects of the paper as closely as possible. Any errors are normal!
The models and data processing workflows recreated in this article can serve as references and learning for beginners and those requiring papers!
Previously excellent content:
Time series prediction: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind speed prediction (I) - Data introduction and preprocessing
Wind speed prediction (II) - EMD-LSTM model based on Pytorch
Wind speed prediction (III) - EMD-LSTM-Attention model
Wind speed prediction (IV) - EMD-Transformer model based on Pytorch
Wind speed prediction (V) - EMD-CNN-LSTM model based on Pytorch
Wind speed prediction (VI) - EMD-CNN-GRU parallel model based on Pytorch
CEEMDAN + combined prediction model (BiLSTM-Attention + ARIMA)
CEEMDAN + combined prediction model (CNN-LSTM + ARIMA)
CEEMDAN + combined prediction model (Transformer - BiLSTM + ARIMA)
CEEMDAN + combined prediction model (CNN-Transformer + ARIMA)
Multi-feature variable sequence prediction (I) - CNN-LSTM wind speed prediction model
Multi-feature variable sequence prediction (II) - CNN-LSTM-Attention wind speed prediction model
Multi-feature variable sequence prediction (III) - CNN-Transformer wind speed prediction model
Multi-feature variable sequence prediction (IV) - Transformer-BiLSTM wind speed prediction model
Multi-feature variable sequence prediction (5) CEEMDAN+CNN-LSTM wind speed prediction model
Multi-feature variable sequence prediction (6) CEEMDAN+CNN-Transformer wind speed prediction model
Prediction model based on the SSA optimization algorithm for CEEMDAN-BiLSTM-Attention:  
Prediction model based on the SSA optimization algorithm for CEEMDAN-Transformer-BiGRU:  
Superior predictive model: Secondary decomposition-combined prediction
Multi-feature variable sequence prediction (7) CEEMDAN+Transformer-BiLSTM prediction model
Multi-feature variable sequence prediction (8) Prediction model based on the SSA optimization algorithm for CEEMDAN-SSA-BiLSTM:  
Multi-feature variable sequence prediction (11) Prediction model based on Pytorch's TCN-GRU:  
VMD + CEEMDAN secondary decomposition, CNN-LSTM prediction model
VMD + CEEMDAN secondary decomposition, CNN-Transformer prediction model
VMD + CEEMDAN secondary decomposition, Transformer-BiGRU prediction model
Prediction model based on the SSA optimization algorithm for - code family of models
 | Superior combined prediction model!
High innovation | CEEMDAN + SSA-TCN-BiLSTM-Attention prediction model
 | Superior combined prediction model!
Lowest price online | Family of models continuously updated!  
| Based on TCN-SENet + BiGRU-GlobalAttention parallel prediction model
Secondary decomposition - innovative model prediction family of models
 | BiTCN-BiGRU-CrossAttention integrated spatiotemporal features high innovation prediction model
Combined prediction update | Superior combined prediction model!
BiTCN-Transformer model for parallel prediction with fusion of spatiotemporal features
Parallel prediction model based on multi-level attention mechanism
| CEEMDAN-CNN-GRU-GlobalAttention + XGBoost Combination Prediction
Multi-step prediction series | LSTM, CNN, Transformer, TCN, serial, parallel model collection
END  
2. Friends who enjoy the content can follow, give the article a "like", share on social media or discuss in groups.

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
