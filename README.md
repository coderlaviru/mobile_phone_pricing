#   Mobile Phone Price Prediction

##   Overview

    This project aims to predict the price range of mobile phones based on their specifications. The goal is to classify phones into price ranges: low, medium, high, and very high.

##   Dataset

    The dataset, `mobile_phone_pricing.csv`, contains various features of mobile phones and their corresponding price ranges.

    **Dataset Columns (from Predict Mobile Phone Pricing.pdf):**

    |   Feature       |   Description                     |
    |   :------------ |   :------------------------------ |
    |   battery_power |   Battery Capacity in mAh         |
    |   blue          |   Has Bluetooth or not            |
    |   clock_speed   |   Processor speed                 |
    |   dual_sim      |   Has dual sim support or not     |
    |   fc            |   Front camera megapixels         |
    |   four_g        |   Has 4G or not                   |
    |   int_memory    |   Internal Memory in GB           |
    |   m_dep         |   Mobile depth in cm              |
    |   mobile_wt     |   Weight in gm                    |
    |   n_cores       |   Processor Core Count            |
    |   pc            |   Primary Camera megapixels       |
    |   px_height     |   Pixel Resolution height         |
    |   px_width      |   Pixel Resolution width          |
    |   ram           |   Ram in MB                       |
    |   sc_h          |   Mobile Screen height in cm      |
    |   sc_w          |   Mobile Screen width in cm      |
    |   talk_time     |   Time a single charge will last |
    |   three_g       |   Has 3G or not                   |
    |   touch_screen  |   Has touch screen or not         |
    |   wifi          |   Has WiFi or not                |
    |   price_range   |   Price range (0-3)               |

    Where `price_range` is categorized as:

    * 0 = low cost
    * 1 = medium cost
    * 2 = high cost
    * 3 = very high cost

  **First 5 rows of the dataset:**

    ```
       battery_power  blue  clock_speed  dual_sim  fc  four_g  int_memory  m_dep  ...  px_height  px_width   ram  sc_h  sc_w  talk_time  three_g  touch_screen  wifi  price_range
    0            842     0          2.2         0   1       0           7    0.6  ...         20       756  2549     9     7         19        0             0     1            1
    1           1021     1          0.5         1   0       1          53    0.7  ...        905      1988  2631    17     3          7        1             1     0            2
    2            563     1          0.5         1   2       1          41    0.9  ...       1263      1716  2603    11     2          9        1             1     0            2
    3            615     1          2.5         0   0       0          10    0.8  ...       1216      1786  2769    16     8         11        1             0     0            2
    4           1821     1          1.2         0  13       1          44    0.6  ...       1208      1212  1411     8     2         15        1             1     0            1
    ```
    

  ##   Files

  * `mobile_phone_pricing.csv`: The dataset.
  * `mobile_phone_pricing.ipynb`: Jupyter Notebook with code.
  * `Predict Mobile Phone Pricing.pdf`: Project description.

  ##   Code and Analysis

  **Libraries Used (from mobile_phone_pricing.ipynb):**

    ```python
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    ```
  #   Feature Scaling
    ```
    from sklearn.preprocessing import LabelEncoder , StandardScaler
    ```

  #   Machine Learning Models
    ```
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    ```

  #   Model Evaluation
    ```
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    import warnings
    warnings.filterwarnings("ignore")
    ```

  **Data Preprocessing (from mobile_phone_pricing.ipynb):**

  * Scaling using StandardScaler
  * Converting categorical features to numerical format using LabelEncoder

    **Models Used (from mobile_phone_pricing.ipynb):**

    * Random Forest Classifier
    * XGBoost Classifier
    * Support Vector Classifier (SVC)

    **Model Evaluation (from mobile_phone_pricing.ipynb):**

    * Accuracy Score
    * Confusion Matrix
    * Classification Report

    **Example Model Comparison (from mobile_phone_pricing.ipynb):**

    ```
          Model  Accuracy
    0  Random Forest      0.88
    1     XGBoost      0.87
    2         SVM      0.96
    ```

    (Snippet of the model comparison results from the notebook)

    ##   Data Preprocessing 🛠️

    The data preprocessing steps included:

    * Scaling of the data using StandardScaler as seen in `mobile_phone_pricing.ipynb`.
    * Converting categorical features to numerical format using LabelEncoder

    ##   Exploratory Data Analysis (EDA) 🔍

    To understand the data:

    * Exploratory Data Analysis was performed using plots

    ##   Model Selection and Training 🧠

    * **Models**: Random Forest Classifier, XGBoost Classifier, and Support Vector Classifier (SVC) 🌳 (as stated in `mobile_phone_pricing.ipynb`).
    * The Support Vector Classifier model was chosen based on the accuracy

    ##   Model Evaluation ✅

    Model performance was evaluated using metrics such as:

    * Accuracy Score
    * Classification Report
    * Confusion Matrix

    ##   Results ✨

    The project aimed to build a model to predict mobile phone price ranges. The results of the model evaluation are detailed in the notebook (`mobile_phone_pricing.ipynb`).

    ##   Setup ⚙️

    1.  Clone the repository ⬇️.
    2.  Install dependencies:

        ```bash
        pip install pandas numpy seaborn matplotlib scikit-learn xgboost
        ```

    3.  Run the notebook:

        ```bash
        jupyter notebook mobile_phone_pricing.ipynb
        ```

    ##   Usage ▶️

    The `mobile_phone_pricing.ipynb` notebook can be used to:

    * Explore the dataset.
    * Preprocess the data.
    * Train the classification models.
    * Evaluate the models' performance.

    ##   Contributing 🤝

    Contributions to this project are welcome! If you have ideas for improvements or find any issues, please feel free to submit a pull request 🚀.

    ##   License 📄

    This project is open source and available under the MIT License.
