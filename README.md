# Team Members

Chaitanya Datta Maddukuri – A20568393
Vamshi Krishna Cheeti – A20582646
Ram Prakash Bollam – A20561314


# Installation and Running Instructions

# 1. Clone the Repository
If the project is hosted on GitHub or another platform, clone it using:

git clone https://github.com/your-repo/LassoHomotopy.git

# 2. Install Dependencies
Ensure Python is installed, then install all required dependencies using:

pip install -r requirements.txt

# 3. Running the Test Suite
To verify the implementation, run the test cases using:

pytest LassoHomotopy/tests/test_LassoHomotopy.py

# 4. Running with collinear Data

pytest LassoHomotopy/tests/test_LassoHomotopyCollinear.py

# Questions & Answers

1. What does the model you have implemented do and when should it be used?
The model performs LASSO regression using the Homotopy Method, which is an efficient approach for solving LASSO problems. It is used for:

Sparse feature selection → When the dataset has many irrelevant or redundant features.
Handling multi-collinearity → LASSO automatically selects one of the highly correlated features while shrinking others to zero.
Regularized regression → When preventing overfitting is crucial.
It should be used in high-dimensional datasets where feature selection is necessary, such as in genomics, finance, and machine learning pipelines.

2. How did you test your model to determine if it is working reasonably correctly?
The model is tested using the test_LassoHomotopy.py script, which verifies:

Correct data processing → Ensures that fit() correctly loads small_test.csv and processes the features and target values.
Prediction consistency → The test case asserts that the predict() function returns 0.5, ensuring basic functionality.
Cross-validation correctness → The model selects the best λ using 5-fold cross-validation, confirming generalization.
Additionally, multi-collinear data (collinear_data.csv) is used to analyze how the model selects relevant features.

3. What parameters have you exposed to users of your implementation in order to tune performance?
The implementation allows users to tune the following parameters:

λ values (lambda_values) → A predefined range of λ values is used for selection. Users can modify this range for finer control.
Tolerance (tol) → Defines the stopping criteria for small coefficient changes. A smaller value leads to more precise solutions but increases computation time.
Maximum iterations (max_iter) → Limits the number of steps in the optimization process. Increasing this allows more refined solutions but increases runtime.
Cross-validation folds (k) → Currently set to 5, but users can adjust it for better model selection.

4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
Yes, the model may struggle with:

Highly correlated features with nearly equal importance → LASSO may arbitrarily select one over the other, which can sometimes lead to instability.
Very small datasets → Cross-validation may not work effectively if the dataset is too small.
Weak signals in high noise environments → LASSO may shrink important coefficients too aggressively when the signal-to-noise ratio is low.
Possible workarounds:

Elastic Net Regularization → Combining LASSO with Ridge regression can help mitigate instability in correlated features.
Adaptive λ selection → Using Bayesian optimization or grid search could refine λ selection beyond the fixed log-space range.
More robust testing → Updating the test case to check for actual predictions rather than enforcing 0.5 as output.
Summary

This project implements LASSO regression using the Homotopy Method, efficiently selecting sparse features and handling multi-collinearity. It includes cross-validation for λ tuning and has been tested using predefined datasets. Users can modify various hyperparameters to improve performance.