// Chapters and Questions Data
const chaptersData = {
  'MLT-01': {
    title: 'Machine Learning-I',
    description: 'Introduction to machine learning: supervised/unsupervised learning, decision trees, random forests, logistic regression, and feature engineering.',
    questions: [
      {
        id: 1,
        question: "What is the primary difference between supervised and unsupervised learning?",
        options: [
          "Supervised learning uses more data than unsupervised learning",
          "Supervised learning has labeled target variables, unsupervised learning does not",
          "Unsupervised learning is faster to train than supervised learning",
          "Supervised learning can only be used for classification problems"
        ],
        correct: 1,
        explanation: "Supervised learning uses labeled training data with clear features and target variables to build predictive models. The target variable 'supervises' the learning process. Unsupervised learning has no target supervision and tries to find patterns based only on features provided. The amount of data, training speed, and problem types can vary in both approaches.",
        difficulty: "Basic",
        concept: "Supervised vs Unsupervised Learning",
        hint: "Think about whether the algorithm has a 'teacher' showing it the correct answers during training."
      },
      {
        id: 2,
        question: "Which of the following is an example of a classification problem in trading?",
        options: [
          "Predicting the exact closing price of a stock tomorrow",
          "Predicting whether the market will go up or down in the next session",
          "Calculating the moving average of stock prices",
          "Determining the correlation between two assets"
        ],
        correct: 1,
        explanation: "Classification algorithms predict categorical/labeled outputs. Predicting whether the market goes up or down is a binary classification problem with discrete labels. Predicting exact price is regression (continuous output), while moving averages and correlations are calculations, not predictions.",
        difficulty: "Basic",
        concept: "Classification Problems",
        hint: "Classification means putting things into categories or groups, not predicting continuous numbers."
      },
      {
        id: 3,
        question: "What is transfer learning?",
        options: [
          "Training a model on one dataset and transferring it to a different computer",
          "Using a pre-trained model from one task with minimal retraining for a related new task",
          "Transferring data from one format to another before training",
          "Moving model weights between layers during training"
        ],
        correct: 1,
        explanation: "Transfer learning involves using a model that required significant effort to train on Task A and applying it to a new context (Task B) with minimal additional training. This works when low-level features from Task A are helpful for Task B, especially when you have more data for Task A than Task B.",
        difficulty: "Basic",
        concept: "Transfer Learning",
        hint: "Think about reusing knowledge learned from one problem to solve a related problem."
      },
      {
        id: 4,
        question: "In the context of machine learning, what is a 'feature'?",
        options: [
          "The final output prediction of the model",
          "The input information used to predict the target variable",
          "The algorithm used to train the model",
          "The accuracy metric of the model"
        ],
        correct: 1,
        explanation: "Features are the set of information/inputs required to predict the target variable. For example, to predict stock price direction, features might include price information (OHLC), technical indicators, and statistical parameters. The target variable is what we want to predict, not the features.",
        difficulty: "Basic",
        concept: "Features and Target Variables",
        hint: "Features are the 'ingredients' you use to make a prediction."
      },
      {
        id: 5,
        question: "What type of machine learning algorithm would you use to cluster stocks based on similar characteristics?",
        options: [
          "Supervised classification",
          "Supervised regression",
          "Unsupervised clustering",
          "Transfer learning"
        ],
        correct: 2,
        explanation: "Clustering is an unsupervised learning task that groups data points based on similarity without pre-defined labels. Since we're grouping stocks by characteristics without a target variable to predict, unsupervised clustering is appropriate. Supervised methods require labeled target data.",
        difficulty: "Basic",
        concept: "Unsupervised Learning",
        hint: "You want to find groups naturally in the data without being told what the groups should be."
      },
      {
        id: 6,
        question: "Which of the following is NOT a supervised learning task?",
        options: [
          "Predicting if a credit card transaction is fraudulent",
          "Grouping customers into segments based on buying behavior",
          "Predicting house prices based on square footage and location",
          "Classifying emails as spam or not spam"
        ],
        correct: 1,
        explanation: "Grouping customers into segments is clustering, an unsupervised learning task with no predefined target labels. The other three are supervised: fraud detection is classification (fraudulent/legitimate), house price prediction is regression, and spam classification is binary classification - all require labeled training data.",
        difficulty: "Basic",
        concept: "Supervised vs Unsupervised Learning",
        hint: "Look for the option where you're not trying to predict a specific known outcome."
      },
      {
        id: 7,
        question: "In decision trees, what are the yellow rectangles called in the tree diagram?",
        options: [
          "Leaves",
          "Roots",
          "Nodes",
          "Branches"
        ],
        correct: 2,
        explanation: "In decision tree diagrams, nodes (yellow rectangles) represent decision points where the data is split based on a feature condition. Leaves (blue rhombus shapes) are the terminal nodes containing the final classification. The root is the topmost node, and branches connect nodes.",
        difficulty: "Basic",
        concept: "Decision Trees",
        hint: "These are the decision points, not the final outcomes."
      },
      {
        id: 8,
        question: "What is the main purpose of the Gini index in decision trees?",
        options: [
          "To calculate the depth of the tree",
          "To measure impurity and help make precise splits at each level",
          "To count the number of features in the dataset",
          "To determine the training time required"
        ],
        correct: 1,
        explanation: "The Gini index (or Gini impurity) measures the impurity at each node and helps determine the best feature and threshold to split on at each level. Lower Gini values indicate purer nodes. The formula is Gini(t) = 1 - Σ(pj²), where pj is the proportion of samples in class j.",
        difficulty: "Basic",
        concept: "Gini Index",
        hint: "Think about measuring how 'mixed' or 'pure' the data is at each split point."
      },
      {
        id: 9,
        question: "Can decision trees be used for both classification and regression problems?",
        options: [
          "Yes, they can handle both types of problems",
          "No, only classification problems",
          "No, only regression problems",
          "Only if the dataset is very small"
        ],
        correct: 0,
        explanation: "Decision trees are versatile supervised learning algorithms that can be used for both classification (predicting categorical outcomes like yes/no) and regression (predicting continuous values like house prices). The tree structure adapts based on the type of target variable.",
        difficulty: "Basic",
        concept: "Decision Trees",
        hint: "Decision trees are quite flexible - they can predict categories or numbers."
      },
      {
        id: 10,
        question: "What is the purpose of splitting a dataset into training and testing sets?",
        options: [
          "To increase the total amount of data available",
          "To evaluate how well the model performs on unseen data and detect overfitting",
          "To reduce the computational time required for training",
          "To balance the number of features and samples"
        ],
        correct: 1,
        explanation: "Splitting data into train and test sets allows you to fit the model on training data and then evaluate its performance on unseen test data. This helps detect overfitting - if the model performs well on training but poorly on testing data, it has overfit and won't generalize well to new data.",
        difficulty: "Basic",
        concept: "Train-Test Split",
        hint: "You need to check if your model works on data it hasn't seen before."
      },
      {
        id: 11,
        question: "In logistic regression, what range of values does the probability output fall between?",
        options: [
          "-1 and 1",
          "0 and 100",
          "0 and 1",
          "Any real number"
        ],
        correct: 2,
        explanation: "Logistic regression returns a probability between 0 and 1 using the sigmoid function: Probability = 1/(1+e^-z). This probability can then be converted to a classification by applying a threshold (commonly 0.5). Values cannot be negative or greater than 1 since they represent probabilities.",
        difficulty: "Basic",
        concept: "Logistic Regression",
        hint: "Logistic regression outputs probabilities, and all probabilities must fall within a specific range."
      },
      {
        id: 12,
        question: "What does the 'target variable' represent in supervised learning?",
        options: [
          "The input features used for prediction",
          "The field/label that the user wants to predict",
          "The algorithm used for training",
          "The size of the training dataset"
        ],
        correct: 1,
        explanation: "The target variable is the field/label in a dataset that you want to predict. For example, if you want to know the direction of a stock, the next day's close price would be your target. If building a portfolio, stock weights would be the target. Features are the inputs used to predict the target.",
        difficulty: "Basic",
        concept: "Features and Target Variables",
        hint: "This is what you're trying to predict or forecast."
      },
      {
        id: 13,
        question: "What is the main advantage of random forests over single decision trees?",
        options: [
          "Random forests are faster to train",
          "Random forests use less memory",
          "Random forests reduce overfitting and improve accuracy through ensemble voting",
          "Random forests require fewer features"
        ],
        correct: 2,
        explanation: "Random forests create multiple decision trees and use voting rules (like majority voting) to make predictions. This randomness in feature selection and tree creation helps the model generalize better and reduces the overfitting problem common in single decision trees, often improving overall accuracy.",
        difficulty: "Basic",
        concept: "Random Forests",
        hint: "Think about the wisdom of crowds - many trees voting together."
      },
      {
        id: 14,
        question: "Which statement best describes 'forecasting' versus 'prediction'?",
        options: [
          "They are exactly the same thing",
          "Forecasting predicts future variables, while prediction is broader and includes non-temporal predictions",
          "Prediction is only for classification, forecasting is only for regression",
          "Forecasting uses unsupervised learning, prediction uses supervised learning"
        ],
        correct: 1,
        explanation: "Forecasting is specifically about predicting future values (a subtopic of prediction). Prediction is broader and includes predicting things not related to future variables - for example, predicting whether an image contains a dog or cat. Both can use various ML algorithms.",
        difficulty: "Basic",
        concept: "Forecasting vs Prediction",
        hint: "One is about the future specifically, the other is more general."
      },
      {
        id: 15,
        question: "What is feature extraction?",
        options: [
          "Removing features from the dataset",
          "Creating new features from existing ones to capture more information",
          "Selecting only the most important features",
          "Normalizing feature values to the same scale"
        ],
        correct: 1,
        explanation: "Feature extraction is the process of creating or deriving new features from existing ones to capture additional patterns. Examples include creating rolling standard deviations, moving averages, or percentage changes from price data. This differs from feature selection (choosing existing features) and feature scaling (normalizing values).",
        difficulty: "Basic",
        concept: "Feature Engineering",
        hint: "Think about generating new information from what you already have."
      },
      {
        id: 16,
        question: "What is Occam's Razor principle in machine learning model selection?",
        options: [
          "Always choose the model with the highest accuracy regardless of complexity",
          "Between models with similar performance, prefer the simpler model",
          "Complex models always perform better than simple models",
          "The model with the most features is always the best"
        ],
        correct: 1,
        explanation: "Occam's Razor states that between two or more models with similar generalization errors, you should prefer the simpler model. For example, if two decision trees have similar accuracy but one has depth 3 and another has depth 5, choose the depth 3 model as it's simpler and less likely to overfit.",
        difficulty: "Basic",
        concept: "Model Selection",
        hint: "Simpler is better when performance is equal."
      },
      {
        id: 17,
        question: "In a decision tree for predicting fitness with Age < 27 as the root node, what happens when Age ≥ 27?",
        options: [
          "The person is automatically classified as fit",
          "The algorithm moves to the next decision node to evaluate other features",
          "The tree stops and returns no prediction",
          "The age feature is recalculated"
        ],
        correct: 1,
        explanation: "When Age ≥ 27 (the 'No' branch), the tree moves to the next decision node to evaluate other features like Exercise Duration or Calorie Intake. Decision trees work by cascading through multiple decision nodes until reaching a leaf node with the final classification.",
        difficulty: "Intermediate",
        concept: "Decision Trees",
        hint: "Decision trees don't stop at the first question - they keep asking until they reach a conclusion."
      },
      {
        id: 18,
        question: "Given a dataset with 30 samples where 20 are 'Fit' and 10 are 'Unfit', what is the Gini index at the root node?",
        options: [
          "0.3333",
          "0.4422",
          "0.6667",
          "0.5000"
        ],
        correct: 1,
        explanation: "Using the Gini formula: Gini(t) = 1 - Σ(pj²). Here, p1 = 20/30 and p2 = 10/30. So Gini = 1 - (20/30)² - (10/30)² = 1 - (0.4444) - (0.1111) = 1 - 0.5555 = 0.4444 ≈ 0.4422. This measures the impurity of the node before any split.",
        difficulty: "Intermediate",
        concept: "Gini Index",
        hint: "Calculate 1 minus the sum of squared proportions for each class."
      },
      {
        id: 19,
        question: "After splitting on Age < 27, one branch has 6 Fit and 1 Unfit. What is the Gini index for this branch?",
        options: [
          "0.1429",
          "0.2449",
          "0.3571",
          "0.4764"
        ],
        correct: 1,
        explanation: "With 6 Fit and 1 Unfit (7 total), p1 = 6/7 and p2 = 1/7. Gini = 1 - (6/7)² - (1/7)² = 1 - (0.7347) - (0.0204) = 1 - 0.7551 = 0.2449. This lower Gini value (compared to the root) indicates this is a purer node with less mixing of classes.",
        difficulty: "Intermediate",
        concept: "Gini Index",
        hint: "With 7 total samples where 6 are one class and 1 is another, the node should be fairly pure."
      },
      {
        id: 20,
        question: "If a split results in weighted Gini = 0.4223 compared to parent Gini = 0.4422, what does this indicate?",
        options: [
          "The split made the tree worse",
          "The split reduced impurity and is a good split",
          "The split has no effect on the tree",
          "The tree is overfitting"
        ],
        correct: 1,
        explanation: "The weighted Gini decreased from 0.4422 to 0.4223, meaning impurity was reduced. This is desirable as it indicates the split created more homogeneous child nodes. Decision trees aim to minimize impurity at each split, so a decrease in Gini indicates a successful split.",
        difficulty: "Intermediate",
        concept: "Gini Index",
        hint: "Lower Gini means purer nodes, which is what we want."
      },
      {
        id: 21,
        question: "In building a decision tree, which feature would you choose as the root node given these Gini impurity values: Age (0.44), Income (0.37), Credit Rating (0.50), Marital Status (0.46)?",
        options: [
          "Age because it has the highest Gini",
          "Income because it has the lowest Gini",
          "Credit Rating because it has the highest Gini",
          "Marital Status because it's in the middle"
        ],
        correct: 1,
        explanation: "You should choose Income with Gini = 0.37 as the root node because it has the lowest Gini impurity. Lower Gini indicates better separation of classes, meaning Income creates the purest split. We want to minimize impurity, so we select the feature with the minimum Gini value.",
        difficulty: "Intermediate",
        concept: "Gini Index",
        hint: "We want the split that creates the purest child nodes - which Gini value indicates this?"
      },
      {
        id: 22,
        question: "What is the primary issue that random forests address in decision trees?",
        options: [
          "Slow training time",
          "High memory usage",
          "Tendency to overfit the training data",
          "Inability to handle categorical variables"
        ],
        correct: 2,
        explanation: "Decision trees tend to overfit training data by creating overly complex trees that memorize training patterns. Random forests address this by creating multiple trees with random sampling (bootstrap) and random feature selection, then voting on predictions. This ensemble approach leads to better generalization.",
        difficulty: "Intermediate",
        concept: "Random Forests",
        hint: "Think about what happens when a single tree becomes too specialized to the training data."
      },
      {
        id: 23,
        question: "In a random forest with majority voting, if 7 out of 10 trees predict a person is 'Unfit', what is the final prediction?",
        options: [
          "Fit, because we need 80% agreement",
          "Unfit, because more than 50% of trees agree",
          "No prediction, because not all trees agree",
          "The average of all predictions"
        ],
        correct: 1,
        explanation: "With majority voting (the most common voting rule), if more than 50% of trees make the same prediction, that becomes the final prediction. Here, 7/10 = 70% predict Unfit, which exceeds the 50% threshold, so the random forest predicts Unfit.",
        difficulty: "Intermediate",
        concept: "Random Forests",
        hint: "Majority voting means more than half the trees need to agree."
      },
      {
        id: 24,
        question: "Why does random forest select features randomly during tree building?",
        options: [
          "To reduce computational time",
          "To reduce memory requirements",
          "To reduce correlation between trees and improve generalization",
          "To simplify the algorithm implementation"
        ],
        correct: 2,
        explanation: "Random forests randomly select a subset of features at each node to create decorrelated trees. If all trees used all features, they might make similar mistakes. Random feature selection ensures diversity among trees, which when combined through voting, leads to better generalization and reduced overfitting.",
        difficulty: "Intermediate",
        concept: "Random Forests",
        hint: "Diversity is strength - we want different trees to look at the problem differently."
      },
      {
        id: 25,
        question: "What is the purpose of the sigmoid function in logistic regression?",
        options: [
          "To calculate the Gini index",
          "To transform linear combinations into probabilities between 0 and 1",
          "To split decision tree nodes",
          "To normalize input features"
        ],
        correct: 1,
        explanation: "The sigmoid function (1/(1+e^-z)) transforms any real-valued number z into a probability between 0 and 1. This S-shaped curve is essential for logistic regression as it converts the linear equation z = β₀ + β₁X₁ + β₂X₂ + ... into a probability that can be used for classification.",
        difficulty: "Intermediate",
        concept: "Logistic Regression",
        hint: "Logistic regression needs to output probabilities, not unbounded numbers."
      },
      {
        id: 26,
        question: "In logistic regression with a 50% threshold, a student has a predicted probability of 0.62 to pass an exam. What is the classification?",
        options: [
          "Fail (0) because 0.62 is less than 1",
          "Pass (1) because 0.62 exceeds the 0.5 threshold",
          "Uncertain, need more data",
          "The probability needs to be recalculated"
        ],
        correct: 1,
        explanation: "With a 50% threshold, probabilities ≥ 0.5 are classified as Pass (1) and probabilities < 0.5 as Fail (0). Since 0.62 > 0.5, the student is classified as Pass. The threshold can be adjusted based on the application's requirements.",
        difficulty: "Intermediate",
        concept: "Logistic Regression",
        hint: "Compare the probability to the threshold to determine the class."
      },
      {
        id: 27,
        question: "What key assumption does logistic regression make that random forests do not require?",
        options: [
          "The data must be normally distributed",
          "There is a specific probability function (sigmoid) relating features to outcome",
          "All features must be numerical",
          "The dataset must be balanced"
        ],
        correct: 1,
        explanation: "Logistic regression assumes a specific mathematical form - the sigmoid probability function linking features to outcomes through a linear equation. Random forests and decision trees make no such assumptions about the underlying probability distribution, making them more flexible but potentially requiring more data.",
        difficulty: "Intermediate",
        concept: "Logistic Regression",
        hint: "One method assumes a particular mathematical relationship, the other doesn't."
      },
      {
        id: 28,
        question: "In feature engineering for a stock prediction model, why would you create a rolling standard deviation feature?",
        options: [
          "To increase the dataset size",
          "To capture volatility patterns over time",
          "To remove outliers from the data",
          "To normalize the price values"
        ],
        correct: 1,
        explanation: "Rolling standard deviation captures the volatility or variability of prices over a specific time window. This extracted feature can reveal patterns in price stability or turbulence that simple price data might not show, helping the model learn relationships between volatility and future price movements.",
        difficulty: "Intermediate",
        concept: "Feature Engineering",
        hint: "Standard deviation measures spread or variability - what does this tell us about price movements?"
      },
      {
        id: 29,
        question: "Why is StandardScaler used before training many machine learning models?",
        options: [
          "To remove missing values from the dataset",
          "To ensure features have similar scales (mean=0, std=1) for better model convergence",
          "To increase the number of features",
          "To split the data into train and test sets"
        ],
        correct: 1,
        explanation: "StandardScaler normalizes features to have mean=0 and standard deviation=1. This is important because features on different scales (e.g., price in hundreds vs. volume in millions) can cause models to weight larger-scale features incorrectly. Scaling improves convergence and performance for algorithms like logistic regression.",
        difficulty: "Intermediate",
        concept: "Feature Scaling",
        hint: "Different features might have wildly different ranges - what problem does this cause?"
      },
      {
        id: 30,
        question: "When applying StandardScaler, should you fit_transform on training data and transform on test data, or fit_transform on both?",
        options: [
          "fit_transform on both train and test data",
          "fit_transform on training data, transform on test data",
          "transform on both without fitting",
          "fit on test data first, then transform training data"
        ],
        correct: 1,
        explanation: "You should fit_transform on training data (learning the mean and std from training set) and only transform test data using those learned parameters. This prevents data leakage - the test set should not influence the scaling parameters, as it represents truly unseen future data.",
        difficulty: "Intermediate",
        concept: "Feature Scaling",
        hint: "The test set should be treated as completely unseen data - it shouldn't influence any parameters."
      },
      {
        id: 31,
        question: "What does pre-pruning do to address overfitting in decision trees?",
        options: [
          "Removes nodes after the tree is fully grown",
          "Stops tree growth when it reaches certain criteria (e.g., maximum depth)",
          "Randomly removes features during training",
          "Increases the minimum samples required at each node after training"
        ],
        correct: 1,
        explanation: "Pre-pruning stops the tree-building algorithm early based on criteria like maximum depth, minimum samples per node, or minimum impurity decrease. This prevents the tree from becoming too complex during initial construction. Post-pruning, in contrast, builds a full tree and then trims nodes in a bottom-up manner.",
        difficulty: "Intermediate",
        concept: "Overfitting Prevention",
        hint: "Pre- means before - so this happens during tree construction, not after."
      },
      {
        id: 32,
        question: "In the logistic regression code, why is shuffle=False used in train_test_split for time-series data?",
        options: [
          "To make the code run faster",
          "To prevent lookahead bias by maintaining temporal order",
          "To ensure equal class distribution",
          "To reduce memory usage"
        ],
        correct: 1,
        explanation: "For time-series data, shuffle=False maintains chronological order, ensuring training data comes before test data. If we shuffled, future data could leak into training, creating lookahead bias where the model learns from future information that wouldn't be available in real trading. This would give unrealistic performance.",
        difficulty: "Intermediate",
        concept: "Train-Test Split",
        hint: "In time-series, can you use tomorrow's data to predict yesterday?"
      },
      {
        id: 33,
        question: "What is the purpose of creating a target variable with np.where(data.returns.shift(-1) > 0, 1, 0)?",
        options: [
          "To predict yesterday's return",
          "To create binary labels for whether tomorrow's return is positive (1) or negative (0)",
          "To calculate today's return",
          "To normalize the returns data"
        ],
        correct: 1,
        explanation: "The shift(-1) moves returns forward by one period, so we're looking at tomorrow's return. np.where checks if tomorrow's return > 0, assigning 1 for positive (up day) and 0 for negative (down day). This creates the binary classification target for predicting next-day market direction.",
        difficulty: "Intermediate",
        concept: "Target Variable Creation",
        hint: "The shift(-1) is key - it's looking ahead one period."
      },
      {
        id: 34,
        question: "Why would a model show 95% accuracy on training data but only 52% on test data?",
        options: [
          "The model is working perfectly",
          "The model has severely overfit the training data and fails to generalize",
          "The test data is corrupted",
          "The model needs more features"
        ],
        correct: 1,
        explanation: "This is a classic sign of overfitting. The model has memorized the training data (95% accuracy) but learned patterns that don't generalize to new data (52% test accuracy, barely better than random guessing for binary classification). The model is too complex for the problem or training data patterns don't represent test data.",
        difficulty: "Intermediate",
        concept: "Overfitting",
        hint: "When training and test performance diverge dramatically, what has gone wrong?"
      },
      {
        id: 35,
        question: "In a confusion matrix for binary classification, what does the value in position [0,1] represent?",
        options: [
          "True Negatives - correctly predicted negative class",
          "False Positives - incorrectly predicted as positive",
          "True Positives - correctly predicted positive class",
          "False Negatives - incorrectly predicted as negative"
        ],
        correct: 1,
        explanation: "In a confusion matrix layout, rows represent actual classes and columns represent predictions. Position [0,1] means actual class 0 but predicted class 1 - these are False Positives. For trading, if 0=Short and 1=Long, this represents cases where we predicted Long but should have predicted Short.",
        difficulty: "Intermediate",
        concept: "Confusion Matrix",
        hint: "Row shows what it actually was, column shows what we predicted."
      },
      {
        id: 36,
        question: "What does entropy measure in decision tree algorithms like ID3?",
        options: [
          "The computational complexity of the tree",
          "The impurity or disorder in the data at a node",
          "The depth of the decision tree",
          "The number of features used"
        ],
        correct: 1,
        explanation: "Entropy measures impurity or disorder at a node, similar to Gini index but using a different formula: Entropy(t) = -Σ(pj * log2(pj)). Information gain is the difference between entropy before the split and average entropy after the split. Lower entropy indicates purer nodes with less mixing of classes.",
        difficulty: "Intermediate",
        concept: "Entropy",
        hint: "Entropy is a concept from information theory about disorder and uncertainty."
      },
      {
        id: 37,
        question: "Why might you prefer a decision tree with depth 3 over one with depth 7 if both have similar accuracy?",
        options: [
          "Depth 3 trees are always more accurate",
          "Depth 3 is simpler, easier to interpret, and less likely to overfit (Occam's Razor)",
          "Depth 7 trees cannot make predictions",
          "Depth 3 uses more features"
        ],
        correct: 1,
        explanation: "Following Occam's Razor principle, when models have similar performance, choose the simpler one. A depth 3 tree is much simpler, more interpretable, trains faster, and is less likely to overfit than a depth 7 tree. Simpler models generalize better to new data when performance is comparable.",
        difficulty: "Intermediate",
        concept: "Model Selection",
        hint: "Remember the principle: simpler is better when performance is equal."
      }
    ]
  },
  'MLT-02': {
    title: 'Machine Learning II: SVM, Clustering & Neural Networks',
    description: 'Advanced ML techniques including Support Vector Machines, K-Means Clustering, model evaluation metrics, and neural network fundamentals with backpropagation.',
    questions: [
      {
        id: 1,
        question: "What is the primary objective of a Support Vector Machine (SVM)?",
        options: [
          "To find the maximum margin hyperplane that separates classes",
          "To minimize the number of support vectors",
          "To maximize the number of misclassifications",
          "To find the minimum distance between data points"
        ],
        correct: 0,
        explanation: "SVM aims to find the hyperplane that maximally separates different classes by maximizing the margin (distance) between the hyperplane and the nearest data points from each class. This maximum margin approach helps with better generalization. The other options are incorrect: we don't minimize support vectors, we never want to maximize misclassifications, and we maximize (not minimize) distances in the context of margins.",
        difficulty: "Basic",
        concept: "Support Vector Machines",
        hint: "Think about what 'maximum margin' means in the context of separation."
      },
      {
        id: 2,
        question: "What is a supporting hyperplane?",
        options: [
          "A hyperplane that divides the feature space into equal parts",
          "A hyperplane where a set is entirely in one half-space with at least one boundary point on it",
          "The hyperplane with the smallest possible dimension",
          "Any hyperplane that touches at least two data points"
        ],
        correct: 1,
        explanation: "A supporting hyperplane has two key properties: (1) the set S is entirely contained in one of the half-spaces, and (2) S has at least one boundary point on the hyperplane. This definition ensures the hyperplane supports the set from one side. The other options don't capture these essential properties of supporting hyperplanes.",
        difficulty: "Intermediate",
        concept: "Hyperplanes",
        hint: "Consider what it means for a plane to 'support' a set from one side."
      },
      {
        id: 3,
        question: "In SVM, what are 'support vectors'?",
        options: [
          "All data points in the training set",
          "The data points that lie exactly on the decision boundary",
          "The data points closest to the hyperplane that define the margin",
          "The eigenvectors of the covariance matrix"
        ],
        correct: 2,
        explanation: "Support vectors are the data points that lie on the margin boundaries (the supporting hyperplanes) and are closest to the decision hyperplane. These points are critical because they determine the position and orientation of the hyperplane. If we removed other points, the hyperplane wouldn't change, but removing support vectors would alter the solution.",
        difficulty: "Basic",
        concept: "Support Vector Machines",
        hint: "These points 'support' the margins - they're the closest to the boundary."
      },
      {
        id: 4,
        question: "What is the distance between the two supporting hyperplanes H1 and H2 in SVM, where the hyperplane equation is w·x + b = 0?",
        options: [
          "1/||w||",
          "2/||w||",
          "||w||",
          "1/(2||w||)"
        ],
        correct: 1,
        explanation: "The margin width (distance between H1 and H2) is 2/||w||. The distance from the decision hyperplane to each supporting hyperplane is 1/||w||, so the total margin is twice this value. Maximizing the margin means minimizing ||w||, which is why SVM optimization seeks to minimize ½w^T w subject to classification constraints.",
        difficulty: "Advanced",
        concept: "SVM Margins",
        hint: "The distance to one margin is 1/||w||. How many margins are there?"
      },
      {
        id: 5,
        question: "Which kernel function would you use for data that is NOT linearly separable?",
        options: [
          "Linear kernel: K(w,x) = w·x",
          "No kernel is needed",
          "Polynomial or Gaussian kernel",
          "Identity kernel"
        ],
        correct: 2,
        explanation: "When data is not linearly separable in the original feature space, we use non-linear kernels like polynomial K(w,x) = (w·x + 1)^p or Gaussian (RBF) K(w,x) = exp(-||w-x||²/2σ²) to implicitly map data to a higher-dimensional space where it becomes linearly separable. Linear kernels only work for linearly separable data.",
        difficulty: "Intermediate",
        concept: "SVM Kernels",
        hint: "When a straight line won't work, you need to transform the space."
      },
      {
        id: 6,
        question: "What is the difference between hard margin and soft margin SVM?",
        options: [
          "Hard margin allows misclassifications, soft margin does not",
          "Hard margin requires perfect separation, soft margin allows some errors",
          "Hard margin uses non-linear kernels, soft margin uses linear kernels",
          "There is no difference, they are the same"
        ],
        correct: 1,
        explanation: "Hard margin SVM requires perfect classification with no errors, placing the hyperplane so all points are correctly classified. Soft margin SVM allows some misclassification errors (controlled by parameter C) to avoid overfitting. Soft margin is more practical for real-world noisy data. The formulation changes from minimizing ½w^T w to minimizing ½w^T w + C∑e_i where e_i are error terms.",
        difficulty: "Intermediate",
        concept: "SVM Margins",
        hint: "Which one is more flexible and tolerates some mistakes?"
      },
      {
        id: 7,
        question: "Why might a hard margin SVM lead to overfitting?",
        options: [
          "It uses too few features",
          "It forces perfect classification of training data, potentially fitting noise",
          "It always uses polynomial kernels",
          "It has too few support vectors"
        ],
        correct: 1,
        explanation: "Hard margin SVM insists on zero training errors, which means it may fit noise and outliers in the training data. This creates a complex decision boundary that doesn't generalize well to new data. Soft margin SVMs (with appropriate C values) generalize better by accepting some training errors and focusing on the overall pattern rather than perfectly fitting every point.",
        difficulty: "Intermediate",
        concept: "Overfitting in SVM",
        hint: "What happens when you try to perfectly fit every single training point?"
      },
      {
        id: 8,
        question: "In the SVM formulation, what does increasing the parameter C do in soft margin optimization?",
        options: [
          "Makes the margin wider",
          "Makes the margin harder (less tolerant of errors)",
          "Decreases the number of support vectors",
          "Changes the kernel function"
        ],
        correct: 1,
        explanation: "The parameter C controls the trade-off between maximizing the margin and minimizing classification errors. Larger C values penalize errors more heavily, leading to a harder margin that tries to classify more training points correctly (potentially overfitting). Smaller C allows more errors, creating a softer margin that may generalize better. C doesn't change the kernel function itself.",
        difficulty: "Advanced",
        concept: "SVM Parameters",
        hint: "C controls the penalty for misclassification errors."
      },
      {
        id: 9,
        question: "What type of learning is clustering?",
        options: [
          "Supervised learning",
          "Unsupervised learning",
          "Reinforcement learning",
          "Semi-supervised learning"
        ],
        correct: 1,
        explanation: "Clustering is unsupervised learning because it groups data based on similarity without using labeled target variables. The algorithm discovers patterns and structures in the data on its own. Supervised learning uses labeled data, reinforcement learning uses rewards/penalties, and semi-supervised uses some labeled and some unlabeled data.",
        difficulty: "Basic",
        concept: "Clustering",
        hint: "Does clustering require labeled target data?"
      },
      {
        id: 10,
        question: "Which similarity measure is defined as the sum of absolute differences across all dimensions?",
        options: [
          "Euclidean distance",
          "Manhattan distance",
          "Mahalanobis distance",
          "Cosine similarity"
        ],
        correct: 1,
        explanation: "Manhattan distance (also called L1 distance or taxicab distance) is calculated as the sum of absolute differences: ∑|x_i - y_i|. Euclidean distance uses squared differences under a square root. Mahalanobis distance is a weighted measure accounting for correlations. Cosine similarity measures the angle between vectors, not distance.",
        difficulty: "Basic",
        concept: "Similarity Measures",
        hint: "Think of walking along city blocks rather than cutting diagonally."
      },
      {
        id: 11,
        question: "What is the Euclidean distance between two points (x₁, y₁) and (x₂, y₂)?",
        options: [
          "|x₁ - x₂| + |y₁ - y₂|",
          "√[(x₁ - x₂)² + (y₁ - y₂)²]",
          "(x₁ - x₂)² + (y₁ - y₂)²",
          "max(|x₁ - x₂|, |y₁ - y₂|)"
        ],
        correct: 1,
        explanation: "Euclidean distance is the straight-line distance calculated using the Pythagorean theorem: √[(x₁ - x₂)² + (y₁ - y₂)²]. Option A is Manhattan distance, option C is squared Euclidean distance (missing the square root), and option D is Chebyshev distance. Euclidean distance is the most common distance metric in clustering.",
        difficulty: "Basic",
        concept: "Similarity Measures",
        hint: "Think of the Pythagorean theorem for the straight-line distance."
      },
      {
        id: 12,
        question: "In hierarchical clustering, what is agglomerative clustering?",
        options: [
          "Starting with one cluster and dividing it recursively",
          "Starting with each point as its own cluster and merging similar ones",
          "Randomly assigning points to clusters",
          "Using a fixed number of clusters from the start"
        ],
        correct: 1,
        explanation: "Agglomerative (or bottom-up) clustering starts with each data point as its own cluster and iteratively merges the closest pairs of clusters until a stopping criterion is met. Option A describes divisive (top-down) clustering. Options C and D don't describe hierarchical approaches. Agglomerative is more common than divisive clustering in practice.",
        difficulty: "Intermediate",
        concept: "Clustering Approaches",
        hint: "Think 'agglomerate' means to gather or collect together."
      },
      {
        id: 13,
        question: "What is the first step in the K-Means clustering algorithm?",
        options: [
          "Calculate distances between all pairs of points",
          "Randomly initialize K cluster centroids",
          "Assign all points to the same cluster",
          "Determine the optimal number of clusters"
        ],
        correct: 1,
        explanation: "K-Means begins by randomly placing K centroids in the feature space. Then it alternates between (1) assigning each point to its nearest centroid and (2) updating centroids to the mean of assigned points. This process repeats until convergence. The algorithm doesn't determine K automatically; you must specify it beforehand.",
        difficulty: "Basic",
        concept: "K-Means Clustering",
        hint: "You need starting positions for the cluster centers before anything else."
      },
      {
        id: 14,
        question: "In K-Means clustering, how are centroids updated after assigning points to clusters?",
        options: [
          "Centroids remain fixed throughout the algorithm",
          "Centroids are moved to random new positions",
          "Centroids are updated to the mean position of all points in their cluster",
          "Centroids are moved to the position of the furthest point"
        ],
        correct: 2,
        explanation: "After assigning points to the nearest centroid, each centroid is recalculated as the mean (average) of all points assigned to that cluster: (Σx_i/n, Σy_i/n) for all points in the cluster. This repositioning minimizes the within-cluster sum of squares. The algorithm then reassigns points based on new centroid positions and repeats until convergence.",
        difficulty: "Intermediate",
        concept: "K-Means Clustering",
        hint: "The centroid should be at the 'center' of its assigned points."
      },
      {
        id: 15,
        question: "What is a potential drawback of K-Means clustering?",
        options: [
          "It always finds the global optimum",
          "It requires labeled training data",
          "Results can vary based on initial centroid placement",
          "It only works with Euclidean distance"
        ],
        correct: 2,
        explanation: "K-Means is sensitive to initial centroid positions and may converge to local optima rather than the global optimum. Running the algorithm multiple times with different initializations is recommended. K-Means doesn't require labeled data (it's unsupervised), and while Euclidean distance is common, other distance metrics can be used.",
        difficulty: "Intermediate",
        concept: "K-Means Clustering",
        hint: "Random starting positions can lead to different final results."
      },
      {
        id: 16,
        question: "What does the confusion matrix evaluate?",
        options: [
          "The clustering quality",
          "The performance of a classification model",
          "The variance in the data",
          "The correlation between features"
        ],
        correct: 1,
        explanation: "A confusion matrix evaluates classification model performance by showing the counts of True Positives, True Negatives, False Positives, and False Negatives. From this matrix, we can calculate metrics like accuracy, precision, recall, and F1-score. It's specifically for classification tasks, not clustering, variance analysis, or correlation.",
        difficulty: "Basic",
        concept: "Confusion Matrix",
        hint: "It shows how often the model's predictions match or don't match reality."
      },
      {
        id: 17,
        question: "In a confusion matrix, what are False Positives?",
        options: [
          "Correctly predicted positive cases",
          "Incorrectly predicted as positive when actually negative",
          "Incorrectly predicted as negative when actually positive",
          "Correctly predicted negative cases"
        ],
        correct: 1,
        explanation: "False Positives (Type I errors) occur when the model predicts the positive class but the true label is negative. For example, predicting a patient has a disease when they don't. True Positives are correct positive predictions, False Negatives are incorrectly predicted as negative, and True Negatives are correct negative predictions.",
        difficulty: "Basic",
        concept: "Confusion Matrix",
        hint: "The word 'False' tells you the prediction was wrong."
      },
      {
        id: 18,
        question: "What is the formula for Precision?",
        options: [
          "TP / (TP + FP)",
          "TP / (TP + FN)",
          "TN / (TN + FP)",
          "(TP + TN) / (TP + TN + FP + FN)"
        ],
        correct: 0,
        explanation: "Precision = TP / (TP + FP), which measures the proportion of positive predictions that were actually correct. It answers: 'Of all the cases we predicted as positive, how many truly were positive?' Option B is Recall, option C is Specificity, and option D is Accuracy. High precision means few false alarms.",
        difficulty: "Intermediate",
        concept: "Precision and Recall",
        hint: "Precision focuses on the accuracy of positive predictions."
      },
      {
        id: 19,
        question: "What is the formula for Recall (Sensitivity)?",
        options: [
          "TP / (TP + FP)",
          "TP / (TP + FN)",
          "TN / (TN + FN)",
          "FP / (FP + TN)"
        ],
        correct: 1,
        explanation: "Recall (Sensitivity) = TP / (TP + FN), which measures the proportion of actual positive cases that were correctly identified. It answers: 'Of all the truly positive cases, how many did we successfully detect?' Option A is Precision, option C is part of Specificity, and option D has no standard interpretation. High recall means we're catching most positive cases.",
        difficulty: "Intermediate",
        concept: "Precision and Recall",
        hint: "Recall measures how well we 'recall' or find all the actual positive cases."
      },
      {
        id: 20,
        question: "What is bias in machine learning?",
        options: [
          "The variability of predictions across different training sets",
          "The inability to capture the true relationship between features and target",
          "The difference between training and test error",
          "The correlation between input features"
        ],
        correct: 1,
        explanation: "Bias represents the model's inability to capture the true underlying relationship in the data. High bias leads to underfitting, where the model is too simple and makes systematic errors even on training data. Option A describes variance, option C relates to generalization gap, and option D is about feature correlation, not bias.",
        difficulty: "Basic",
        concept: "Bias-Variance Tradeoff",
        hint: "Bias is about systematic errors from wrong assumptions about the data."
      },
      {
        id: 21,
        question: "What is variance in machine learning?",
        options: [
          "The systematic error in predictions",
          "The variability in predictions when the training dataset changes",
          "The mean squared error on training data",
          "The number of features in the model"
        ],
        correct: 1,
        explanation: "Variance measures how much the model's predictions would change if we trained it on a different dataset. High variance indicates the model is overly sensitive to the specific training data, leading to overfitting. Different training sets would produce very different models. Option A describes bias, not variance.",
        difficulty: "Basic",
        concept: "Bias-Variance Tradeoff",
        hint: "Variance is about how much predictions vary with different training data."
      },
      {
        id: 22,
        question: "A model with high bias and low variance is likely to exhibit:",
        options: [
          "Overfitting",
          "Underfitting",
          "Perfect fit",
          "No systematic pattern"
        ],
        correct: 1,
        explanation: "High bias and low variance indicates underfitting. The model is too simple to capture the underlying patterns (high bias) but produces consistent predictions across different datasets (low variance). Like using a straight line to fit a curved relationship - it will consistently miss the mark. Overfitting occurs with low bias and high variance.",
        difficulty: "Intermediate",
        concept: "Bias-Variance Tradeoff",
        hint: "High bias means the model is too simple to capture the pattern."
      },
      {
        id: 23,
        question: "A model with low bias and high variance is likely to exhibit:",
        options: [
          "Underfitting",
          "Overfitting",
          "Optimal generalization",
          "High training error"
        ],
        correct: 1,
        explanation: "Low bias and high variance indicates overfitting. The model fits the training data very well (low bias) but is overly sensitive to training data specifics (high variance), resulting in poor generalization to new data. Like memorizing training examples rather than learning the underlying pattern. High training error suggests underfitting, not overfitting.",
        difficulty: "Intermediate",
        concept: "Bias-Variance Tradeoff",
        hint: "High variance means the model changes drastically with different training sets."
      },
      {
        id: 24,
        question: "In a neural network, what is the purpose of an activation function?",
        options: [
          "To initialize weights randomly",
          "To introduce non-linearity into the model",
          "To calculate the loss function",
          "To update the learning rate"
        ],
        correct: 1,
        explanation: "Activation functions introduce non-linearity, allowing neural networks to learn complex, non-linear relationships. Without activation functions (or with only linear ones), even a deep network would behave like a single-layer linear model. Common activation functions include sigmoid, ReLU, and tanh. They don't initialize weights, calculate loss, or update learning rates.",
        difficulty: "Intermediate",
        concept: "Neural Networks",
        hint: "Without this, stacking layers would be pointless - it would still be linear."
      },
      {
        id: 25,
        question: "What is the sigmoid activation function formula?",
        options: [
          "max(0, x)",
          "1 / (1 + e^(-x))",
          "tanh(x)",
          "x if x > 0 else 0"
        ],
        correct: 1,
        explanation: "The sigmoid function is σ(x) = 1/(1 + e^(-x)), which maps any input to a value between 0 and 1. Its derivative is σ'(x) = σ(x)(1 - σ(x)), which is useful for backpropagation. Option A is ReLU, option C is tanh, and option D is another way to write ReLU. Sigmoid is commonly used in binary classification output layers.",
        difficulty: "Basic",
        concept: "Activation Functions",
        hint: "This function squashes values to the range (0, 1)."
      },
      {
        id: 26,
        question: "What is a key disadvantage of the sigmoid activation function?",
        options: [
          "It's computationally expensive",
          "It can cause vanishing gradients for large positive or negative inputs",
          "It produces negative outputs",
          "It's not differentiable"
        ],
        correct: 1,
        explanation: "For large positive or negative inputs, the sigmoid's gradient approaches zero, causing vanishing gradients during backpropagation. This means learning stops or becomes very slow in deep networks. The sigmoid is differentiable everywhere, produces outputs in (0,1) not negative values, and is not particularly computationally expensive compared to alternatives.",
        difficulty: "Advanced",
        concept: "Activation Functions",
        hint: "Look at the derivative plot - what happens at the extremes?"
      },
      {
        id: 27,
        question: "In gradient descent, what does the gradient indicate?",
        options: [
          "The current value of the loss function",
          "The direction of steepest increase in the loss function",
          "The optimal learning rate",
          "The number of iterations needed"
        ],
        correct: 1,
        explanation: "The gradient (∇f) is a vector pointing in the direction of steepest increase of the function. In gradient descent, we move in the negative gradient direction to decrease the loss function. The magnitude of the gradient indicates how steep the slope is. The gradient doesn't directly give us the loss value, optimal learning rate, or iteration count.",
        difficulty: "Intermediate",
        concept: "Gradient Descent",
        hint: "A gradient is a directional derivative - it points uphill."
      },
      {
        id: 28,
        question: "In the context of neural networks, what is backpropagation?",
        options: [
          "A method to initialize weights",
          "An algorithm to compute gradients of the loss with respect to weights",
          "A technique to reduce the number of layers",
          "A way to select the best activation function"
        ],
        correct: 1,
        explanation: "Backpropagation is an efficient algorithm for computing gradients of the loss function with respect to all weights in the network using the chain rule. These gradients are then used in gradient descent to update weights. It propagates errors backward through the network from output to input layers. It doesn't initialize weights, reduce layers, or select activation functions.",
        difficulty: "Intermediate",
        concept: "Backpropagation",
        hint: "It's about propagating error information backward through the network."
      },
      {
        id: 29,
        question: "What is the chain rule's role in backpropagation?",
        options: [
          "It determines the network architecture",
          "It allows us to compute derivatives of composite functions layer by layer",
          "It initializes the weights",
          "It selects the best optimization algorithm"
        ],
        correct: 1,
        explanation: "The chain rule enables us to compute derivatives of composite functions by multiplying derivatives of each component. In backpropagation, we compute ∂Loss/∂w = (∂Loss/∂output) × (∂output/∂activation) × (∂activation/∂w), working backward through layers. This mathematical principle is fundamental to training deep networks efficiently.",
        difficulty: "Advanced",
        concept: "Backpropagation",
        hint: "It's a calculus rule for derivatives of nested functions f(g(x))."
      },
      {
        id: 30,
        question: "What problem does the vanishing gradient cause in deep neural networks?",
        options: [
          "Weights become too large",
          "Learning becomes very slow or stops in early layers",
          "The network overfits the training data",
          "The loss function increases"
        ],
        correct: 1,
        explanation: "Vanishing gradients occur when gradients become extremely small as they're backpropagated through many layers, especially with sigmoid/tanh activations. This causes weight updates in early layers to be tiny, essentially stopping learning in those layers. The opposite problem is exploding gradients (large weights). Vanishing gradients don't directly cause overfitting or loss increases.",
        difficulty: "Advanced",
        concept: "Neural Network Training",
        hint: "Gradients get multiplied through layers - what if they're all less than 1?"
      },
      {
        id: 31,
        question: "Which of the following best describes a hyperplane in n-dimensional space?",
        options: [
          "A point in n-dimensional space",
          "A line in n-dimensional space",
          "An (n-1)-dimensional subspace that divides the space",
          "An n-dimensional volume"
        ],
        correct: 2,
        explanation: "A hyperplane in n-dimensional space is an (n-1)-dimensional flat subspace. In 2D, it's a line; in 3D, it's a plane; in higher dimensions, we call it a hyperplane. It divides the space into two half-spaces, which is crucial for classification in SVM. Options A, B, and D don't correctly describe the dimensionality of a hyperplane.",
        difficulty: "Intermediate",
        concept: "Hyperplanes",
        hint: "In 3D space, what 2D shape divides the space into two parts?"
      },
      {
        id: 32,
        question: "In K-Means clustering, what does K represent?",
        options: [
          "The number of features",
          "The number of iterations",
          "The number of clusters to create",
          "The dimensionality of the data"
        ],
        correct: 2,
        explanation: "K is the number of clusters you want the algorithm to create. It's a hyperparameter that must be specified before running K-Means. Choosing the right K is important - too few clusters oversimplifies, too many overfits. Techniques like the elbow method can help select K. It's not related to features, iterations, or data dimensionality.",
        difficulty: "Basic",
        concept: "K-Means Clustering",
        hint: "K-Means divides data into K groups."
      },
      {
        id: 33,
        question: "What is the objective function that K-Means clustering tries to minimize?",
        options: [
          "The maximum distance between any two points",
          "The within-cluster sum of squared distances",
          "The between-cluster distance",
          "The number of iterations"
        ],
        correct: 1,
        explanation: "K-Means minimizes the within-cluster sum of squares (WCSS), also called inertia: Σᵢ Σₓ∈Cᵢ ||x - μᵢ||², where μᵢ is the centroid of cluster i. This makes clusters compact with points close to their centroid. It doesn't maximize between-cluster distance directly, minimize max distance, or minimize iterations.",
        difficulty: "Advanced",
        concept: "K-Means Clustering",
        hint: "We want points to be close to their assigned cluster center."
      },
      {
        id: 34,
        question: "Which statement about SVM kernels is TRUE?",
        options: [
          "Linear kernels always outperform non-linear kernels",
          "The Gaussian (RBF) kernel can handle non-linear decision boundaries",
          "Kernels are only used for regression problems",
          "Using kernels always improves model performance"
        ],
        correct: 1,
        explanation: "The Gaussian (Radial Basis Function) kernel can map data to infinite-dimensional space, allowing SVM to learn complex non-linear decision boundaries. Linear kernels work well for linearly separable data but can't handle non-linear patterns. Kernels are used in classification too, and they can lead to overfitting if not chosen carefully, so they don't always improve performance.",
        difficulty: "Intermediate",
        concept: "SVM Kernels",
        hint: "Some kernels can transform linearly inseparable data to be separable."
      },
      {
        id: 35,
        question: "What is the main difference between supervised and unsupervised learning?",
        options: [
          "Supervised learning uses neural networks, unsupervised doesn't",
          "Supervised learning has labeled target data, unsupervised doesn't",
          "Supervised learning is always more accurate",
          "Unsupervised learning requires more computational power"
        ],
        correct: 1,
        explanation: "The fundamental difference is that supervised learning uses labeled data (input-output pairs) to learn a mapping function, while unsupervised learning finds patterns in unlabeled data. SVM is supervised (needs class labels), clustering is unsupervised (discovers groupings). Both can use neural networks and various computational resources.",
        difficulty: "Basic",
        concept: "Machine Learning Fundamentals",
        hint: "Think about whether you have the 'answers' (labels) during training."
      },
      {
        id: 36,
        question: "In the bias-variance bulls-eye diagram, what does high bias and high variance represent?",
        options: [
          "Predictions are consistently accurate and tightly grouped",
          "Predictions are consistently off-target and widely scattered",
          "Predictions are scattered around the correct target",
          "Predictions are tightly grouped near the target"
        ],
        correct: 1,
        explanation: "High bias means predictions systematically miss the target (off-center), and high variance means predictions are widely scattered (not grouped). This is the worst case - predictions are both inaccurate on average and inconsistent. Option D (low bias, low variance) is ideal. Option C describes low bias with high variance. Option A describes low bias with low variance.",
        difficulty: "Intermediate",
        concept: "Bias-Variance Tradeoff",
        hint: "High bias = missing the target center; high variance = scattered shots."
      },
      {
        id: 37,
        question: "What does the precision metric prioritize?",
        options: [
          "Minimizing False Negatives",
          "Minimizing False Positives",
          "Maximizing True Negatives",
          "Balancing all error types equally"
        ],
        correct: 1,
        explanation: "Precision = TP/(TP + FP) focuses on minimizing False Positives. High precision means when we predict positive, we're usually correct. This is important when false alarms are costly (e.g., spam detection - marking important email as spam is bad). Recall prioritizes minimizing False Negatives. Accuracy balances all error types.",
        difficulty: "Intermediate",
        concept: "Precision and Recall",
        hint: "Precision asks: of our positive predictions, how many were right?"
      },
      {
        id: 38,
        question: "What does the recall metric prioritize?",
        options: [
          "Minimizing False Positives",
          "Minimizing False Negatives",
          "Maximizing True Negatives",
          "Reducing the total number of predictions"
        ],
        correct: 1,
        explanation: "Recall = TP/(TP + FN) focuses on minimizing False Negatives. High recall means we catch most of the actual positive cases. This is crucial when missing a positive case is costly (e.g., disease diagnosis - missing a sick patient is dangerous). Precision prioritizes minimizing False Positives.",
        difficulty: "Intermediate",
        concept: "Precision and Recall",
        hint: "Recall asks: of all actual positives, how many did we find?"
      },
      {
        id: 39,
        question: "Why is the ReLU activation function popular in deep neural networks?",
        options: [
          "It's bounded between 0 and 1",
          "It helps mitigate the vanishing gradient problem",
          "It always produces the best accuracy",
          "It requires less memory than other activation functions"
        ],
        correct: 1,
        explanation: "ReLU (Rectified Linear Unit) f(x) = max(0,x) helps mitigate vanishing gradients because its gradient is 1 for positive inputs (not approaching zero like sigmoid). This allows better gradient flow in deep networks. It's unbounded above (not 0-1), doesn't guarantee best accuracy, and doesn't inherently use less memory. It can suffer from dying ReLU problem though.",
        difficulty: "Advanced",
        concept: "Activation Functions",
        hint: "Think about the gradient for positive inputs - it's constant."
      },
      {
        id: 40,
        question: "What is the fundamental assumption of K-Means clustering?",
        options: [
          "Clusters have equal variance and are roughly spherical",
          "Data is linearly separable",
          "All features are equally important",
          "The number of clusters equals the number of features"
        ],
        correct: 0,
        explanation: "K-Means assumes clusters are convex and roughly spherical with similar variances, since it uses Euclidean distance and assigns points to nearest centroids. It struggles with elongated or irregularly shaped clusters. It doesn't require linear separability, doesn't assume equal feature importance (though scaling helps), and K is independent of feature count.",
        difficulty: "Advanced",
        concept: "K-Means Clustering",
        hint: "K-Means uses distance to centroids - what shape works best for this?"
      },
      {
        id: 41,
        question: "In SVM, what is the 'kernel trick'?",
        options: [
          "A method to reduce the number of support vectors",
          "A way to compute dot products in high-dimensional space without explicit transformation",
          "A technique to automatically select the best kernel",
          "A method to initialize support vectors"
        ],
        correct: 1,
        explanation: "The kernel trick allows computing dot products in high (even infinite) dimensional feature space without explicitly transforming the data. Instead of mapping x → φ(x) and computing φ(x)·φ(x'), we directly compute K(x,x') = φ(x)·φ(x'). This makes non-linear SVM computationally feasible. It doesn't reduce support vectors, select kernels automatically, or initialize anything.",
        difficulty: "Advanced",
        concept: "SVM Kernels",
        hint: "It's a computational shortcut for working in high-dimensional spaces."
      },
      {
        id: 42,
        question: "What happens during the forward pass in a neural network?",
        options: [
          "Gradients are computed and weights are updated",
          "Input data flows through the network to produce predictions",
          "The loss function is minimized",
          "Weights are initialized randomly"
        ],
        correct: 1,
        explanation: "In the forward pass, input data flows through the network layer by layer: inputs → weights → activations → outputs, ultimately producing predictions. No gradients are computed or weights updated (that's backward pass). The loss is calculated after forward pass but not minimized. Weight initialization happens before training begins.",
        difficulty: "Intermediate",
        concept: "Neural Networks",
        hint: "Data moves from input toward output - which direction is that?"
      },
      {
        id: 43,
        question: "In hierarchical clustering, what is a dendrogram used for?",
        options: [
          "To calculate distances between points",
          "To visualize the hierarchical relationship and decide where to cut for clusters",
          "To initialize cluster centers",
          "To compute the optimal number of features"
        ],
        correct: 1,
        explanation: "A dendrogram is a tree-like diagram showing the hierarchical relationships between clusters as they merge (agglomerative) or split (divisive). The height of branches indicates dissimilarity. By cutting the dendrogram at different heights, you can obtain different numbers of clusters. It doesn't calculate distances, initialize centers, or determine feature count.",
        difficulty: "Intermediate",
        concept: "Clustering Approaches",
        hint: "It's a tree diagram showing how clusters join together."
      },
      {
        id: 44,
        question: "What is the purpose of the learning rate in gradient descent?",
        options: [
          "To determine the number of iterations",
          "To control the step size when updating weights",
          "To calculate the loss function",
          "To initialize the weights"
        ],
        correct: 1,
        explanation: "The learning rate (α or η) controls how big a step we take in the direction of the negative gradient when updating weights: w_new = w_old - α × gradient. Too large causes instability/divergence; too small causes slow convergence. It doesn't determine iterations, calculate loss, or initialize weights. Choosing an appropriate learning rate is crucial for effective training.",
        difficulty: "Intermediate",
        concept: "Gradient Descent",
        hint: "It determines how far we move in the gradient direction."
      },
      {
        id: 45,
        question: "Which scenario would benefit most from using a soft margin SVM?",
        options: [
          "Perfectly linearly separable data with no noise",
          "Real-world noisy data that is approximately linearly separable",
          "Data that requires a non-linear decision boundary",
          "Data with only two features"
        ],
        correct: 1,
        explanation: "Soft margin SVM is ideal for real-world data with noise and outliers that is approximately (but not perfectly) linearly separable. It tolerates some misclassifications to avoid overfitting to noise. Hard margin works for perfectly separable data. For non-linear boundaries, you'd use kernels (with soft margin). Feature count doesn't determine margin type.",
        difficulty: "Intermediate",
        concept: "SVM Margins",
        hint: "Real-world data is rarely perfect - which margin type handles this?"
      },
      {
        id: 46,
        question: "What is the main disadvantage of hierarchical clustering compared to K-Means?",
        options: [
          "It always produces worse results",
          "It has higher computational complexity for large datasets",
          "It requires specifying the number of clusters in advance",
          "It can only use Euclidean distance"
        ],
        correct: 1,
        explanation: "Hierarchical clustering has O(n³) or O(n²) complexity (depending on linkage method) compared to K-Means' O(nkdi) where k=clusters, d=dimensions, i=iterations. This makes hierarchical impractical for large datasets. However, it doesn't require pre-specifying cluster count (advantage), can use various distance metrics, and doesn't necessarily produce worse results.",
        difficulty: "Advanced",
        concept: "Clustering Approaches",
        hint: "Think about having to compute distances between all pairs of clusters."
      },
      {
        id: 47,
        question: "What does the Mean Squared Error (MSE) measure in machine learning?",
        options: [
          "The number of classification errors",
          "The average squared difference between predicted and actual values",
          "The correlation between features",
          "The bias of the model"
        ],
        correct: 1,
        explanation: "MSE = (1/N)Σ(predicted - actual)² measures the average squared error between predictions and true values. It's commonly used for regression problems and neural network training. Squaring penalizes larger errors more heavily. It doesn't count classification errors (that's accuracy/confusion matrix), measure feature correlation, or directly measure bias.",
        difficulty: "Basic",
        concept: "Loss Functions",
        hint: "It's about how far predictions are from actual values, on average."
      },
      {
        id: 48,
        question: "In the context of SVM, what does maximizing the margin help achieve?",
        options: [
          "Faster training time",
          "Better generalization to new data",
          "Reduced number of features",
          "Higher training accuracy only"
        ],
        correct: 1,
        explanation: "Maximizing the margin (the distance between the decision boundary and nearest points) provides a buffer zone that helps the model generalize better to unseen data. A larger margin means the model is less sensitive to small variations and noise. It doesn't necessarily speed training, reduce features, or focus solely on training accuracy.",
        difficulty: "Intermediate",
        concept: "SVM Margins",
        hint: "A wider buffer zone makes the classifier more robust to variations."
      },
      {
        id: 49,
        question: "What is a key difference between K-Means and hierarchical clustering?",
        options: [
          "K-Means requires pre-specifying the number of clusters, hierarchical doesn't",
          "Hierarchical is always faster",
          "K-Means can only use Manhattan distance",
          "Hierarchical can't handle numerical data"
        ],
        correct: 0,
        explanation: "K-Means requires you to specify K (number of clusters) beforehand, while hierarchical clustering creates a dendrogram that allows you to choose the number of clusters afterward by cutting at different heights. Hierarchical is typically slower for large datasets. K-Means typically uses Euclidean distance but can use others. Both handle numerical data.",
        difficulty: "Intermediate",
        concept: "Clustering Approaches",
        hint: "One requires knowing K upfront, the other lets you decide later."
      },
      {
        id: 50,
        question: "Why is feature scaling important for SVM and K-Means?",
        options: [
          "It increases the number of support vectors",
          "Both algorithms use distance metrics that are sensitive to feature scales",
          "It reduces the need for cross-validation",
          "It eliminates the need for regularization"
        ],
        correct: 1,
        explanation: "SVM and K-Means both use distance-based calculations (Euclidean distance typically) which are sensitive to feature scales. Features with larger scales dominate the distance calculation. Standardizing features (mean=0, std=1) or normalizing (0-1 range) ensures all features contribute appropriately. Scaling doesn't affect support vector count, eliminate cross-validation, or replace regularization.",
        difficulty: "Advanced",
        concept: "Data Preprocessing",
        hint: "If one feature ranges 0-1 and another 0-1000, which dominates distance?"
      }
    ]
  }
};

export default chaptersData;
