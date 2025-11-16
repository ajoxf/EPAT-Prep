// Chapter data for Quantitative Finance LMS

const chaptersData = {
  'MLT-01': {
    title: 'Machine Learning Techniques - I',
    description: 'Fundamentals of Machine Learning for Quantitative Finance',
    questions: [
      {
        id: 1,
        question: "What is the primary goal of supervised learning?",
        options: [
          "To find patterns in unlabeled data",
          "To predict outcomes based on labeled training data",
          "To reduce the dimensionality of data",
          "To cluster similar data points together"
        ],
        correct: 1,
        explanation: "Supervised learning uses labeled training data to learn a mapping function that can predict outcomes for new, unseen data. The algorithm learns from examples where the correct answer is known.",
        difficulty: "Basic",
        concept: "Supervised Learning",
        hint: "Think about learning with a teacher who provides the correct answers."
      },
      {
        id: 2,
        question: "Which of the following is an example of unsupervised learning?",
        options: [
          "Linear regression",
          "Decision trees",
          "K-means clustering",
          "Logistic regression"
        ],
        correct: 2,
        explanation: "K-means clustering is an unsupervised learning algorithm that groups similar data points together without using labeled data. Linear regression, decision trees, and logistic regression are all supervised learning methods.",
        difficulty: "Basic",
        concept: "Unsupervised Learning",
        hint: "Look for the algorithm that doesn't require labeled training data."
      },
      {
        id: 3,
        question: "What is overfitting in machine learning?",
        options: [
          "When a model performs well on training data but poorly on test data",
          "When a model performs poorly on both training and test data",
          "When a model is too simple to capture patterns",
          "When training takes too long"
        ],
        correct: 0,
        explanation: "Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor generalization to new data. The model becomes too complex and specific to the training set.",
        difficulty: "Intermediate",
        concept: "Model Evaluation",
        hint: "Consider what happens when a model memorizes rather than learns."
      },
      {
        id: 4,
        question: "What is the purpose of cross-validation?",
        options: [
          "To increase the size of the training dataset",
          "To assess how well a model will generalize to independent data",
          "To speed up the training process",
          "To reduce the number of features"
        ],
        correct: 1,
        explanation: "Cross-validation is a technique used to evaluate model performance by partitioning data into subsets, training on some and testing on others. This helps estimate how well the model will perform on unseen data.",
        difficulty: "Intermediate",
        concept: "Model Validation",
        hint: "Think about testing a model's ability to work with new data."
      },
      {
        id: 5,
        question: "In the context of bias-variance tradeoff, what does high bias indicate?",
        options: [
          "The model is too complex",
          "The model is underfitting the data",
          "The model has high variance",
          "The model is overfitting the data"
        ],
        correct: 1,
        explanation: "High bias indicates that the model is too simple and is underfitting the data. It makes strong assumptions and fails to capture the underlying patterns in the data.",
        difficulty: "Intermediate",
        concept: "Bias-Variance Tradeoff",
        hint: "Bias relates to assumptions made by the model."
      },
      {
        id: 6,
        question: "What is the main advantage of ensemble methods?",
        options: [
          "They are faster to train than single models",
          "They require less data than single models",
          "They combine multiple models to improve prediction accuracy",
          "They eliminate the need for feature engineering"
        ],
        correct: 2,
        explanation: "Ensemble methods combine predictions from multiple models to achieve better performance than any single model. Methods like Random Forest and Gradient Boosting use this approach to reduce variance and improve accuracy.",
        difficulty: "Advanced",
        concept: "Ensemble Learning",
        hint: "Many heads are better than one."
      },
      {
        id: 7,
        question: "What is regularization in machine learning?",
        options: [
          "A technique to normalize input features",
          "A method to prevent overfitting by adding a penalty term",
          "A way to increase model complexity",
          "A data preprocessing step"
        ],
        correct: 1,
        explanation: "Regularization adds a penalty term to the loss function to discourage complex models and prevent overfitting. Common techniques include L1 (Lasso) and L2 (Ridge) regularization.",
        difficulty: "Intermediate",
        concept: "Regularization",
        hint: "Think about constraining model complexity."
      },
      {
        id: 8,
        question: "Which metric is most appropriate for imbalanced classification problems?",
        options: [
          "Accuracy",
          "Mean Squared Error",
          "F1-Score",
          "R-squared"
        ],
        correct: 2,
        explanation: "F1-Score is the harmonic mean of precision and recall, making it more suitable for imbalanced datasets where accuracy can be misleading. MSE and R-squared are for regression, and accuracy doesn't account for class imbalance.",
        difficulty: "Advanced",
        concept: "Model Evaluation Metrics",
        hint: "Consider a metric that balances precision and recall."
      },
      {
        id: 9,
        question: "What is feature engineering?",
        options: [
          "The process of selecting machine learning algorithms",
          "The process of creating new features from existing data",
          "The process of splitting data into train and test sets",
          "The process of normalizing features"
        ],
        correct: 1,
        explanation: "Feature engineering involves creating new features or transforming existing ones to improve model performance. This can include creating interaction terms, polynomial features, or domain-specific transformations.",
        difficulty: "Basic",
        concept: "Feature Engineering",
        hint: "Think about preparing and transforming input variables."
      },
      {
        id: 10,
        question: "What is the curse of dimensionality?",
        options: [
          "Having too few features for modeling",
          "The exponential increase in data needed as dimensions increase",
          "The problem of categorical features",
          "The issue of missing data"
        ],
        correct: 1,
        explanation: "The curse of dimensionality refers to the exponential increase in volume associated with adding dimensions (features). As dimensions increase, the amount of data needed to maintain the same density grows exponentially.",
        difficulty: "Advanced",
        concept: "Dimensionality",
        hint: "Consider what happens when you have many features but limited data."
      }
    ]
  },

  'ST-01': {
    title: 'Statistics - I',
    description: 'Fundamental Statistical Concepts',
    questions: [
      {
        id: 1,
        question: "What does the Central Limit Theorem state?",
        options: [
          "All distributions are normal",
          "The sampling distribution of the mean approaches normal as sample size increases",
          "Large samples always have normal distributions",
          "The population must be normally distributed"
        ],
        correct: 1,
        explanation: "The Central Limit Theorem states that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the population's distribution.",
        difficulty: "Intermediate",
        concept: "Central Limit Theorem",
        hint: "Focus on what happens to the sampling distribution."
      },
      {
        id: 2,
        question: "What is the difference between population and sample variance formulas?",
        options: [
          "There is no difference",
          "Sample variance divides by n-1 instead of n",
          "Population variance divides by n-1 instead of n",
          "Sample variance uses absolute deviations"
        ],
        correct: 1,
        explanation: "Sample variance divides by n-1 (degrees of freedom) instead of n to provide an unbiased estimate of the population variance. This is called Bessel's correction.",
        difficulty: "Basic",
        concept: "Variance",
        hint: "Consider degrees of freedom adjustment."
      },
      {
        id: 3,
        question: "What does a p-value represent?",
        options: [
          "The probability that the null hypothesis is true",
          "The probability of observing results as extreme as those observed, assuming the null hypothesis is true",
          "The probability that the alternative hypothesis is true",
          "The significance level of the test"
        ],
        correct: 1,
        explanation: "A p-value is the probability of obtaining test results at least as extreme as the observed results, assuming that the null hypothesis is true. It does not give the probability that the null hypothesis itself is true.",
        difficulty: "Intermediate",
        concept: "Hypothesis Testing",
        hint: "Think about the probability of the data, not the hypothesis."
      },
      {
        id: 4,
        question: "What is Type I error in hypothesis testing?",
        options: [
          "Failing to reject a false null hypothesis",
          "Rejecting a true null hypothesis",
          "Accepting the alternative hypothesis",
          "Failing to reject a true null hypothesis"
        ],
        correct: 1,
        explanation: "Type I error occurs when we reject the null hypothesis when it is actually true (false positive). The probability of Type I error is denoted by α (alpha), the significance level.",
        difficulty: "Basic",
        concept: "Hypothesis Testing",
        hint: "This is also called a false positive."
      },
      {
        id: 5,
        question: "What is the relationship between confidence level and confidence interval width?",
        options: [
          "Higher confidence level leads to narrower intervals",
          "Higher confidence level leads to wider intervals",
          "They are not related",
          "Confidence level doesn't affect interval width"
        ],
        correct: 1,
        explanation: "A higher confidence level (e.g., 99% vs 95%) requires a wider interval to ensure the true parameter is captured with higher probability. There's a tradeoff between confidence and precision.",
        difficulty: "Intermediate",
        concept: "Confidence Intervals",
        hint: "Think about the tradeoff between certainty and precision."
      }
    ]
  },

  'FIN-01': {
    title: 'Fixed Income',
    description: 'Fixed Income Securities and Markets',
    questions: [
      {
        id: 1,
        question: "What is duration of a bond?",
        options: [
          "The time until maturity",
          "A measure of interest rate sensitivity",
          "The coupon payment frequency",
          "The bond's credit rating"
        ],
        correct: 1,
        explanation: "Duration measures a bond's sensitivity to interest rate changes. It represents the weighted average time to receive cash flows and indicates how much the bond's price will change for a 1% change in yield.",
        difficulty: "Intermediate",
        concept: "Duration",
        hint: "Think about price sensitivity to rate changes."
      },
      {
        id: 2,
        question: "What is convexity in bond pricing?",
        options: [
          "The linear relationship between price and yield",
          "The curvature in the price-yield relationship",
          "The bond's maturity date",
          "The issuer's creditworthiness"
        ],
        correct: 1,
        explanation: "Convexity measures the curvature of the relationship between bond prices and yields. It captures the fact that the price-yield relationship is not linear, providing a more accurate estimate of price changes than duration alone.",
        difficulty: "Advanced",
        concept: "Convexity",
        hint: "Duration is linear; this captures the curve."
      },
      {
        id: 3,
        question: "What happens to bond prices when interest rates rise?",
        options: [
          "Bond prices rise",
          "Bond prices fall",
          "Bond prices remain unchanged",
          "Only zero-coupon bond prices are affected"
        ],
        correct: 1,
        explanation: "Bond prices and interest rates have an inverse relationship. When interest rates rise, existing bonds with lower rates become less attractive, causing their prices to fall.",
        difficulty: "Basic",
        concept: "Interest Rate Risk",
        hint: "Think about the inverse relationship."
      },
      {
        id: 4,
        question: "What is the yield to maturity (YTM)?",
        options: [
          "The coupon rate of the bond",
          "The total return anticipated if the bond is held until maturity",
          "The current yield on the bond",
          "The bond's face value"
        ],
        correct: 1,
        explanation: "Yield to maturity is the total return anticipated on a bond if held until maturity. It accounts for all coupon payments, the difference between purchase price and face value, and assumes all coupons are reinvested at the same rate.",
        difficulty: "Intermediate",
        concept: "Yield to Maturity",
        hint: "Consider the total return from holding to maturity."
      },
      {
        id: 5,
        question: "What distinguishes a zero-coupon bond?",
        options: [
          "It has a variable coupon rate",
          "It pays no periodic interest and is sold at a discount",
          "It has zero credit risk",
          "It matures in zero years"
        ],
        correct: 1,
        explanation: "A zero-coupon bond makes no periodic interest payments. Instead, it is sold at a deep discount to its face value and pays the full face value at maturity. The return comes entirely from the price appreciation.",
        difficulty: "Basic",
        concept: "Zero-Coupon Bonds",
        hint: "Think about bonds with no regular payments."
      }
    ]
  }
};

export default chaptersData;
