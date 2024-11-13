---
permalink: /blog/
title: "Posts"
author_profile: true
---

<details>

<summary style="cursor: pointer; font-weight: bold;">Towards Winning Insights: A Guide to Collecting and Curating Soccer Data</summary>

<h3>Introduction</h3>

<p>
  Soccer has always been a game of skill and strategy, but in recent years, the explosion of data is revolutionizing how we approach the sport. The sheer volume of data 
  available is transforming how teams measure performance and make decisions. This surge of data is changing the game, and those who are able to use it to generate powerful 
  insights have the potential to gain a competitive edge that could change the course of history.
  
<div style="text-align:center; margin: 20px;">
  <img src="/images/treble.png" alt="Treble Winners Manchester United" style="width: 450px;"/>
</div>

</p>

<p>
  One key to the generation of meaningful insights is effective data collection and curation. The quality of 
  your analysis will heavily depend on the quality of the data you are working with, so it is crucial that these steps are not overlooked. In this post, I 
  will walk you through the process of collection and curation for soccer data. Now, let's lace up our cleats and set you on the path towards game-winning insights!
</p>

<h3>Research Question</h3>

<p>
  The question motivating our analysis is: Can we use match data to build a model that accurately quantifies and tracks a team’s form? In soccer, "form" goes beyond wins, losses, 
  and draws; it encompasses the underlying quality and consistency of a team’s performance over time. By tracking a team’s form with detailed match data, we may gain a clearer 
  picture of trends that may not be visible in the final scores alone. For example, a team might be creating high-quality chances consistently, even if the results aren’t showing 
  on the scoreboard yet. Conversely, a team winning matches might be showing signs of poor form if they’re frequently outperformed by their opponents.
</p>

<div style="text-align:center; margin: 20px;">
  <img src="/images/trends.png" alt="Trend Photo" style="width: 450px;"/>
</div>

<p>
  Quantifying form allows us to recognize when a team is on an upward trajectory or dealing with serious issues before they show up in the win-loss column. By analyzing factors 
  such as possession, strength of the opponent, chances created, and recent results, we may be able to gain a deeper understanding of when things are going especially well or 
  poorly. This approach would enable us to look beyond short-term outcomes and assess the true health of a team’s performance. If successful, a model like this could help teams 
  know when they are improving enough to trust the process, or when it may be time to hit the panic button.
</p>

<h3>Data Collection</h3>

<p>
  For the initial collection, I decided to gather Premier League match data for Manchester United between 2018 and 2024. I obtained the data using <a 
  href="https://www.api-football.com">API-Football</a>, a high-quality, easy to use API with tons of available data. There are two ways to use the API, through the API-Sports 
  interface or with RapidAPI. I tried both and had an easier time using the API-Sports version, so that is what I will use for the demonstration.
</p>

<p>
  API-Football is free to use. The free plan comes with 100 requests per day, and gives you access to all the competitions and endpoints. Upon signing up, you will be given an API 
  key and access to a dashboard to track your requests. The most important factor in ensuring that my data was gathered ethically was keeping the API key private. I also read 
  through the terms and service, which emphasized the importance of not selling the data. The data is provided for users to create projects, and that is all we are doing today, 
  so we can move forward confident that our use of the data is ethical. You can find the documentation <a href="https://www.api-football.com/documentation-v3">here</a>. I would 
  highly recommend having it pulled up as you work. Let's code!
</p>

<p>
  First, let's make sure your API key is only visible to you. Create a txt file, then copy and past your API key on the first line.
</p>

<div style="text-align:center; margin: 20px;">
  <img src="/images/txt.png" alt="text file" style="width: 450px;"/>
</div>

<p>
  Next, make sure your repository has a .gitignore file and add the txt file to it.
</p>

<div style="text-align:center; margin: 20px;">
  <img src="/images/gitignore.png" alt="gitignore" style="width: 450px;"/>
</div>

<p>
  Perfect! Now your API key will not show up on your github page when make changes to your repository. In order to access it during collection, you can use the following code:
</p>

<pre style="font-size: 12px; padding: 10px; line-height: 1.2;"><code class="language-python">
  with open('soccer_key.txt', 'r') as file:
    api_key = file.read().strip()
</code></pre>

<p>
  This code reads your txt file and makes your key a variable. Anytime you need to use it, you can simply call the api_key variable, ensuring that the actual key value is never      exposed. 
</p>

<p>
  I started by using the fixtures endpoint to pull all of Manchester United's Premier League fixtures for each year. When I tried to run a loop that would pull the data for all 
  the years in one go, it would only fetch some of them, so I did it one year at a time instead. Here is an example:
</p>

<pre style="font-size: 12px; padding: 10px; line-height: 1.2;"><code class="language-python">
import requests
import pandas as pd
import time

  url = "https://v3.football.api-sports.io/fixtures"

# Initial query parameters: Premier League (league=39), season for the current year
querystring = {"league": "39", "season": "2018", "team": "33"}

headers = {
    "x-apisports-key": str(api_key)
}

# Initialize an empty list to hold all fixture statistics
fixture_stats = []

# Make the API request
response = requests.get(url, headers=headers, params=querystring)

time.sleep(60)  # Wait for 60 seconds

result = response.json()

# Extract the fixtures data from the current page
fixtures = result.get('response', [])

# Add fixture statistics to the list
for fixture in fixtures:
    fixture_info = {
        'fixture_id': fixture['fixture']['id'],
        'date': fixture['fixture']['date'],
        'status': fixture['fixture']['status']['long'],
        'home_team': fixture['teams']['home']['name'],
        'away_team': fixture['teams']['away']['name'],
        'home_score': fixture['goals']['home'],
        'away_score': fixture['goals']['away'],
        'league': fixture['league']['name']
    }
    fixture_stats.append(fixture_info)

# Create a DataFrame from the fixture statistics list
df_fixtures_2018 = pd.DataFrame(fixture_stats)
</code></pre>

</details>

<details>

<summary style="cursor: pointer; font-weight: bold;">Kaggle Slayer: Introduction to XGBoost</summary>

<h3>Introduction</h3>

<p>
  Every Kaggle competition presents unique obstacles—requiring the sharpest of skills and the most powerful of tools. In this post, I will introduce you to the greatest of weapons—one every data scientist must have in their arsenal: <span 
  style="color:#007BFF;">XGBoost</span>.
</p>

<div style="text-align:center; margin: 20px;">
  <img src="/images/dragon_slayer.png" alt="Dragon Slayer" style="width: 450px;"/>
</div>

<p>
  Short for “Extreme Gradient Boosting,” XGBoost is a highly scalable, efficient decision tree machine learning library that makes use of the gradient boosting framework to provide excellent results in classification and regression tasks. If you're just     
  entering the battlefield of machine learning, this tutorial will teach you the basics of XGBoost and help you begin using it to conquer your next challenge.
</p>

<h3>Foundations</h3>

<p>
  The term “gradient boosting” comes from the concept of “boosting”—improving a single weak model (in this case, a decision tree) by combining it with other weak models to form a collectively strong model.
</p>

<p>
  In gradient boosting, an ensemble of shallow decision trees is iteratively trained, with each tree focusing on correcting the residual errors of the previous model. The final prediction is a weighted sum of all the tree predictions, creating a robust model   from individually weak learners.
</p>

<p>
  If you want to learn more about any of these concepts, I highly recommend checking out the YouTube channel "StatQuest with Josh Starmer." He offers a ton of simple yet thorough videos on machine learning. Here are a few of his playlists that explain the     foundational components of XGBoost:
</p>

<ul>
  <li><a href="https://www.youtube.com/watch?v=_L39rN6gz7Y&list=PLblh5JKOoLUKAtDViTvRGFpphEc24M-QH">Decision Trees</a></li>
  <li><a href="https://www.youtube.com/watch?v=3CC4N4z3GJc&list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6">Gradient Boosting</a></li>
</ul>

<div style="text-align:center;">
  <img src="/images/xgboost.png" alt="XGBoost"/>
</div>

<h3>Why XGBoost?</h3>

<p>
  So what is it that makes XGBoost reign king when it comes to working with tabular data? For the purpose of this post, I have narrowed it down to four key factors:
</p>

<ul>
  <li>
      Efficiency: XGBoost is extremely fast. It uses parallel processing and optimization techniques, allowing it to construct different parts of the trees simultaneously. This significantly expedites the process when compared to traditional gradient boosting         methods, which build each tree one by one. It is also sparse-aware, which just means it only stores non-zero values in order to conserve memory. The method's handling of the data enables it to excel with large datasets, making it a great option for   
      applications in big data.
  </li>
  <li>
      Flexibility: XGBoost is highly adaptable, offering built-in functions for handling missing values and responding well to outliers. It includes tons of tuning parameters, which allows users to customize the model depending on the task and dataset. By creating a simple parameter grid, users can quickly test various combinations of hyperparameters to identify the optimal configuration for their problem.
  </li>
  <li>
      Performance: As XGBoost is a tree-based method, it is capable of identifying complex, non-linear relationships in the data. It also uses regularization parameters (<a href="https://www.youtube.com/watch?v=NGf0voTMlcs">L1</a> and <a   
      href="https://www.youtube.com/watch?v=Q81RR3yKn30">L2</a>)to prevent overfitting, enhancing the model's ability to generalize well to out-of-sample data. These capabilities allow XGBoost to be a consistent, high performer, both in Kaggle competitions   
      and in real-world settings. 
  </li>
  <li>
      Community: Lastly, XGBoost is supported by a massive community, meaning extensive up-to-date documentation, tutorials, and resources are readily available to users. There are implementations of XGBoost in Python, R, Julia, Java, and more, so it is 
      accessible for everyone. It also integrates seamlessly with popular data science libraries, such as sci-kit learn and TensorFlow, which makes it easy for users to incorporate it into their workflows.
  </li>
</ul>

<h3>Applications</h3>

<div style="text-align:center; margin: 20px;">
  <img src="/images/fernando_tatis.png" alt="baseball" style="width: 450px;"/>
</div>

<p>
  XGBoost has been successfully applied to:
</p>

<ul>
    <li>
        Sports: Predicting Match Outcomes, Sports Betting, Play Calling, Personnel Strategy, Draft Decisions, Injury Prediction and Prevention
    </li>
    <li>
        Business: Customer Segmentation, Credit Scoring, Risk Assessment, Sale Forecasting
    </li>
    <li>
        Health: Precision Medicine, Healthcare Cost Prediction, Pharmaceutical Studies, Genomics 
    </li>
</ul>

<h3>Demo</h3>

<p>
    Now, I will show you how easy it is start using XGBoost. Below, I trained a simple XGBoost model and compared to three common methods. 
</p>

<pre style="font-size: 12px; padding: 10px; line-height: 1.2;"><code class="language-python">
# Load the dataset
data = pd.read_csv('insurance.csv')

# Define the features and the target variable
X, y = data.iloc[:, :-1], data.iloc[:, -1]

# Define the categorical and numerical features
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Define a parameter grid for XGBoost
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [3, 4, 5, 6],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0]
}

# Perform RandomizedSearchCV for XGBoost
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', models['XGBoost'])])

random_search = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=param_grid,
                                   n_iter=50, scoring='neg_mean_squared_error', cv=3, verbose=1, random_state=42, n_jobs=-1)

random_search.fit(X_train, y_train)

# Get the best model
best_xgb_model = random_search.best_estimator_

# Train and evaluate each model
results = {}
for name, model in models.items():
    if name == 'XGBoost':
        pipeline = best_xgb_model
    else:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    results[name] = rmse

# Print the results
for name, rmse in results.items():
    print(f'{name} - RMSE: {rmse}')
</code></pre>

<div style="text-align:center; margin: 20px;">
  <img src="/images/xgboost-demo 2.png" alt="Demo Results" style="width: 450px;"/>
</div>

<p>
    Voila! XGBoost has the lowest RMSE, meaning that its predicted values were closest to the true values in data.
</p>

<h3>Conclusion</h3>

<p>
    With this brief introduction, I hope you have started to appreciate the power of XGBoost. I have avoided diving into the math behind the method here, but if you're interested, it’s fascinating to see what’s happening under the hood.
</p>

<p>
    If you are interested, you can find more information <a href="https://www.geeksforgeeks.org/xgboost/">here</a>.
</p>
  
<p>
    Now that you have some basic information, I would encourage you to try it yourself. With a bit of practice, I am confident you will become a   
    master of XGBoost, slaying even the mightiest of Kaggle competitions in no time. 
</p>

</details>






