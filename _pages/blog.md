---
permalink: /blog/
title: "Posts"
author_profile: true
---

<details>

<summary style="cursor: pointer; font-weight: bold;">Manchester United's Form in the Premier League: Key Insights and a Streamlit App</summary>

<p>
    In this post, I more thoroughly explore the data we collected in the previous post. I will summarize a key insight that I derived and explain the streamlit app that I   
    created to help gain a better understanding of the trends in Manchester United's performance over the years.
</p>

<h3>Introduction</h3>

Expected goals (xG) has become one of the most talked-about metrics in soccer analytics in recent years. This emergence is so recent, that the API I used to gather the data did not even start including it as a match statistic until 2023. Expected goals offers a way to quantify the quality of scoring chances in a match. But how well does it align with match results? To explore this, I visualized expected goals by match result using boxplots. 

<div style="text-align:center; margin: 20px;">
  <img src="/images/newplot.png" alt="boxplot" style="width: 750px;"/>
</div>

<h3>Key Insight: The Value of Expected Goals</h3>

<p>
To me, the disparity among the boxplots was shocking. Upon initial observation, it seems that there may be a significant difference in the mean number of expected goals depending on the result of the match. Wins tend to show higher expected goals, while losses appear to have lower values, with draws falling somewhere in between. While this pattern is compelling, it’s important to validate it statistically. Running hypothesis tests, such as an ANOVA or pairwise t-tests, could confirm whether the differences in means are statistically significant.

If this hypothesis holds true, it would highlight the predictive and analytical value of expected goals. This insight would be critical as we move toward building a model to track and evaluate Manchester United's form over time. Including expected goals as a feature in our model could provide a deeper understanding of performance trends, beyond surface-level statistics like possession or shot counts. This would allow us to assess whether the team is generating high-quality scoring chances consistently, regardless of match results, and how that aligns with overall form.

Incorporating expected goals in our analysis could also provide a more nuanced lens through which to view Manchester United’s matches, highlighting underperformance in games where the xG was high but the result didn’t align. Conversely, it could reveal overperformance when the team wins with a low xG. These insights could serve to help us identify areas of improvement or recognize aspects of the game in which the team is successful.

Even this initial exploration of expected goals reveals its potential importance in understanding team form. It’s no surprise this metric is considered revolutionary, as it has redefined how we analyze the game beyond the traditional match statistics. If you are interested in learning more about expected goals, this <a href="https://www.amazon.com/Expected-Goals-conquered-football-changed/dp/0008484031">book</a> may be worth a read.
</p>
  
</details>
<details>

<summary style="cursor: pointer; font-weight: bold;">Towards Winning Insights: A Guide to Collecting and Curating Soccer Data</summary>

<p>
  This is a simple guide for gathering soccer data via an API and preparing it for basic analysis. The motivation for this demonstration is the desire to understand trends in a 
  team's form beyond results.
</p>

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
  interface or with RapidAPI. I tried both and had an easier time using the API-Sports version, so that is what I will use for the demonstration. API-Football is free to use. The 
  free plan comes with 100 requests per day, and gives you access to all the competitions and endpoints. Upon signing up, you will be given an API 
  key and access to a dashboard to track your requests. 
</p>

<div style="text-align:center; margin: 20px;">
  <img src="/images/dashboard.png" alt="API-Sports Dashboard" style="width: 750px;"/>
</div>

<p>
  The data I pulled for this project required 240 requests, so you can either gather the data over a few days or upgrade to a plan that allows for more daily requests. For this 
  API, the most important step for ensuring ethical practice is securing your API key. I read through the terms and service, and since the data we are using is 
  for a personal project, we can move forward confident that our use of the data is ethical as long as we keep the key private. 
</p>  

<p>
  You can find the API documentation <a href="https://www.api-football.com/documentation-v3">here</a>. I would highly recommend having it pulled up as you work. It really helped 
  me understand what data was available and showed me how to get exactly what I wanted. Let's code!
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
  the years in one go, it would only fetch some of them, so I did it one year at a time instead. Here is how I did it:
</p>

<pre style="font-size: 12px; padding: 10px; line-height: 1.2;"><code class="language-python">
import requests
import pandas as pd
import time

url = "https://v3.football.api-sports.io/fixtures"

# Initial query parameters: Premier League ID (39), Year, and team ID for Manchester United (33)

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

print(f"Dataframe for 2018:")
print(df_fixtures_2018)
</code></pre>

<p>
  Once I created a dataframe for every year, I combined them to create a single dataframe.
</p>

<pre style="font-size: 12px; padding: 10px; line-height: 1.2;"><code class="language-python">
fixture_df = pd.concat([df_fixtures_2018, df_fixtures_2019, df_fixtures_2020, df_fixtures_2021, df_fixtures_2022, df_fixtures_2023, df_fixtures_2024])
</code></pre>

<p>
  This dataframe has all the matches for 2024, including those that have not been played yet, but I did not need those, so I took them out. I thought it would be easier to read if 
  the datetime was just the date instead, so I adjusted that as well. This is how I did that:
</p>

<pre style="font-size: 12px; padding: 10px; line-height: 1.2;"><code class="language-python">
past_fixtures_df = fixture_df[fixture_df['status'] == 'Match Finished']

past_fixtures_df = past_fixtures_df[['fixture_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score']]

past_fixtures_df['date'] = pd.to_datetime(past_fixtures_df['date'])

past_fixtures_df['date'] = past_fixtures_df['date'].dt.date
</code></pre>

<p>
  The dataframe has the fixture id, date, home team, away team, home score, and away score for every game from 2018-2024. This is a great start, but I wanted more detailed   
  match statistics. In order to do that, I used the fixture/statistics endpoint. The main purpose of creating that first dataframe was to get all the fixture ids in a single 
  place. Now that they are, I can run a for loop that takes each fixture id and gathers the in-depth match statistics for that game. Here is how I did it (it might take a while to 
  run):
</p>

<pre style="font-size: 12px; padding: 10px; line-height: 1.2;"><code class="language-python">
import requests
import pandas as pd
import time

url = "https://v3.football.api-sports.io/fixtures/statistics"

# Initial query parameters: Premier League ID (39), Year, and team ID for Manchester United (33)

querystring = {"league": "39", "season": "2023", "team": "33"}

headers = {
    "x-apisports-key": str(api_key)
}

# Initialize an empty list to hold all fixture statistics
  
match_stats = []

for fixture_id in past_fixtures_df['fixture_id']:
    querystring = {"fixture": str(fixture_id), "team": "33"}

    response = requests.get(url, headers=headers, params=querystring)
    result = response.json()

    statistics = result.get('response', [])

    fixture_statistics = {
        'fixture_id': fixture_id
    }

    for stat in statistics:
        for item in stat['statistics']:
            fixture_statistics[item['type']] = item['value']

    # Append the fixture statistics to the list
  
    match_stats.append(fixture_statistics)

    # Respect rate limits
  
    time.sleep(10)  # Adjust the sleep time as needed to stay within rate limits

# Create a DataFrame from the match statistics list
  
df_match_statistics = pd.DataFrame(match_stats)
  
</code></pre>

<p>
  Once that finished, the final step was to merge the fixture dataframe with the match statistics dataframe. 
</p>

<pre style="font-size: 12px; padding: 10px; line-height: 1.2;"><code class="language-python">
final_df = pd.merge(past_fixtures_df, df_match_statistics, on='fixture_id')
</code></pre>

<p>
  Sensational! Just like that we have our tidy dataframe.
</p>

<div style="text-align:center; margin: 20px;">
  <img src="/images/soccer_df.png" alt="Man United Dataframe" style="width: 900px;"/>
</div>

<h3>Data Summary</h3>

<p>
  The final dataframe has 239 matches and the following 24 features: fixture_id, date, home_team, away_team, home_score, away_score, Shots on Goal, Shots off Goal, Total Shots, 
  Blocked Shots, Shots insidebox, Shots outsidebox, Fouls, Corner Kicks, Offsides, Ball Possession, Yellow Cards, Red Cards, Goalkeeper Saves, Total Passes, Accurate Passes, Pass 
  %, expected_goals, and goals_prevented. 
</p>

<p>
  One thing I was interested in looking at was the average number of goals that Manchester United was scoring and allowing by month across    the years. In order to do that, I 
  split the dataframe into two so that I could create new columns for goals scored and allowed when Manchester United was the home team versus 
  when they were the away team. Once those new columns were created, I rejoined the dataframes, grouped by month and year, and calculated the average number of goals scored and 
  allowed by Manchester United per month. Here is how I did it: 
</p>

<pre style="font-size: 12px; padding: 10px; line-height: 1.2;"><code class="language-python">
# Filter matches where Manchester United is the home team

home_matches = final_df[final_df['home_team'] == 'Manchester United']

# Filter matches where Manchester United is the away team

away_matches = final_df[final_df['away_team'] == 'Manchester United']

# Calculate goals scored by Manchester United as home team

home_matches['man_utd_goals'] = home_matches['home_score']

# Calculate goals scored by Manchester United as away team

away_matches['man_utd_goals'] = away_matches['away_score']

# Combine home and away matches

man_utd_matches = pd.concat([home_matches, away_matches])

# Group by year and month and calculate the average goals per game

goals_per_game_by_month_year = man_utd_matches.groupby(['year', 'month']).agg(
    total_goals=('man_utd_goals', 'sum'),
    games_played=('man_utd_goals', 'count')
).reset_index()

# Calculate average goals per game

goals_per_game_by_month_year['average_goals_per_game'] = goals_per_game_by_month_year['total_goals'] / goals_per_game_by_month_year['games_played']

# Calculate goals allowed by Manchester United as home team

home_matches['goals_allowed'] = home_matches['away_score']

# Calculate goals allowed by Manchester United as away team

away_matches['goals_allowed'] = away_matches['home_score']

# Combine home and away matches

man_utd_matches = pd.concat([home_matches, away_matches])

# Group by year and month and calculate the average goals allowed per game

goals_allowed_per_game_by_month_year = man_utd_matches.groupby(['year', 'month']).agg(
    total_goals_allowed=('goals_allowed', 'sum'),
    games_played=('goals_allowed', 'count')
).reset_index()

# Calculate average goals allowed per game

goals_allowed_per_game_by_month_year['average_goals_allowed_per_game'] = goals_allowed_per_game_by_month_year['total_goals_allowed'] / goals_allowed_per_game_by_month_year['games_played']
</code></pre>

<p>
  The last step was to visualize that data using matplotlib. I created a simple line graph with both averages so that I could understand when Manchester United was performing 
  well, and when they were performly poorly, on both sides of the ball. 
</p>

<div style="text-align:center; margin: 20px;">
  <img src="/images/Goals_by_month_year.png" alt="gitignore" style="width: 750px;"/>
</div>

<p>
  This is just one example of the many things we can look at with our data. With some more analysis, we will be well on our way towards building a model to track the team's 
  form. As you go about creating your own dataset, think about what else you could add and what questions you might ask to derive deeper insights about a team's current trend. 
</p>
<p>
  If you would like to learn more about how the data boom is shaping soccer, I would highly recommend <a href="https://www.amazon.com/Football-Hackers-Science-Data- 
  Revolution/dp/1788702050/ref=asc_df_1788702050?mcid=0778f8cc61f338c8966c747b9489671e&tag=hyprod- 
  20&linkCode=df0&hvadid=693608721823&hvpos=&hvnetw=g&hvrand=15295822945045814355&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9029858&hvtargid=pla-7  
  57604096041&psc=1">Football Hackers: The Science and Art of a Data Revolution</a>. Maybe it could give you some ideas for a future project!
</p>

<p>
  Lastly, if you are interested in my code for this project, it can be found <a href="https://github.com/sballs9/Soccer_Team_Form_Analysis/tree/main">here</a>. I hope this was 
  helpful, and I am sure that with a little practice, you will be generating game-winning insights in no time. Best of luck!
</p>

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
    With this brief introduction, I hope you have started to appreciate the power of XGBoost. I have avoided diving into the math behind the method here, but if you're interested, 
    it’s fascinating to see what’s happening under the hood.
</p>

<p>
    If you are interested, you can find more information <a href="https://www.geeksforgeeks.org/xgboost/">here</a>.
</p>
  
<p>
    Now that you have some basic information, I would encourage you to try it yourself. With a bit of practice, I am confident you will become a   
    master of XGBoost, slaying even the mightiest of Kaggle competitions in no time. 
</p>

</details>






