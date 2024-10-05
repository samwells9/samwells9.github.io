---
permalink: /blog/
title: "Latest"
author_profile: true
---

<details>
<summary style="cursor: pointer; font-weight: bold;">Kaggle Slayer: Introduction to XGBoost</summary>

<div style="text-align:center;">
  <img src="/images/dragon_slayer.png" alt="Dragon Slayer" class="float-right" style="max-width: 600px; height: auto;"/>
</div>

<p>
  Every Kaggle competition presents unique obstacles—requiring the sharpest of skills and the most powerful of tools. In this post, I will introduce you to one of the greatest weapons, which every data scientist should have in their arsenal: <span 
  style="color:#007BFF;">XGBoost</span>.
</p>

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

<h3>Why XGBoost?</h3>

<p>
  So what is it that makes XGBoost reign king when it comes to working with tabular data? For the purpose of this post, I have narrowed it down to four key factors:
</p>

<ul>
  <li>
      Efficiency: XGBoost is extremely fast. It uses parallel processing and optimization techniques, allowing it to construct different parts of the trees simultaneously. This significantly expedites the process when compared traditional gradient boosting         methods, which build each tree one by one. It is also sparse-aware, which just means it only stores non-zero values in order to conserve memory. The method's handling of the data enables to excel with large datasets, making it a great option for   
      applications in big data.
  </li>
  <li>
      Flexibility: XGBoost is highly adaptable, offering built-in functions for handling missing values and responding well to outliers. It includes tons of tuning parameters, which allows users to customize the model depending on the task and dataset. By          creating a simple parameter grid, users can quickly test various combinations of hyperparameters to identify the optimal configuration for their problem.
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

<p>
  XGBoost has been successfully applied to:
</p>

<ul>
    <li>
        Sports:
    </li>
    <li>
        Business:
    </li>
    <li>
        Health:
    </li>
</ul>
</details>






