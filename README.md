# Project partner: Madhav Karthikeyakannan

# Cloud Jupyter Notebook Video: https://cmu.box.com/s/h74d5b2ro4canlbu11da3g8cra1ja07k

# Demo and Code explanaiton Video: https://cmu.box.com/s/c91egc2sgmf2cnxjeq3dx46b7iyq1a39

# Guide to running the code
1. Open the Project.ipynb
2. FOR TASK 1: Run the Cells from the top. EXCEPT: the 7th cell that has `# WRITING THE DATAFRAME TO GOOGLE CLOUD SQL`  as commented in the beginning. Please do NOT run this, otherwise the Postgres on CloudSQL will get duplicate data.
3. For TASK 2: The code can be run sequentially in the order it is written in, beginning with reading the data frame from the cloud. For each of Q1, Q2, and Q3 in Task 2, there are two code blocks. the code block defining the function should be run first, and the second code block allows for the user to change the values of the input variables (if applicable) and will then run the above function for the input variables, and provide the target output.
4. For TASK 3 - PyTorch for Both MLP and Linear Reg, you can run all the code cells one by one from top to bottom. Every cell should be run in order and sequentially.
5. FOR TASK 3 - SPARK ML for both the Linear and Random Forest Regressors, you can run the code cells one by one from top to bottom, it should be run in the order it is currently in.


#Description of the features in the dataset.

`sofifa_id` : represents the unique ID number for each player assigned from Fifa. type: IntegerType()<br>
`player_url`: represents url link of the player on sofifa.com. type: StringType()<br>
`short_name` : represents the shortened version of the player's name. type: StringType()<br>
`long_name`  : represents the full name of the player's name. type: StringType()<br>
`player_positions` : represents the positions the player can take in a football game. type: StringType()<br>
`overall` : represents overall rating for the player. type: IntegerType()<br>
`potential` : represents a player's ability to develop into a better overall rating. type: IntegerType()<br>
`value_eur` : represents the monetary value of the player. type: IntegerType()<br>
`wage_eur` : represents the salary of the player. type: IntegerType()<br>
`age` : represents the age of the player. type: IntegerType()<br>
`dob` : represents the date of birth of the player. type: DateType()<br>
`height_cm` : represents the height of the player in cm. type IntegerType()<br>
`weight_kg`: represents the weight of the player in kg. type IntegerType()<br>
`club_team_id` : represents the unique ID number of the team that this player plays in. type IntegerType()<br>
`club_name` : represents the club name that this player plays in. type: StringType()<br>
`league_name` : represents the league in which a plery is currently playing. type: StringType()<br>
`league_level` : represents the level of the league. type: IntegerType()<br>
`club_position` : represents which position this player is playing in for the club. type: StringType()<br>
`club_jersey_number`: represents the number of the jersey that the player wears. type: IntegerType()<br>
`club_loaned_from` : represents the club from which the player was loaned from for short term. type: StringType()<br>
`club_joined` : represents the date in which the player has joined the specific club. type: DateType()<br>
`club_contract_valid_until` : represents end of contract date. type: DateType()<br>
`nationality_id` : represents the country of origin in unique country id number. type: IntegerType()<br>
`nationality_name`: represents the country of origin. type:  StringType()<br>
`nation_team_id`: represents which national team the player plays in with unique id number. type: IntegerType()<br>
`nation_position`: represents the position the player takes in the national team. type: StringType()<br>
`nation_jersey_number`: represents the number of the jersey that the player wears in the national team. type: IntegerType()<br>
`preferred_foot` : represents which foor the player prefers to use while in game. type: StringType()<br>
`weak_foot` : represents how well the player can handle with non dominant foot in game. type: IntegerType()<br>
`skill_moves` : represents player's ability to perform skill moves on the field. Players are rated from one to five stars. type: IntegerType()<br>
`international_reputation` : represent how internationally recognized the player is. type: IntegerType()<br>
`work_rate` : represents player's level of activity both in attack and defense during a match. type: StringType()<br>
`body_type` : represents the physical build and appearance of a player. type: StringType()<br>
`real_face` : represents whether player has the true face of the real player. type: BooleanType()<br>
`release_clause_eur` : represents monetary value (in euros) set in a player's contract that allows them to leave their club if another team meets that fee. type :  IntegerType()<br>
`player_tags` : represents hashtag labels assigned to players that describe specific characteristics or styles of play. type: StringType()<br>
`player_traits` : represents specific characteristics or abilities that improbe a player's performance during game. type: StringType()<br>
`pace` : represents player's speed on the pitch. type: IntegerType()<br>
`shooting` : represents player's ability or rating when it comes to shooting the ball. type: IntegerType()<br>
`passing` :  represents player's ability or rating when it comes to passing the ball. type: IntegerType()<br>
`dribbling`: represents player's ability or rating when it comes to maneuvering the ball. type: IntegerType()<br>
`defending`: represents player's ability or rating when it comes to defending against the opponent team's player. type: IntegerType()<br>
`physic`: represents the physical strength of the player. type: IntegerType<br>
`attacking_crossing`: represents player's ability to deliver accurate crosses into penalty area. type: IntegerType<br>
`attacking_finishing` : represents player's ability to turn opportunity into a goal. type: IntegerType()<br>
`attacking_heading_accuracy` : represents player's accuracy in performing header. type: IntegerType()<br>
`attacking_short_passing` : represents player's ability to execute precise short passes during attacking.typ: IntegerType()<br>
`attacking_volleys` :  represents player's ability to execute volleyed shots. type: IntegerType()<br>
`skill_dribbling` : represents player's ability to maneuver the ball while dribbling. type: IntegerType()<br>
`skill_curve` : represents player's ability to execute shots with spin. type: IntegerType()<br>
`skill_fk_accuracy`: represents player's ability to execute free kicks. type: IntegerType()<br>
`skill_long_passing`: represents player's ability to execute long pass. type: IntegerType()<br>
`skill_ball_control`: represents player's ability to maintain ball possession. type: IntegerType()<br>
`movement_acceleration`: represents player's ability to quickly increase speed. type: IntegerType()<br>
`movement_sprint_speed` : represents player's maximum speed when sprinting. type: IntegerType()<br>
`movement_agility` : represents player's ability to change direction quickly while keeping the ball. type: IntegerType()<br>
`movement_reactions` : represents the player's ability to respond quickly to in-game situations. type: IntegerType()<br>
`movement_balance` : represents the player's ability to maintain stability and composure during physical challenges. type: IntegerType()<br>
`power_shot_power` : represents the player's strength in shooting the ball. type: IntegerType()<br>
`power_jumping` : represents the player's ability to jump. type: IntegerType()<br>
`power_stamina` : represents the player's endurance. type: IntegerType()<br>
`power_strength` : represents the physical strength of the player. type: IntegerType()<br>
`power_long_shots` : represents the player's ability to score from long distances. type: IntegerType()<br>
`mentality_aggression` : represents the player's aggression level during play. type: IntegerType()<br>
`mentality_interceptions` : represents the player's ability to read the game and intercept passes. type: IntegerType()<br>
`mentality_positioning` : represents the player's ability of positioning. type: IntegerType()<br>
`mentality_vision` : represents the player's ability to see plays. type: IntegerType()<br>
`mentality_penalties` : represents the player's ability in taking penalties. type: IntegerType()<br>
`mentality_composure` : represents the player's composure. type: IntegerType()<br>
`defending_marking_awareness` : represents the player's ability to mark opponents. type: IntegerType()<br>
`defending_standing_tackle` : represents the player's ability to performing standing tackles. type: IntegerType()<br>
`defending_sliding_tackle` : represents the player's ability to perform sliding tackles. type: IntegerType()<br>
`goalkeeping_diving` : represents the goalkeeper's ability to dive and reach for shots on goal. type: IntegerType()<br>
`goalkeeping_handling` : represents the goalkeeper's ability to catching and holding onto the ball. type: IntegerType()<br>
`goalkeeping_kicking` : represents the goalkeeper's ability to kick the ball accurately. type: IntegerType()<br>
`goalkeeping_positioning` : represents the goalkeeper's ability to positioning. type: IntegerType()<br>
`goalkeeping_reflexes` : represents the goalkeeper's quickness. type: IntegerType()<br>
`goalkeeping_speed` : represents the goalkeeper's speed when rushing out to challenge attackers. type: IntegerType()<br>
`ls` : represents the player's ability in the left striker position. type: IntegerType()<br>
`st` : represents the player's ability in the central striker position. type: IntegerType()<br>
`rs` : represents the player's ability in the right striker position. type: IntegerType()<br>
`lw` : represents the player's ability in the left winger position. type: IntegerType()<br>
`lf` : represents the player's ability in the left forward position. type: IntegerType()<br>
`cf` : represents the player's ability in the center forward position. type: IntegerType()<br>
`rf` : represents the player's ability in the right forward position. type: IntegerType()<br>
`rw` : represents the player's ability in the right winger position. type: IntegerType()<br>
`lam` : represents the player's ability in the left attacking midfielder position. type: IntegerType()<br>
`cam` : represents the player's ability in the central attacking midfielder position. type: IntegerType()<br>
`ram` : represents the player's ability in the right attacking midfielder position. type: IntegerType()<br>
`lm` : represents the player's ability in the left midfielder position. type: IntegerType()<br>
`lcm` : represents the player's ability in the left central midfielder position. type: IntegerType()<br>
`cm` : represents the player's ability in the central midfielder position. type: IntegerType()<br>
`rcm` : represents the player's ability in the right central midfielder position. type: IntegerType()<br>
`rm` : represents the player's ability in the right midfielder position. type: IntegerType()<br>
`lwb` : represents the player's ability in the left wing-back position. type: IntegerType()<br>
`ldm` : represents the player's ability in the left defensive midfielder position. type: IntegerType()<br>
`cdm` : represents the player's ability in the central defensive midfielder position. type: IntegerType()<br>
`rdm` : represents the player's ability in the right defensive midfielder position. type: IntegerType()<br>
`rwb` : represents the player's ability in the right wing-back position. type: IntegerType()<br>
`lb` : represents the player's ability in the left back position. type: IntegerType()<br>
`lcb` : represents the player's ability in the left center back position. type: IntegerType()<br>
`cb` : represents the player's ability in the center back position. type: IntegerType()<br>
`rcb` : represents the player's ability in the right center back position. type: IntegerType()<br>
`rb` : represents the player's ability in the right back position. type: IntegerType()<br>
`gk` : represents the player's ability in the goalkeeper position. type: IntegerType()<br>
`player_face_url` : represents the URL link to the player's real-life face model. type: StringType()<br>
`club_logo_url` : represents the URL link to the club's logo. type: StringType()<br>
`club_flag_url` : represents the URL link to the club's flag. type: StringType()<br>
`nation_logo_url` : represents the URL link to the nation's logo. type: StringType()<br>
`nation_flag_url` : represents the URL link to the nation's flag. type: StringType()<br>


# The benefit of using PostgreSQL DB table compared to a NoSQL Database

PostgreSQL is the better choice in this case because it enforces a clear structure and schema, which is important for working with the highly organized and rigid nature of FIFA data. PostgreSQL also offers good support for data analytics, especially when paired with Spark, making it easy to run complex queries and gain insights from the data.

In contrast, NoSQL databases excel with unstructured or semi-structured data. However, they are less suited for handling complex relational queries or ensuring strict data consistency, which are crucial for structured datasets like FIFA. NoSQL databases are appropriate when the data can be denormalized and schema-less, with a simpler structure, a condition that doesn’t apply to FIFA data.

Finally, since FIFA data does not require terabytes of storage or deal with massive-scale operations, the scalability benefits of NoSQL are unnecessary. PostgreSQL, with its support for structured and relational data, is the better solution in this case.

# In your ReadMe file, explain why you chose the classifiers/regressors and provide comments on the impact of the tunable parameters on the accuracy. Also, compare the selected models.

FOR PYTORCH:
We have decided to run MultiLayer Perceptron and Linear regression for PyTorch. 

MultiLayer Perceptron was chosen because fifa dataset is quite complex and has many different features that is relevant to the abilities of the football player. Multilayered approach should be able to learn all the nuance, relationship, and complexities in estimating the Overall rating of a player. It is also likely that these features and target have non-linear relationship, requiring non-linear models such as MLP. This type of model also has great flexibility when it comes to hypertuning. Selection of number of layers to activation function gives the coder much more option than Linear regression model. 

For MLP: With shallow layer depth, the R2 and Loss were quite high, indicating that model is not learning well from the data. After adding about 5-6 layers, the model significantly performed better, with each epoch progressively increasin the R2 and decreasing the loss. Dropout of neurons was experimented on the data. However, it hasn't improved the performance on the validation set that much. Hence dropout was not implemented. 

When it comes to Learning rate and Batch size, different values were tried. In the code hyperparameter grid was built to try different values. In general, when lr = 0.1 or higher, model was overshooting and was not stabilizing. Also with SGD, model convergence was very difficult. Thus ADAM was chosen which can introduce much more smoother descent toward local minima than SGD. With lr greater than 0.1, and ADAM as optimizer, convergence became stable and model was learning with decent speed. 

As for Linear regression, there is a possibility that the data points have linear relationship with the target (Overall Rating). If this is the case, we would be able to get a decent Regression model that can predict the Overall rating with decent Loss and R^2 with less computation cost. Linear regression model is foundational yet widely used and can be way cheaper computationally than MLP as it doesn't train multiple layers of functions and perform backpropogation. Linear regression can also have less tendency to overfit when compared to MLP. With many deep layers, MLP is likely to overfit the data. This model is also simple, in that it has fewer parameters to tune and can be setup and run much faster than MLP can. 

For Linear Reg: Since it is simpler model, ADAM was also chosen due to similar reasons with MLP. Similar range of learning rate as well as batch size was tested using hyperparameter grid. 

Overall, MLP resulted R2 ≥ 0.9 which means it is capturing 90% of the variance of the testing data which is a good result. The linear reg had R2 ≥ 0.8 which is also decent. When compared to Random Forest regressor which is also in similar range of performance, it is preferred to use in general, Random Forest Regressor, considering the amount of code that needs to be written, hypertuning complexity, and computational cost which is quite high with MLP. 





[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/VuODydzp)
