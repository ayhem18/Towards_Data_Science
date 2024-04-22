import random, os
import pandas as pd
from typing import List
from tqdm import tqdm

random.seed(69) # for reproducibility


class BanditClass:
	def __init__(self, 
			num_vists: int,
			eps: float,
			ratings: pd.DataFrame) -> None:
		self.visits = num_vists
		self.ratings = ratings
		# save a list of unique movies
		self.movies = list(self.ratings['movie_id'].unique())
		# save the number of times each move is encountered
		self.counts = {movie: 0 for movie in self.movies}
		self.rewards = {mv: 0 for mv in self.movies}
		self.eps = eps
		
		# variables to save the current iteration index and current total reward
		self.current_iteration = 0
		self.current_count = 0
		self.relevance_fraction = []

	def _visit(self, user_id: int):
		# extract the movies rated by the user
		user_movies = self.ratings[self.ratings['user_id'] == user_id]['movie_id'].tolist()
		# generat a random number
		rand = random.random()
		selected_movie = None
		if rand < self.eps:
			# select randomly
			selected_movie = random.choice(user_movies)
		else:
			# choose the movie with the highest reward so far
			selected_movie = max(user_movies, key=lambda m: (self.rewards.get(m, 0) / (self.counts[m]) if self.counts[m] != 0 else 0))

		# get whether the user liked the movie or not
		user_movie_rating = self.ratings[(self.ratings['movie_id'] == selected_movie) & (self.ratings['user_id'] == user_id)]['reward'].item()
		self.rewards[selected_movie] += user_movie_rating
		self.counts[selected_movie] += 1

		self.current_iteration += 1
		self.current_count += user_movie_rating
		self.relevance_fraction.append(self.current_count / self.current_iteration)


	def run(self,
			users: List[int], 
			num_iterations: int = 20) -> pd.DataFrame:
		# create a dataframe to save the reuslts
		output_df = pd.DataFrame(data=[], columns = [f'relevance_fraction_{i}' for i in range(1, self.visits + 1)])
		for _ in tqdm(range(1, num_iterations + 1), desc=f'iterations'):
			# first set the current iteration to 0
			self.current_iteration = 0
			self.current_count = 0
			self.relevance_fraction = []
			for _ in range(self.visits):
				# choose a random user
				user = random.choice(users)
				self._visit(user)

			# create a dataframe out of the results of the last visit
			data = {col_name: v for col_name, v in zip([f'relevance_fraction_{i}' for i in range(1, self.visits + 1)], self.relevance_fraction)}

			last_visit_output = pd.DataFrame(data=data, index=[1])
			output_df = pd.concat([output_df, last_visit_output], ignore_index=True)

		return output_df
	

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    rating_df = pd.read_csv(os.path.join(script_dir, 'top-n-movies_user-ratings.csv')).drop(columns = 'Unnamed: 0')
    reward_threshold = 4
    rating_df['reward'] = rating_df.eval('rating > @reward_threshold').astype(int)
    print(rating_df.head())
	
    # create an object
    epsilon = 0.1
    bandit_obj = BanditClass(num_vists=2000, eps=epsilon, ratings=rating_df)
    users_list = list(rating_df['user_id'].unique())
    bandit_obj.run(users=users_list, num_iterations=3)




