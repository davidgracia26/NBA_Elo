import munge
import forecast


games_all_time_df = munge.retrieve_games_data('NBA_History.csv')

ratings_df, games_all_time_df = munge.elo_algorithm(games_all_time_df)

games_all_time_df = munge.add_days_rest_columns(games_all_time_df)

games_all_time_df = munge.add_win_streak_columns(games_all_time_df)
games_all_time_df = munge.add_ppg_columns(games_all_time_df)

rating_adjustment = munge.calculate_rating_adjustment()

upcoming_games = forecast.determine_best_upcoming_games(ratings_df, ratings_df['Ratings'].tolist(),
                                                        ratings_df['Teams'].tolist(), rating_adjustment)

sim_df = forecast.simulate_season(ratings_df, ratings_df['Teams'].tolist(), 100, 2018)

games_all_time_df['Ratings_Diff'] = (games_all_time_df['Visitor_Rating'] - games_all_time_df['Home_Rating']).abs()



