from scipy.stats import zscore
import datetime
import pandas as pd
import numpy
import munge
import math


def determine_best_upcoming_games(ratings_df, team_ratings, teams, rating_adjustment):
    team_z_scores = zscore(ratings_df['Ratings']).tolist()
    final_df_teams = ratings_df['Teams'].tolist()
    current_season_games = pd.read_csv(filepath_or_buffer='{}_Games.csv'.format(2018))

    current_season_games['Date'] = pd.to_datetime(current_season_games['Date'], format='%a %b %d %Y')
    start_date = datetime.datetime.today()
    end_date = start_date + datetime.timedelta(days=7)

    # if start_date < current_season_games['Date'].iloc[0]:
    #     start_date = current_season_games['Date'].iloc[0]
    #     end_date = start_date + datetime.timedelta(days=7)
    #     ratings_df['Ratings'] = ratings_df['Ratings'] * 0.75 + 1505 * 0.25
    #     ratings_df['Game_Index'] = 0
    #     ratings_df['Wins'] = 0
    #     ratings_df['Losses'] = 0
    #     ratings_df['Rating_Delta'] = 0
    #     ratings_df['Season_Init_Rating'] = ratings_df['Ratings']

    # fix date comparison, it excludes today for some reason
    upcoming_games_mask = (current_season_games['Date'] >= start_date) & (current_season_games['Date'] <= end_date)
    upcoming_games = current_season_games[upcoming_games_mask]

    mean_final_df_rating = numpy.mean(ratings_df['Ratings'])
    sd_final_df_rating = numpy.std(ratings_df['Ratings'])
    missing_team_z_score = (1300 - mean_final_df_rating) / sd_final_df_rating
    upcoming_games_count = len(upcoming_games)

    upcoming_games_rating_diffs = munge.create_n_length_list(0, upcoming_games_count)
    upcoming_games_home_win_chances = munge.create_n_length_list(0, upcoming_games_count)
    upcoming_games_exp_home_spreads = munge.create_n_length_list(0, upcoming_games_count)
    upcoming_games_sim_score = munge.create_n_length_list(0, upcoming_games_count)
    upcoming_games_z_sums = munge.create_n_length_list(0, upcoming_games_count)
    upcoming_games_weight = munge.create_n_length_list(0, upcoming_games_count)

    row_index_counter = 0
    for index, row in upcoming_games.iterrows():

        home = row['Home']
        visitor = row['Visitor']

        home_z_score_rating = 0
        visitor_z_score_rating = 0

        home_team_rating = team_ratings[teams.index(home)]
        visitor_team_rating = team_ratings[teams.index(visitor)]

        if home_team_rating == 0:
            home_team_rating = 1300 + rating_adjustment
        else:
            home_team_rating += rating_adjustment

        if visitor_team_rating == 0:
            visitor_team_rating = 1300

        home_z_score_rating = (home_team_rating - mean_final_df_rating) / sd_final_df_rating

        if team_z_scores[final_df_teams.index(visitor)]:
            visitor_z_score_rating = team_z_scores[final_df_teams.index(visitor)]
        else:
            visitor_z_score_rating = missing_team_z_score

        upcoming_games_rating_diffs[row_index_counter] = home_team_rating - visitor_team_rating
        upcoming_games_home_win_chances[row_index_counter] = 1 / (
                    1 + 10 ** (-upcoming_games_rating_diffs[row_index_counter] / 400))
        upcoming_games_exp_home_spreads[row_index_counter] = -upcoming_games_rating_diffs[row_index_counter] / 25
        upcoming_games_sim_score[row_index_counter] = 1 / (1 + abs(visitor_z_score_rating - home_z_score_rating))
        upcoming_games_z_sums[row_index_counter] = visitor_z_score_rating + home_z_score_rating
        upcoming_games_weight[row_index_counter] = upcoming_games_sim_score[row_index_counter] * upcoming_games_z_sums[
            row_index_counter]

        row_index_counter += 1

    upcoming_games_data = {'Date': upcoming_games['Date'],
                           'Visitor': upcoming_games['Visitor'],
                           'Home': upcoming_games['Home'],
                           'Home_Win_Prob': upcoming_games_home_win_chances,
                           'Home_Spread': upcoming_games_exp_home_spreads,
                           'Sim_Score': upcoming_games_sim_score,
                           'Z_Sums': upcoming_games_z_sums,
                           'Weight': upcoming_games_weight}

    upcoming_games = pd.DataFrame.from_dict(upcoming_games_data).sort_values(by='Weight', ascending=False)
    upcoming_games['Home_Spread'] = round(upcoming_games['Home_Spread'] / 0.5) * 0.5

    return upcoming_games


def simulate_season(ratings_df, teams_list, num_sims, current_season):
    # Predict Record for team record
    season_sim_team_ratings = ratings_df['Ratings'].tolist()
    season_sim_wins = ratings_df['Wins'].tolist()
    season_sim_init_ratings = ratings_df['Ratings'].tolist()
    k_factor = 20
    rating_adjustment = munge.calculate_rating_adjustment()
    start_date = datetime.datetime.today()
    current_season_games = pd.read_csv(filepath_or_buffer='{}_Games.csv'.format(current_season))
    current_season_games['Date'] = pd.to_datetime(current_season_games['Date'], format='%a %b %d %Y')

    seasons_sim_wins = [0] * len(ratings_df)
    seasons_sim_team_ratings = [0] * len(ratings_df)
    # only uncomment to debug
    # numpy.random.seed(42)
    games_to_sim_mask = (current_season_games['Date'] >= start_date)
    games_to_sim = current_season_games[games_to_sim_mask]
    number_of_iterations = num_sims
    for x in range(number_of_iterations):
        for index, row in games_to_sim.iterrows():
            home_team = row['Home']
            home_team_index = teams_list.index(home_team)
            home_team_init_rating = season_sim_team_ratings[home_team_index]

            visitor_team = row['Visitor']
            visitor_team_index = teams_list.index(visitor_team)
            visitor_team_init_rating = season_sim_team_ratings[visitor_team_index]

            adj_visitor_team_rating = visitor_team_init_rating
            adj_home_team_rating = home_team_init_rating

            adj_home_team_rating = home_team_init_rating + rating_adjustment

            home_relative_rating_diff = adj_home_team_rating - adj_visitor_team_rating

            e_home = munge.calculate_e_value(-home_relative_rating_diff)
            e_visitor = munge.calculate_e_value(home_relative_rating_diff)

            e_home_as_int = math.floor(e_home * 100)

            random_int = numpy.random.randint(0, 100)
            is_home_team_winner = False

            if random_int <= e_home_as_int:
                is_home_team_winner = True
            else:
                is_home_team_winner = False
            # now compare e_home_as_int to random_int
            # fix this to determine the random number
            if is_home_team_winner is True:
                s_home = 1
                s_visitor = 0
                season_sim_wins[home_team_index] += 1
            else:
                s_home = 0
                s_visitor = 1
                season_sim_wins[visitor_team_index] += 1

            home_rating = munge.calculate_rating(home_team_init_rating, 1, k_factor, s_home, e_home)
            visitor_rating = munge.calculate_rating(visitor_team_init_rating, 1, k_factor, s_visitor, e_visitor)

            season_sim_team_ratings[home_team_index] = home_rating
            season_sim_team_ratings[visitor_team_index] = visitor_rating
        # add final rating to list
        season_sim_team_ratings = ratings_df['Ratings'].tolist()

    sim_df_data = {'Teams': teams_list, 'Wins': season_sim_wins}
    sim_df = pd.DataFrame.from_dict(sim_df_data).sort_values(by='Wins', ascending=False)
    sim_df['Wins'] = sim_df['Wins'] / number_of_iterations

    return sim_df
    # needs to for each game:
    # use a dictionary to store team final rating
    # use a dictionary to store final games won
    # repeat

    # addendum
    # start the sim from the current date and add wins to the total already accrued

    # later work on adding playoff simulation capability