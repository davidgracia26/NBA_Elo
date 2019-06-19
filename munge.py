import pandas as pd
import math
import numpy as np


def replace_deprecated_teams(df, replacement_list):
    for x in range(len(replacement_list)):
        df = df.replace(replacement_list[x])
    return df


def calculate_rating(init_rating, elo_multiplier, k_factor, s_value, e_value):
    return init_rating + elo_multiplier * k_factor * (s_value - e_value)


def calculate_elo_multiplier(adj_home_team_rating, adj_visitor_team_rating, visitor_score, home_score, abs_mov):
    if adj_home_team_rating > adj_visitor_team_rating and visitor_score > home_score:
        return (abs_mov + 3) ** 0.8 / (7.5 + 0.006 * (adj_visitor_team_rating - adj_home_team_rating))
    elif adj_home_team_rating < adj_visitor_team_rating and visitor_score < home_score:
        return (abs_mov + 3) ** 0.8 / (7.5 + 0.006 * (adj_home_team_rating - adj_visitor_team_rating))
    elif adj_home_team_rating > adj_visitor_team_rating and visitor_score < home_score:
        return (abs_mov + 3) ** 0.8 / (7.5 + 0.006 * (adj_home_team_rating - adj_visitor_team_rating))
    elif adj_home_team_rating < adj_visitor_team_rating and visitor_score > home_score:
        return (abs_mov + 3) ** 0.8 / (7.5 + 0.006 * (adj_visitor_team_rating - adj_home_team_rating))


def calculate_e_value(rating_diff):
    return 1 / (1 + 10 ** (rating_diff / 400))


def determine_s_tuple(home_score, visitor_score):
    # produce a tuple with s_home as first value
    # and s_visitor as second value
    if home_score > visitor_score:
        return 1, 0
    elif home_score < visitor_score:
        return 0, 1
    else:
        return 0.5, 0.5


# visitor_team_init_rating = visitor_team_init_rating * 0.75 + 1505 * 0.25
# season_init_ratings[visitor_team_index] = visitor_team_init_rating
# latest_years[visitor_team_index] = season
def adjust_rating_and_year(team_init_rating, team_index, season_init_ratings, latest_years, season):
    team_init_rating = team_init_rating * 0.75 + 1505 * 0.25
    season_init_ratings[team_index] = team_init_rating
    latest_years[team_index] = season
    return team_init_rating, season_init_ratings, latest_years


def increment_wins_and_losses(wins, losses, winner_team_index, loser_team_index):
    wins[winner_team_index] += 1
    losses[loser_team_index] += 1
    return wins, losses


def increment_ties(ties, visitor_team_index, home_team_index):
    ties[home_team_index] += 1
    ties[visitor_team_index] += 1
    return ties


def retrieve_games_data(file):
    """
    Create a dataframe filled with NBA games data for all time
    :param file: filepath of data to retrieve
    :return: dataframe with all NBA games ever
    """

    df = pd.read_csv(filepath_or_buffer=file)
    df['Notes'] = df['Notes'].astype('str')
    replacements = [
        {'New Jersey Nets': 'Brooklyn Nets', 'New York Nets': 'Brooklyn Nets', 'New Jersey Americans': 'Brooklyn Nets'},
        {'Syracuse Nationals': 'Philadelphia 76ers'},
        {'Fort Wayne Pistons': 'Detroit Pistons'},
        {'St. Louis Hawks': 'Atlanta Hawks', 'Milwaukee Hawks': 'Atlanta Hawks',
         'Tri-Cities Blackhawks': 'Atlanta Hawks'},
        {'Charlotte Bobcats': 'Charlotte Hornets'},
        {'Washington Bullets': 'Washington Wizards', 'Capital Bullets': 'Washington Wizards',
         'Baltimore Bullets': 'Washington Wizards', 'Chicago Zephyrs': 'Washington Wizards',
         'Chicago Packers': 'Washington Wizards'},
        {'San Francisco Warriors': 'Golden State Warriors', 'Philadelphia Warriors': 'Golden State Warriors'},
        {'San Diego Clippers': 'Los Angeles Clippers', 'Buffalo Braves': 'Los Angeles Clippers'},
        {'Minneapolis Lakers': 'Los Angeles Lakers'},
        {'Kansas City Kings': 'Sacramento Kings', 'Kansas City-Omaha Kings': 'Sacramento Kings',
         'Cincinnati Royals': 'Sacramento Kings', 'Rochester Royals': 'Sacramento Kings'},
        {'San Diego Rockets': 'Houston Rockets'},
        {'Vancouver Grizzlies': 'Memphis Grizzlies'},
        {'New Orleans/Oklahoma City Hornets': 'New Orleans Pelicans', 'New Orleans Hornets': 'New Orleans Pelicans'},
        {'Texas Chaparrals': 'San Antonio Spurs', 'Dallas Chaparrals': 'San Antonio Spurs'},
        {'Denver Rockets': 'Denver Nuggets'},
        {'Seattle SuperSonics': 'Oklahoma City Thunder'},
        {'New Orleans Jazz': 'Utah Jazz'},
        {'Los Angeles Stars': 'Utah Stars', 'Anaheim Amigos': 'Utah Stars'},
        {'Carolina Cougars': 'Spirits of St. Louis', 'Houston Mavericks': 'Spirits of St. Louis'},
        {'Memphis Tams': 'Memphis Sounds', 'Memphis Pros': 'Memphis Sounds',
         'New Orleans Buccaneers': 'Memphis Sounds'},
        {'Miami Floridians': 'The Floridians', 'Minnesota Muskies': 'The Floridians'},
        {'Pittsburgh Pipers': 'Pittsburgh Condors', 'Minnesota Pipers': 'Pittsburgh Condors'},
        {'Washington Capitols': 'Virginia Squires', 'Oakland Oaks': 'Virginia Squires'},
        {'San Diego Conquistadors': 'San Diego Sails'}]

    df = replace_deprecated_teams(df, replacements)
    df['Date'] = pd.to_datetime(df['Date'], format='%a %b %d %Y')

    return df


def create_n_length_list(value, length):
    """
    Create a list of specified length filled with a single value
    :param value: value to fill the list
    :param length: length of the list
    :return: a list of length occupied by the specified value
    """
    return [value] * length


def create_list_of_unique_teams(df):
    """
    Create a list of unique teams
    :param df: dataframe
    :return: a list of unique team names from the specified columns
    """
    home_col_teams = df['Home'].tolist()
    visitor_col_teams = df['Visitor'].tolist()

    return list(set(home_col_teams + visitor_col_teams))


def calculate_rating_adjustment():
    return -400 * math.log10(1 / 0.638 - 1)


def elo_algorithm(df):
    number_of_games = len(df)
    current_season = df['Season'].max()

    teams = create_list_of_unique_teams(df)
    team_count = len(teams)

    # Create lists for df columns
    team_ratings = create_n_length_list(1300, team_count)
    games_played_counts = create_n_length_list(0, team_count)
    current_season_game_counts = create_n_length_list(0, team_count)
    latest_years = create_n_length_list(1946, team_count)
    wins = create_n_length_list(0, team_count)
    losses = create_n_length_list(0, team_count)
    ties = create_n_length_list(0, team_count)
    season_init_ratings = create_n_length_list(0, team_count)
    rating_changes = create_n_length_list(0, team_count)
    game_visitor_ratings = create_n_length_list(0, number_of_games)
    game_home_ratings = create_n_length_list(0, number_of_games)
    game_model_corrects = create_n_length_list(False, number_of_games)

    k_factor = 20
    rating_adjustment = calculate_rating_adjustment()

    # Run Elo Algorithm
    for index, row in df.iterrows():
        home_team = row['Home']
        home_team_index = teams.index(home_team)
        home_team_init_rating = team_ratings[home_team_index]
        home_team_game_count = games_played_counts[home_team_index]
        home_team_latest_year = latest_years[home_team_index]

        visitor_team = row['Visitor']
        visitor_team_index = teams.index(visitor_team)
        visitor_team_init_rating = team_ratings[visitor_team_index]
        visitor_team_game_count = games_played_counts[visitor_team_index]
        visitor_team_latest_year = latest_years[visitor_team_index]

        season = row['Season']

        if home_team_game_count == 0 and home_team_latest_year != season:
            latest_years[home_team_index] = season
        elif home_team_game_count > 0 and home_team_latest_year != season:
            home_team_init_rating, season_init_ratings, latest_years = adjust_rating_and_year(home_team_init_rating,
                                                                                              home_team_index,
                                                                                              season_init_ratings,
                                                                                              latest_years, season)

        if visitor_team_game_count == 0 and visitor_team_latest_year != season:
            latest_years[visitor_team_index] = season
        elif visitor_team_game_count > 0 and visitor_team_latest_year != season:
            visitor_team_init_rating, season_init_ratings, latest_years = adjust_rating_and_year(
                visitor_team_init_rating, visitor_team_index, season_init_ratings, latest_years, season)

        home_score = row['Home_PTS']
        visitor_score = row['Visitor_PTS']

        adj_visitor_team_rating = visitor_team_init_rating
        adj_home_team_rating = home_team_init_rating

        if row['Notes'] == 'nan':
            adj_home_team_rating = home_team_init_rating + rating_adjustment

        abs_mov = math.fabs(home_score - visitor_score)
        home_relative_rating_diff = adj_home_team_rating - adj_visitor_team_rating

        e_home = calculate_e_value(-home_relative_rating_diff)
        e_visitor = calculate_e_value(home_relative_rating_diff)

        s_home, s_visitor = determine_s_tuple(home_score, visitor_score)

        elo_multiplier = calculate_elo_multiplier(adj_home_team_rating, adj_visitor_team_rating, visitor_score,
                                                  home_score, abs_mov)

        game_visitor_ratings[index] = adj_visitor_team_rating
        game_home_ratings[index] = adj_home_team_rating

        if adj_home_team_rating > adj_visitor_team_rating and home_score > visitor_score:
            game_model_corrects[index] = True
        elif adj_home_team_rating < adj_visitor_team_rating and home_score < visitor_score:
            game_model_corrects[index] = True

        new_home_rating = calculate_rating(home_team_init_rating, elo_multiplier, k_factor, s_home, e_home)
        new_visitor_rating = calculate_rating(visitor_team_init_rating, elo_multiplier, k_factor, s_visitor, e_visitor)

        if season == current_season:
            if home_score > visitor_score:
                wins, losses = increment_wins_and_losses(wins, losses, home_team_index, visitor_team_index)
            elif home_score < visitor_score:
                wins, losses = increment_wins_and_losses(wins, losses, visitor_team_index, home_team_index)
            elif home_score == visitor_score:
                ties = increment_ties(ties, visitor_team_index, home_team_index)

        team_ratings[home_team_index] = new_home_rating
        team_ratings[visitor_team_index] = new_visitor_rating

        games_played_counts[home_team_index] += 1
        games_played_counts[visitor_team_index] += 1

        if season == current_season:
            current_season_game_counts[home_team_index] += 1
            current_season_game_counts[visitor_team_index] += 1

    # Add these columns to the original dataframe
    df['Visitor_Rating'] = pd.Series(game_visitor_ratings)
    df['Home_Rating'] = pd.Series(game_home_ratings)
    df['Model_Correct'] = pd.Series(game_model_corrects)

    data = {'Teams': teams,
            'Game_Index': current_season_game_counts,
            'Ratings': team_ratings,
            'Wins': wins,
            'Losses': losses,
            'Rating_Delta': rating_changes,
            'Season_Init_Rating': season_init_ratings,
            'Latest_Year': latest_years}

    results_df = pd.DataFrame.from_dict(data).sort_values(by='Ratings', ascending=False)
    results_df['Rating_Delta'] = results_df['Ratings'] - results_df['Season_Init_Rating']

    mask = results_df['Latest_Year'] == current_season

    results_df = results_df[mask]
    results_df = results_df.drop(columns=['Latest_Year'])

    return results_df, df


def date_diff(date1, date2):
    return abs(date2 - date1).days


def create_initial_dates_dict(subset_df, teams_list):
    teams_count = len(teams_list)
    team_date_dict = {}

    for x in range(teams_count):
        team = teams_list[x]

        mask = (subset_df['Visitor'] == team) | (subset_df['Home'] == team)
        team_games_df = subset_df[mask]
        initial_team_date = team_games_df.iloc[0].at['Date']
        team_date_dict[team] = initial_team_date

    return team_date_dict


def create_days_rest_lists_tuple(subset_df, team_date_dict):
    visitor_day_rest_list = create_n_length_list(0, len(subset_df))
    home_day_rest_list = create_n_length_list(0, len(subset_df))

    for index, row in subset_df.iterrows():
        visitor = row['Visitor']
        home = row['Home']
        game_date = row['Date']

        visitor_last_date_played = team_date_dict[visitor]
        home_last_date_played = team_date_dict[home]

        visitor_days_rest = date_diff(game_date, visitor_last_date_played)
        home_days_rest = date_diff(game_date, home_last_date_played)

        visitor_day_rest_list[index] = visitor_days_rest
        home_day_rest_list[index] = home_days_rest

        team_date_dict[visitor] = game_date
        team_date_dict[home] = game_date

    return visitor_day_rest_list, home_day_rest_list


def create_cat_code_columns(df, col_names):
    for x in range(len(col_names)):
        col_name = col_names[x]

        new_col_name = col_name + '_Cat'
        df[new_col_name] = df[col_name].astype('category').cat.codes

    return df


def create_date_columns(df, col_names):
    for x in range(len(col_names)):
        col_name = col_names[x]

        year_col_name = col_name + '_Year'
        month_col_name = col_name + '_Month'
        day_col_name = col_name + '_Day'
        day_week_col_name = col_name + '_Day_Of_Week'

        df[year_col_name] = df[col_name].dt.year
        df[month_col_name] = df[col_name].dt.month
        df[day_col_name] = df[col_name].dt.day
        df[day_week_col_name] = df[col_name].dt.dayofweek

    return df


def add_days_rest_columns(games_df):
    subsetting_col_names = ['Date', 'Visitor', 'Home']
    subset_df = games_df[subsetting_col_names].copy()
    subset_df['Date'] = pd.to_datetime(subset_df['Date'], format='%a %b %d %Y')

    teams_list = create_list_of_unique_teams(subset_df)

    team_date_dict = create_initial_dates_dict(subset_df, teams_list)
    visitor_day_rest_list, home_day_rest_list = create_days_rest_lists_tuple(subset_df, team_date_dict)

    games_df['Visitor_Days_Rest'] = pd.Series(visitor_day_rest_list)
    games_df['Home_Days_Rest'] = pd.Series(home_day_rest_list)

    return games_df


def add_win_streak_columns(games_df):
    subsetting_col_names = ['Visitor', 'Visitor_PTS', 'Home', 'Home_PTS', 'Season']
    subset_df = games_df[subsetting_col_names].copy()

    teams_list = create_list_of_unique_teams(subset_df)

    visitor_streak_list, home_streak_list = create_streak_lists_tuple(subset_df, teams_list)

    games_df['Visitor_Streak'] = pd.Series(visitor_streak_list)
    games_df['Home_Streak'] = pd.Series(home_streak_list)

    return games_df


def create_initial_value_dict(teams_list, initial_value):
    teams_count = len(teams_list)
    team_streak_dict = {}

    for x in range(teams_count):
        team = teams_list[x]

        team_streak_dict[team] = initial_value

    return team_streak_dict


def create_team_season_dict(subset_df, teams_list):
    teams_count = len(teams_list)
    team_season_dict = {}

    for x in range(teams_count):
        team = teams_list[x]
        initial_year = subset_df[(subset_df['Visitor'] == team) | (subset_df['Home'] == team)].iloc[0].Season
        team_season_dict[team] = initial_year

    return team_season_dict


def create_streak_lists_tuple(subset_df, teams_list):
    team_streak_dict = create_initial_value_dict(teams_list, 0)
    team_season_dict = create_team_season_dict(subset_df, teams_list)
    visitor_streak_list = create_n_length_list(0, len(subset_df))
    home_streak_list = create_n_length_list(0, len(subset_df))

    for index, row in subset_df.iterrows():
        visitor = row['Visitor']
        home = row['Home']

        if team_season_dict[visitor] != row['Season']:
            team_streak_dict[visitor] = 0
            team_season_dict[visitor] = row['Season']
        if team_season_dict[home] != row['Season']:
            team_streak_dict[home] = 0
            team_season_dict[home] = row['Season']

        visitor_streak = team_streak_dict[visitor]
        home_streak = team_streak_dict[home]

        visitor_streak_list[index] = visitor_streak
        home_streak_list[index] = home_streak

        is_home_winner = row['Home_PTS'] > row['Visitor_PTS']
        # Home Logic
        if is_home_winner and team_streak_dict[home] >= 0:
            team_streak_dict[home] += 1
        elif is_home_winner and team_streak_dict[home] < 0:
            team_streak_dict[home] = 1

        if is_home_winner is False and team_streak_dict[home] <= 0:
            team_streak_dict[home] -= 1
        elif is_home_winner is False and team_streak_dict[home] > 0:
            team_streak_dict[home] = -1
        # End Home Logic

        # Start Visitor Logic
        if is_home_winner and team_streak_dict[visitor] >= 0:
            team_streak_dict[visitor] = -1
        elif is_home_winner and team_streak_dict[visitor] < 0:
            team_streak_dict[visitor] -= 1

        if is_home_winner is False and team_streak_dict[visitor] <= 0:
            team_streak_dict[visitor] = 1
        elif is_home_winner is False and team_streak_dict[visitor] > 0:
            team_streak_dict[visitor] += 1
        # End Visitor Logic

    return visitor_streak_list, home_streak_list


def add_ppg_columns(games_df):
    subsetting_col_names = ['Visitor', 'Visitor_PTS', 'Home', 'Home_PTS', 'Season']
    subset_df = games_df[subsetting_col_names].copy()

    teams_list = create_list_of_unique_teams(subset_df)

    visitor_ppg_for_list, home_ppg_for_list = create_ppg_for_lists_tuple(subset_df, teams_list)
    visitor_ppg_against_list, home_ppg_against_list = create_ppg_against_lists_tuple(subset_df, teams_list)

    games_df['Visitor_PPG_For'] = pd.Series(visitor_ppg_for_list)
    games_df['Visitor_PPG_Against'] = pd.Series(visitor_ppg_against_list)
    games_df['Visitor_PPG_Diff'] = games_df['Visitor_PPG_For'] - games_df['Visitor_PPG_Against']
    games_df['Home_PPG_For'] = pd.Series(home_ppg_for_list)
    games_df['Home_PPG_Against'] = pd.Series(home_ppg_against_list)
    games_df['Home_PPG_Diff'] = games_df['Home_PPG_For'] - games_df['Home_PPG_Against']

    return games_df


def create_ppg_for_lists_tuple(subset_df, teams_list):
    team_pts_for_total_dict = create_initial_value_dict(teams_list, 0)
    team_season_dict = create_team_season_dict(subset_df, teams_list)
    team_num_games_played_dict = create_initial_value_dict(teams_list, 0)
    visitor_ppg_for_list = create_n_length_list(0, len(subset_df))
    home_ppg_for_list = create_n_length_list(0, len(subset_df))

    for index, row in subset_df.iterrows():
        visitor = row['Visitor']
        home = row['Home']

        if team_season_dict[visitor] != row['Season']:
            team_pts_for_total_dict[visitor] = 0
            team_num_games_played_dict[visitor] = 0
            team_season_dict[visitor] = row['Season']
        if team_season_dict[home] != row['Season']:
            team_pts_for_total_dict[home] = 0
            team_num_games_played_dict[home] = 0
            team_season_dict[home] = row['Season']

        visitor_pts_for_total = team_pts_for_total_dict[visitor]
        home_pts_for_total = team_pts_for_total_dict[home]

        visitor_games_played = team_num_games_played_dict[visitor]
        home_games_played = team_num_games_played_dict[home]

        if visitor_pts_for_total != 0 and visitor_games_played != 0:
            visitor_ppg_for_list[index] = visitor_pts_for_total / visitor_games_played
        else:
            visitor_ppg_for_list[index] = 0

        if home_pts_for_total != 0 and home_games_played != 0:
            home_ppg_for_list[index] = home_pts_for_total / home_games_played
        else:
            home_ppg_for_list[index] = 0

        team_pts_for_total_dict[visitor] += row['Visitor_PTS']
        team_pts_for_total_dict[home] += row['Home_PTS']

        team_num_games_played_dict[visitor] += 1
        team_num_games_played_dict[home] += 1

    return visitor_ppg_for_list, home_ppg_for_list


def create_ppg_against_lists_tuple(subset_df, teams_list):
    team_pts_against_total_dict = create_initial_value_dict(teams_list, 0)
    team_season_dict = create_team_season_dict(subset_df, teams_list)
    team_num_games_played_dict = create_initial_value_dict(teams_list, 0)
    visitor_ppg_against_list = create_n_length_list(0, len(subset_df))
    home_ppg_against_list = create_n_length_list(0, len(subset_df))

    for index, row in subset_df.iterrows():
        visitor = row['Visitor']
        home = row['Home']

        if team_season_dict[visitor] != row['Season']:
            team_pts_against_total_dict[visitor] = 0
            team_num_games_played_dict[visitor] = 0
            team_season_dict[visitor] = row['Season']
        if team_season_dict[home] != row['Season']:
            team_pts_against_total_dict[home] = 0
            team_num_games_played_dict[home] = 0
            team_season_dict[home] = row['Season']

        visitor_pts_against_total = team_pts_against_total_dict[visitor]
        home_pts_against_total = team_pts_against_total_dict[home]

        visitor_games_played = team_num_games_played_dict[visitor]
        home_games_played = team_num_games_played_dict[home]

        if visitor_pts_against_total != 0 and visitor_games_played != 0:
            visitor_ppg_against_list[index] = visitor_pts_against_total / visitor_games_played
        else:
            visitor_ppg_against_list[index] = 0

        if home_pts_against_total != 0 and home_games_played != 0:
            home_ppg_against_list[index] = home_pts_against_total / home_games_played
        else:
            home_ppg_against_list[index] = 0

        team_pts_against_total_dict[visitor] += row['Home_PTS']
        team_pts_against_total_dict[home] += row['Visitor_PTS']

        team_num_games_played_dict[visitor] += 1
        team_num_games_played_dict[home] += 1

    return visitor_ppg_against_list, home_ppg_against_list


def sturges_binning(games_df, column):
    minimum = games_df[column].min()
    maximum = games_df[column].max()
    bins = math.ceil(math.log2(len(games_df[column]))) + 1
    binwidth = (maximum - minimum) / bins

    bin_names = list(range(1, bins + 1))
    bin_ranges = list(range(minimum, maximum + binwidth, binwidth))

    new_range_col_name = column + '_Custom_Range'
    new_label_col_name = column + '_Custom_Label'

    games_df[new_range_col_name] = pd.cut(np.array(games_df[column]),
                                                              bins=bin_ranges)
    games_df[new_label_col_name] = (
        pd.cut(np.array(games_df[column]), bins=bin_ranges, labels=bin_names)).astype(np.float64)

    return games_df