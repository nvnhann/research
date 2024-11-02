import time
import pandas as pd
import numpy as np

CITY_DATA = {
    'chicago': 'chicago.csv',
    'new york city': 'new_york_city.csv',
    'washington': 'washington.csv'
}


def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter or "all" to apply no month
        (str) day - name of the day of week ilter by, or "all" to apply no day
    """

    print('Hello! Let\'s explore some US bikeshare data!')
    city = input('Would you like to see data for Chicago, '
                 'New York City, or Washington?\n').lower()
    while city not in ['chicago', 'new york city', 'washington']:
        print("Invalid input. Please try again.")
        city = input('Would you like to see data for Chicago, '
                     'New York City, or Washington?\n').lower()

    # Get user input for month (all, january, february, ... , june)
    month = input('Which month? January, February, March'
                  ', April, May, June, or "all"?\n').lower()
    while month not in ['all', 'january', 'february',
                        'march', 'april', 'may', 'june']:
        print("Invalid input. Please try again.")
        month = input('Which month? January, February, March,'
                      ' April, May, June, or "all"?\n').lower()
    # Get user input for day of week (all, monday, tuesday, ... sunday)
    day = input('Which day? Please type your response'
                ' as an example: Sunday, Monday, ..., or "all".\n').lower()
    while day not in ['all', 'monday', 'tuesday', 'wednesday',
                      'thursday', 'friday', 'saturday', 'sunday']:
        print("Invalid input. Please try again.")
        day = input('Which day? Please type your response'
                    'as an example: Sunday, Monday, ..., or "all".\n').lower()
    print('-'*40)
    return city, month, day


def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month, day if applicable.

    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month, "all" to apply no month filter
        (str) day - name of the day of week or "all" to apply no day filter
    Returns:
        df - Pandas DataFrame containing city data filtered by month and day
    """
    # load data file into a dataframe
    df = pd.read_csv(CITY_DATA[city])

    # convert the Start Time column to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])

    # extract month and day of week from Start Time to create new columns
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.day_name()

    # filter by month if applicable
    if month != 'all':
        # use the index of the months list to get the corresponding int
        months = ['january', 'february', 'march', 'april', 'may', 'june']
        month = months.index(month) + 1

        # filter by month to create the new dataframe
        df = df[df['month'] == month]

    # filter by day of week if applicable
    if day != 'all':
        # filter by day of week to create the new dataframe
        df = df[df['day_of_week'].str.lower() == day]

    return df


def time_stats(df):
    """Displays statistics on the most frequent times of travel."""

    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    # display the most common month
    popular_month = df['month'].mode()[0]
    print('Most Popular Month:', popular_month)

    # display the most common day of week
    popular_day = df['day_of_week'].mode()[0]
    print('Most Popular Day of Week:', popular_day)

    # display the most common start hour
    df['hour'] = df['Start Time'].dt.hour
    popular_hour = df['hour'].mode()[0]
    print('Most Popular Start Hour:', popular_hour)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    # display most commonly used start station
    popular_start_station = df['Start Station'].mode()[0]
    print('Most Commonly Used Start Station:', popular_start_station)

    # display most commonly used end station
    popular_end_station = df['End Station'].mode()[0]
    print('Most Commonly Used End Station:', popular_end_station)

    # display most frequent combination of start station and end station trip
    df['station_combination'] = df['Start Station']+" to "
    df['station_combination'] = df['station_combination'] + df['End Station']
    popular_station_combination = df['station_combination'].mode()[0]
    print('Most Frequent Combination of '
          'Start Station and End Station Trip:', popular_station_combination)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    # display total travel time
    total_travel_time = df['Trip Duration'].sum()
    print('Total Travel Time:', total_travel_time, 'seconds')

    # display mean travel time
    mean_travel_time = df['Trip Duration'].mean()
    print('Mean Travel Time:', mean_travel_time, 'seconds')

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def user_stats(df):
    """Displays statistics on bikeshare users."""

    print('\nCalculating User Stats...\n')
    start_time = time.time()
    # Display counts of user types
    user_types = df['User Type'].value_counts()
    print('Counts of User Types:\n', user_types)
    # Display counts of gender (only available for NYC and Chicago)
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        print('\nCounts of Gender:\n', gender_counts)
    else:
        print('Gender stats cannot be calculated because '
              'Gender data does not exist in this dataset.')
    if 'Birth Year' in df.columns:
        earliest_year = df['Birth Year'].min()
        most_recent_year = df['Birth Year'].max()
        most_common_year = df['Birth Year'].mode()[0]
        print('\nEarliest Year of Birth:', int(earliest_year))
        print('Most Recent Year of Birth:', int(most_recent_year))
        print('Most Common Year of Birth:', int(most_common_year))
    else:
        print('Birth Year stats cannot be calculated because '
              'Birth Year data does not exist in this dataset.')

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def display_data(df):
    """Displays statistics."""

    view_data = input('\nWould you like to view 5 rows of individual'
                      ' trip data? Enter yes or no\n').lower()
    start_loc = 0
    while view_data == 'yes':
        print(df.iloc[start_loc:start_loc+5])
        start_loc += 5
        view_data = input('\nWould you like to view 5 more rows of individual'
                          ' trip data? Enter yes or no\n').lower()


def main():
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)
        display_data(df)
        time_stats(df)
        station_stats(df)
        trip_duration_stats(df)
        user_stats(df)

        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() != 'yes':
            break


if __name__ == "__main__":
    main()
