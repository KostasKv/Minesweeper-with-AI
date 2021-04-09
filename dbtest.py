from sqlalchemy import MetaData, create_engine, insert, select
from sqlalchemy.sql import and_
from bitarray import bitarray
from sqlalchemy.types import LargeBinary
import pandas as pd
import seaborn as sns
import matplotlib as plt

def main():
    engine, meta_data = get_database_engine_and_reflected_meta_data()

    table_difficulty = meta_data.tables['difficulty']
    table_game = meta_data.tables['game']
    table_grid = meta_data.tables['grid']
    table_sample = meta_data.tables['sample']
    table_turn = meta_data.tables['turn']
    

    query = select([table_grid])
    raw_query = "SELECT difficulty.name, sample_width, sample_height, use_mine_count, first_click_always_zero, SUM(win) AS wins, COUNT(*) AS num_games, SUM(win) / COUNT(*) AS win_rate FROM grid JOIN game on grid.id=game.grid_id JOIN difficulty ON difficulty.id=grid.difficulty_id GROUP BY difficulty_id, sample_width, sample_height, use_mine_count, first_click_always_zero;"
    

    # Fetch data and store in panda's DataFrame
    result = engine.execute(raw_query)
    df = pd.DataFrame(result.fetchall())
    df.columns = result.keys()

    # sns.barplot(x='difficulty_id', y='samples_considered', hue='sample_size', data=data1)
    # plt.show()
    order =['Beginner (9x9) \nwith mine count',
        'Beginner (9x9) \nwithout mine count',
        'Intermediate (16x16) \nwith mine count',
        'Intermediate (16x16) \nwithout mine count',
        'Expert (16x30) \nwith mine count',
        'Expert (16x30) \nwithout mine count',]

    print(df.info(verbose=True))
    df = df.convert_dtypes()
    df['win_rate'] = pd.to_numeric(df['win_rate'])

    sns.barplot(x='name', y='win_rate', hue='sample_width', data=df)
    # fig = plt.gcf()
    # fig.set_size_inches(12, 8)
    plt.show()
    # for row in result:
    #     print(row)
        # b = bitarray()
        # b.frombytes(row[3])
        # print(len(b))
        # x = 5
    # print(result[0])
    

def get_database_engine_and_reflected_meta_data():
    # bank account details
    user = "cokk"
    # topsecretword = "8iCyrvxoK4RMitkZ" 
    # host = "lnx-cokk-1.lunet.lboro.ac.uk"
    db_name = "cokk"

    topsecretword = "password"
    host = "localhost"

    engine = create_engine(f'mysql://{user}:{topsecretword}@{host}/{db_name}?charset=utf8mb4', isolation_level='SERIALIZABLE')
    meta_data = MetaData()
    meta_data.reflect(engine)

    return (engine, meta_data)

if __name__ == '__main__':
    main()