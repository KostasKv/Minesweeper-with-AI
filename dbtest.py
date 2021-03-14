from sqlalchemy import MetaData, create_engine, insert, select
from sqlalchemy.sql import and_
from bitarray import bitarray
from sqlalchemy.types import LargeBinary

def main():
    # # bank account details
    # user = "cokk"
    # topsecretword = "8iCyrvxoK4RMitkZ" 
    # host = "lnx-cokk-1.lunet.lboro.ac.uk"
    # db_name = "cokk"
    
    user = "cokk"
    topsecretword = "password"
    host = "localhost"
    db_name = "cokk"

    # connect
    engine = create_engine(f'mysql://{user}:{topsecretword}@{host}/{db_name}?charset=utf8mb4')
    engine.connect()


    meta_data = MetaData(bind=engine, reflect=True)

    table_difficulty = meta_data.tables['difficulty']
    table_game = meta_data.tables['game']
    table_grid = meta_data.tables['grid']
    table_sample = meta_data.tables['sample']
    table_turn = meta_data.tables['turn']
    
    # query = select([table_difficulty.c.id]).where(
    #     and_(
    #         table_difficulty.c.rows == 16,
    #         table_difficulty.c.columns == 30,
    #         table_difficulty.c.mines == 99
    #         )
    #     )
    query = select([table_grid])
    
    result = engine.execute(query)

    for row in result:
        print(row)
        b = bitarray()
        b.frombytes(row[3])
        print(len(b))
        x = 5
    # print(result[0])
    



if __name__ == '__main__':
    main()
    a = bitarray()
    a.frombytes(b'\x01@\x010\x12\x10\x12\x00')
    x = 5