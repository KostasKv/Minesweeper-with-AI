from sqlalchemy import create_engine, Table, Column, MetaData, Integer, String

def main():
    # bank account details
    user = "cokk"
    topsecretword = "8iCyrvxoK4RMitkZ" 
    host = "lnx-cokk-1.lunet.lboro.ac.uk"
    db_name = "cokk"

    # connect
    engine = create_engine(f'mysql://{user}:{topsecretword}@{host}/{db_name}', echo = True)
    engine.connect()

    # Home deco
    meta = MetaData()
    chair = Table(
        'chair', meta, 
        Column('id', Integer, primary_key = True), 
        Column('whatwood', String(69)), 
        Column('whowood', String(69)), 
    )
    meta.create_all(engine)

    # print(engine)


if __name__ == '__main__':
    main()