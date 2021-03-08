from sqlalchemy import create_engine

def main():
    engine = create_engine('mysql://cokk@lnx-cokk-1.lunet.lboro.ac.uk', echo = True)



if __name__ == '__main__':
    main()