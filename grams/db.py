import grams.misc as M
from grams.config import DATA_DIR
from grams.kg_data.wikidatamodels import QNode, WDProperty, WDClass


class WDClassDB(M.RocksDBStore[str, WDProperty]):
    instance = None

    @staticmethod
    def get_instance():
        if WDClassDB.instance is None:
            WDClassDB.instance = WDClassDB(DATA_DIR / "wdclasses.db")
        return WDClassDB.instance

    def deserialize(self, value):
        return WDClass.deserialize(value)


class QNodeDB(M.RocksDBStore[str, QNode]):
    instance = None

    @staticmethod
    def get_instance():
        if QNodeDB.instance is None:
            QNodeDB.instance = QNodeDB(DATA_DIR / "qnodes.db")
        return QNodeDB.instance

    def __setitem__(self, key, qnode):
        self.db.put(key.encode(), qnode.serialize())

    def deserialize(self, value):
        return QNode.deserialize(value)


class Wikipedia2WikidataDB(M.RocksDBStore[str, str]):
    instance = None

    @staticmethod
    def get_instance():
        if Wikipedia2WikidataDB.instance is None:
            Wikipedia2WikidataDB.instance = Wikipedia2WikidataDB(DATA_DIR / "enwiki_links.db")
        return Wikipedia2WikidataDB.instance

    def deserialize(self, value):
        return value.decode()


if __name__ == '__main__':
    # db = Wikipedia2WikidataDB.get_instance()
    # print(db.get('Cristiano Ronaldo11', None))
    db = WDClassDB.get_instance()
    print("Q69756896" in db)
    print(db.db.key_may_exist(b"Q69756896", fetch=True))
    print(db['Q69756896'])