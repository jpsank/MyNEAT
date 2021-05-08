

class Index(dict):
    """
    A way to store objects indexed by a primary attribute.
    """
    def __init__(self, primary_key):
        super().__init__()
        self.primary_key = primary_key

    def remove(self, item):
        self.__delitem__(getattr(item, self.primary_key))

    def add(self, item):
        """ Add an items to the index. """
        self.__setitem__(getattr(item, self.primary_key), item)


def flatten_items(d, *key_chain):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from flatten_items(value, *key_chain, key)
        else:
            yield (*key_chain, key) if key_chain else key, value


def flatten_keys(d, *key_chain):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from flatten_items(value, *key_chain, key)
        else:
            yield (*key_chain, key) if key_chain else key


def flatten_values(d):
    for value in d.values():
        if isinstance(value, dict):
            yield from flatten_values(value)
        else:
            yield value


def count(d):
    return sum(count(v) if isinstance(v, dict) else 1 for v in d.values())


class MultipleIndex:
    """
    A way to store objects indexed by multiple of their attributes.
    """
    def __init__(self, primary_keys):
        super().__init__()
        self.primary_keys = primary_keys
        self.data = {}

    def __getitem__(self, k_chain):
        ret = self.data
        for k in k_chain:
            ret = ret[k]
        return ret

    def __setitem__(self, k_chain, value):
        dic = self.data
        for k in k_chain[:-1]:
            dic = dic.setdefault(k, {})
        dic[k_chain[-1]] = value

    def __delitem__(self, k_chain):
        dic = self.data
        for k in k_chain[:-1]:
            dic = dic[k]
        del dic[k_chain[-1]]

    def __contains__(self, k_chain):
        dic = self.data
        for k in k_chain:
            if (dic := dic.get(k)) is None:
                return False
        return True

    def __len__(self):
        """ Return number of items in index. """
        return count(self.data)

    def get(self, k_chain, default=None):
        ret = self.data
        for k in k_chain:
            if (ret := ret.get(k)) is None:
                return default
        return ret

    def items(self):
        """ Iterate index by key chain and value. """
        return flatten_items(self.data)

    def keys(self):
        return flatten_keys(self.data)

    def values(self):
        return flatten_values(self.data)

    def remove(self, item):
        self.__delitem__([getattr(item, k) for k in self.primary_keys])

    def add(self, item):
        """ Add an items to the index. """
        self.__setitem__([getattr(item, k) for k in self.primary_keys], item)


# if __name__ == '__main__':
#
#     class Item:
#         def __init__(self, _id, x, y, angle=0):
#             self.id = _id
#             self.x, self.y = x, y
#             self.angle = angle
#
#         def __repr__(self):
#             return f"Item<id={self.id}, pos=({self.x}, {self.y}), angle={self.angle}>"
#
#
#     index = Index(Item, ["y", "x", "id"])
#
#     for i in range(11):
#         index.add(Item(i+1, 10*i, 10*i))
#
#     print(index[100, 100, 11])
#     index[36, 36, 12] = Item(12, 36, 36)
#
#     for keys, value in index:
#         print(keys, value)
#
#     print(index.size())
#     del index[36, 36, 12]
#     print(index.size())
#
#     print()
#     print()
#
#     index = Index(Item, ["id"])
#
#     for i in range(10):
#         index.add(Item(i+1, 10*i, 10*i))
#
#     print(index[10])
#     index[11] = Item(12, 36, 36)
#
#     for keys, value in index:
#         print(keys, value)
