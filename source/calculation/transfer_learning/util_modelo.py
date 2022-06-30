"""
rotinas utilitárias para modelos de reranking
"""
from typing import  Optional, List, Union, Mapping, Any
from dataclasses import dataclass

class Query:
    """Class representing a query.
    A query contains the query text itself and potentially other metadata.

    Parameters
    ----------
    text : str
        The query text.
    id : Optional[str]
        The query id.
    """

    def __init__(self, text: str, id: Optional[str] = None): # pylint: disable=redefined-builtin, invalid-name # código de Query externa
        self.text = text
        self.id = id # pylint: disable= invalid-name # código de Query externa


class Text:
    """Class representing a text to be reranked.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.

    Parameters
    ----------
    text : str
        The text to be reranked.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the text. For example, the score might be the BM25 score
        from an initial retrieval stage.
    title : Optional[str]
        The text's title.
    """

    def __init__(self,
                 text: str,
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0,
                 title: Optional[str] = None):
        self.text = text
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.title = title


class ClassDocto(object): # pylint: disable=missing-class-docstring # código de origem externa
    def __init__(self, id_docto, text:str):
        self._id_docto = id_docto
        self._text = text
        self._score = None

    @property
    def text(self): # pylint: disable=missing-function-docstring # código de origem externa
        return self._text

    @property
    def id_docto(self): # pylint: disable=missing-function-docstring # código de origem externa
        return self._id_docto

    @property
    def score(self): # pylint: disable=missing-function-docstring # código de origem externa
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

def compare_dicionarios(dict_1, dict_2, dict_1_name, dict_2_name, path=""):
    """Compare two dictionaries recursively to find non mathcing elements (keys and values)

    Args:
        dict_1: dictionary 1
        dict_2: dictionary 2

    Returns: difference

    Fonte:  https://stackoverflow.com/questions/27265939/comparing-python-dictionaries-and-nested-dictionaries/27266178

    """
    err = ''
    key_err = ''
    value_err = ''
    old_path = path
    for k in dict_1.keys():  # antes old_path + "[%s]" % k
        path = old_path + f"[{k}]"
        if not k in dict_2:
            key_err += f"Key {dict_2_name}{path} not in {dict_2_name}\n"
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dicionarios(dict_1[k],
                                           dict_2[k], 'd1', 'd2', path)
            else:
                if dict_1[k] != dict_2[k]:
                    value_err += f"Value of {dict_1_name}{path} ({dict_1[k]}) not same as {dict_2_name}{path} ({dict_2[k]})\n"

    for k in dict_2.keys():
        path = old_path + f"[{k}]"
        if not k in dict_1:
            key_err += f"Key {dict_1_name}{path} not in {dict_2_name}\n"

    return key_err + value_err + err

def compare_dicionarios_chaves(dict_1, dict_2, dict_1_name, dict_2_name, path=""):
    """Compare two dictionaries recursively to find non mathcing keys

    Args:
        dict_1: dictionary 1
        dict_2: dictionary 2

    Returns: difference

    Fonte:  https://stackoverflow.com/questions/27265939/comparing-python-dictionaries-and-nested-dictionaries/27266178

    """
    err = ''
    key_err = ''
    old_path = path
    for k in dict_1.keys():
        path = old_path + f"[{k}]"
        if not k in dict_2:
            key_err += f"Key {dict_2_name}{path} not in {dict_2_name}\n"
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dicionarios_chaves(
                    dict_1[k], dict_2[k], 'd1', 'd2', path)

    for k in dict_2.keys():
        path = old_path + f"[{k}]"
        if not k in dict_1:
            key_err += f"Key {dict_1_name}{path} not in {dict_2_name}\n"
    return key_err + err

