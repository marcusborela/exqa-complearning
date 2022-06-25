class class_docto_encontrado(object):

    def __init__(self, id, text:str):
        self._id = id
        self._text = text

    @property
    def text(self):
        return self._text
    @property
    def id(self):
        return self._id