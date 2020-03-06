import re
from typing import List, Union, Iterable


class NaturalSort:

    @staticmethod
    def atoi(text: str) -> int:
        return int(text) if text.isdigit() else text

    @staticmethod
    def natural_keys(text: str) -> List[Union[str, int]]:
        """
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        """
        return [NaturalSort.atoi(c) for c in re.split(r'(\d+)', text)]

    @staticmethod
    def sorted(data: Iterable):
        return sorted(data, key=NaturalSort.natural_keys)
