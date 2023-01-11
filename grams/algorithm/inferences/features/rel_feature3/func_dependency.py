from collections import defaultdict
from grams.inputs.linked_table import LinkedTable
from sm.misc.fn_cache import CacheMethod


class FunctionalDependencyDetector:
    def __init__(self, table: LinkedTable):
        self.table = table

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def is_functional_dependency(
        self, source_column_index: int, target_column_index: int
    ) -> bool:
        """Test whether values in the target column is uniquely determiend by the values in the source column. True if
        it's FD.

        Parameters
        ----------
        source_column_index
        target_column_index

        Returns
        -------
        """
        sci = source_column_index
        tci = target_column_index

        # find a mapping from
        source_map = self._get_value_map(source_column_index)
        target_map = {
            ri: key
            for key, rows in self._get_value_map(target_column_index).items()
            for ri in rows
        }

        n_violate_fd = 0
        for key, rows in source_map.items():
            target_keys = {target_map[ri] for ri in rows}
            if len(target_keys) > 1:
                n_violate_fd += 1

        if len(source_map) == 0:
            return True

        if n_violate_fd / len(source_map) > 0.01:
            return False
        return True

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def _get_value_map(self, column_index: int):
        """Get a map of values in a column to its row numbers (possible duplication).
        This function is not perfect now. The value of the column is considered to be list of entities (if exist) or
        just the value of the cell
        """
        map = defaultdict(list)
        col = self.table.table.get_column_by_index(column_index)
        for ri in range(self.table.size()):
            # TODO: check why we need entity_id to determine functional dependency here
            # the reason we may want to use entity_id is because we may have a case where
            # two entities have the same name but different id. how rare is this case?
            # links = self.table.links[ri][column_index]
            # ents = [link.entity_id for link in links if link.entity_id is not None]
            # if len(ents) > 0:
            #     key = tuple(ents)
            # else:
            key = col.values[ri].strip()
            map[key].append(ri)
        return dict(map)
