from typing import Optional, Any, List
from dataclasses import dataclass, field
import uuid


@dataclass
class Op:
    '''
    Operation data class.
    '''
    output: Optional[List] = field(default_factory=list)
    func_name: Optional[str] = None
    input: Optional[List] = field(default_factory=list)

    place: Optional[Any] = None
    owner: Optional[Any] = None
    requires_grad: Optional[bool] = False
    operation_type: Optional[str] = None  # _opertaions, _init_operations, _standalone_operations
    func: Optional[Any] = None

    def set_identifier(self, nid=None):
        """Initialize self._identifier"""
        if nid is None:
            self._identifier = str(uuid.uuid1())
        else:
            self._identifier = nid