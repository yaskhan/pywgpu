from dataclasses import dataclass
from typing import List, Iterator, Tuple, Any
from naga import ir, UniqueArena
from .overload_set import OverloadSet
from .rule import Rule
from naga.proc import TypeResolution
from .utils import non_struct_equivalent, automatically_converts_to, is_abstract

@dataclass(frozen=True)
class ListOverloadSet: # Renamed from List to avoid conflict with typing.List
    """
    A simple list of overloads.
    """
    members_mask: int
    rules: List[Rule]

    @classmethod
    def from_rules(cls, rules: List[Rule]) -> 'ListOverloadSet':
        if len(rules) >= 64:
            raise ValueError("ListOverloadSet can only hold up to 63 rules")
        mask = (1 << len(rules)) - 1
        return cls(mask, rules)

    def is_empty(self) -> bool:
        return self.members_mask == 0

    def min_arguments(self) -> int:
        if self.is_empty(): raise ValueError("OverloadSet is empty")
        return min(len(rule.arguments) for _, rule in self.members())

    def max_arguments(self) -> int:
        if self.is_empty(): raise ValueError("OverloadSet is empty")
        return max(len(rule.arguments) for _, rule in self.members())

    def members(self) -> Iterator[Tuple[int, Rule]]:
        from .one_bits_iter import OneBitsIter
        for bit in OneBitsIter(self.members_mask):
            yield (1 << bit, self.rules[bit])

    def filter(self, pred) -> 'ListOverloadSet':
        filtered_members = 0
        for mask, rule in self.members():
            if pred(rule):
                filtered_members |= mask
        return ListOverloadSet(filtered_members, self.rules)

    def arg(self, i: int, ty: ir.TypeInner, types: UniqueArena[ir.Type]) -> 'ListOverloadSet':
        def pred(rule: Rule) -> bool:
            if i >= len(rule.arguments):
                return False
            rule_ty = rule.arguments[i].inner_with(types)
            return (non_struct_equivalent(ty, rule_ty, types) or 
                    automatically_converts_to(ty, rule_ty, types) is not None)
        return self.filter(pred)

    def concrete_only(self, types: UniqueArena[ir.Type]) -> 'ListOverloadSet':
        def pred(rule: Rule) -> bool:
            return all(not is_abstract(arg_ty.inner_with(types), types) for arg_ty in rule.arguments)
        return self.filter(pred)

    def most_preferred(self) -> Rule:
        if self.is_empty(): raise ValueError("OverloadSet is empty")
        _, rule = next(self.members())
        return rule

    def overload_list(self, gctx: Any = None) -> List[Rule]:
        return [rule for _, rule in self.members()]

    def allowed_args(self, i: int, gctx: Any = None) -> List[TypeResolution]:
        return [rule.arguments[i] for _, rule in self.members()]

    def for_debug(self, types: UniqueArena[ir.Type]) -> Any:
        return [rule for _, rule in self.members()]
