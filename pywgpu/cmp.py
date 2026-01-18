# Comparison utilities
# In Rust this provides macros for Eq, Ord, Hash proxies.
# In Python, we use decorators to add comparison methods.

def impl_eq_ord_hash_proxy(inner_field: str = '_inner'):
    """Decorator to implement equality and hashing by proxying to an inner field.
    
    This decorator adds __eq__, __ne__, __hash__, __lt__, __le__, __gt__, __ge__
    methods to a class by delegating to an inner field.
    
    Args:
        inner_field: Name of the field to proxy to (default: '_inner').
        
    Returns:
        A decorator function.
        
    Example:
        @impl_eq_ord_hash_proxy('_id')
        class Resource:
            def __init__(self, id):
                self._id = id
    """
    def decorator(cls):
        # Add equality methods
        def __eq__(self, other):
            if not isinstance(other, cls):
                return NotImplemented
            return getattr(self, inner_field) == getattr(other, inner_field)
        
        def __ne__(self, other):
            if not isinstance(other, cls):
                return NotImplemented
            return getattr(self, inner_field) != getattr(other, inner_field)
        
        # Add hash method
        def __hash__(self):
            return hash(getattr(self, inner_field))
        
        # Add ordering methods
        def __lt__(self, other):
            if not isinstance(other, cls):
                return NotImplemented
            return getattr(self, inner_field) < getattr(other, inner_field)
        
        def __le__(self, other):
            if not isinstance(other, cls):
                return NotImplemented
            return getattr(self, inner_field) <= getattr(other, inner_field)
        
        def __gt__(self, other):
            if not isinstance(other, cls):
                return NotImplemented
            return getattr(self, inner_field) > getattr(other, inner_field)
        
        def __ge__(self, other):
            if not isinstance(other, cls):
                return NotImplemented
            return getattr(self, inner_field) >= getattr(other, inner_field)
        
        # Attach methods to class
        cls.__eq__ = __eq__
        cls.__ne__ = __ne__
        cls.__hash__ = __hash__
        cls.__lt__ = __lt__
        cls.__le__ = __le__
        cls.__gt__ = __gt__
        cls.__ge__ = __ge__
        
        return cls
    
    return decorator

