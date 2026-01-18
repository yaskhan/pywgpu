# SPIR-V utilities

def validate_spirv(data: bytes) -> bool:
    """Validates SPIR-V binary data.
    
    Performs basic validation of SPIR-V binary format:
    - Checks magic number
    - Validates header structure
    - Checks version
    
    Args:
        data: The SPIR-V binary data to validate.
        
    Returns:
        True if valid SPIR-V, False otherwise.
    """
    # SPIR-V files must be at least 20 bytes (5 words)
    if len(data) < 20:
        return False
    
    # Check magic number (0x07230203 in little-endian)
    magic = int.from_bytes(data[0:4], 'little')
    if magic != 0x07230203:
        return False
    
    # Check version (major.minor in bytes 4-7)
    version = int.from_bytes(data[4:8], 'little')
    major = (version >> 16) & 0xFF
    minor = (version >> 8) & 0xFF
    
    # Support SPIR-V 1.0 through 1.6
    if major != 1 or minor > 6:
        return False
    
    # Check generator magic number (bytes 8-11)
    # Any value is valid, just check it exists
    
    # Check bound (bytes 12-15) - must be non-zero
    bound = int.from_bytes(data[12:16], 'little')
    if bound == 0:
        return False
    
    # Check schema (bytes 16-19) - must be 0 for now
    schema = int.from_bytes(data[16:20], 'little')
    if schema != 0:
        return False
    
    # Basic validation passed
    return True

