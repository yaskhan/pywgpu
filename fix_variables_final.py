
with open('naga/front/glsl/variables.py', 'r+') as f:
    content = f.read()
    # Find the start of get_errors or clear_errors and truncate before it
    idx = content.rfind('def get_errors(self)')
    if idx != -1:
        f.seek(idx)
        f.truncate()
        f.write('    def get_errors(self) -> List[str]:\n')
        f.write('        """Get variable parsing errors."""\n')
        f.write('        return self.errors.copy()\n\n')
        f.write('    def clear_errors(self) -> None:\n')
        f.write('        """Clear variable parsing errors."""\n')
        f.write('        self.errors.clear()\n')
