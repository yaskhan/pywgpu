
with open('naga/front/glsl/variables.py', 'r') as f:
    lines = f.readlines()

# It seems the last line is bad or the file is truncated
# Let's just rewrite the last few lines safely
# The last good line is 366:     
# 367:     def clear_errors(self) -> None:
# 368:         """Clear variable parsing errors."""
# 369:         self.errors.clear()

new_lines = lines[:366]
new_lines.append('    def clear_errors(self) -> None:\n')
new_lines.append('        """Clear variable parsing errors."""\n')
new_lines.append('        self.errors.clear()\n')

with open('naga/front/glsl/variables.py', 'w') as f:
    f.writelines(new_lines)
