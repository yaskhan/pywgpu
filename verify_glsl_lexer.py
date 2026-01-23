from naga.front.glsl.lexer import Lexer
import os

def test_lexer():
    source = """
    #version 450
    layout(location = 0) in vec3 position;
    layout(location = 0) out vec4 fragColor;
    
    uniform float time;
    
    void main() {
        fragColor = vec4(position, 1.0 + time);
    }
    """
    
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    print(f"Total tokens: {len(tokens)}")
    for i, token in enumerate(tokens):
        print(f"{i}: {token.value} (data: {token.data})")

if __name__ == "__main__":
    test_lexer()
