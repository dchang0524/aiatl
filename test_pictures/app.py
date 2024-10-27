import streamlit as st
import requests
import urllib.parse
from PIL import Image
import io
import anthropic
from io import StringIO
import contextlib
import ast
import math
import numpy as np
from sympy import *
import sympy
from pix2tex.cli import LatexOCR
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import copy
import traceback
import tempfile
import os
import matlab.engine
from pathlib import Path
import re

# Initialize the OCR model
@st.cache_resource
def load_model():
    return LatexOCR()

def create_claude_client():
    """Create and return Anthropic API client"""
    api_key = st.session_state.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        return None
    return anthropic.Client(api_key=api_key)

def execute_python_safely(code: str, prev_locals: Optional[Dict[str, Any]] = None):
    """Execute Python code in a safe environment and capture output while preserving state"""
    # Create string buffer to capture output
    output_buffer = StringIO()
    
    # Create globals dictionary with necessary modules and functions
    globals_dict = {
        # Math modules
        'math': math,
        'np': np,
        'sympy': sympy,
        # SymPy functions and classes
        'Symbol': Symbol,
        'solve': solve,
        'Eq': Eq,
        'expand': expand,
        'factor': factor,
        'simplify': simplify,
        'latex': latex,
        'symbols': symbols,
        'init_printing': init_printing,
        'sin': sin,
        'cos': cos,
        'tan': tan,
        'sqrt': sqrt,
        'pi': pi,
        'E': E,
        'log': log,
        'exp': exp,
        # Python builtins
        'abs': abs,
        'all': all,
        'any': any,
        'len': len,
        'max': max,
        'min': min,
        'pow': pow,
        'print': print,
        'range': range,
        'round': round,
        'sum': sum,
        'int': int,
        'float': float,
        'str': str,
    }
    
    try:
        # Parse the code to check for potentially harmful operations
        tree = ast.parse(code)
        
        # Analyze the AST to ensure no harmful operations
        for node in ast.walk(tree):
            if isinstance(node, (ast.Delete, ast.AsyncFunctionDef, ast.ClassDef)):
                raise ValueError(f"Disallowed operation: {type(node).__name__}")
        
        # Initialize the local namespace with previous state if provided
        local_namespace = {}
        if prev_locals is not None:
            # Deep copy mutable objects to prevent modification of original namespace
            for key, value in prev_locals.items():
                if not key.startswith('_'):  # Skip internal variables
                    try:
                        if isinstance(value, (np.ndarray, sympy.Basic)):
                            # Special handling for NumPy arrays and SymPy objects
                            local_namespace[key] = value
                        else:
                            # For other objects, try to create a copy
                            local_namespace[key] = copy.deepcopy(value)
                    except (TypeError, AttributeError):
                        # If copying fails, use the original value
                        local_namespace[key] = value
        
        # Initialize sympy printing
        init_printing()
        
        print(f"Code: {code}")


        # Execute the code with captured output
        with contextlib.redirect_stdout(output_buffer):
            exec(code, globals_dict, local_namespace)
        
        # Get the output
        output = output_buffer.getvalue()

        print(f"Output: {output}")
        
        # Process and format the results
        results = []
        if local_namespace:
            for var_name, value in local_namespace.items():
                if not var_name.startswith('_'):  # Skip internal variables
                    if isinstance(value, sympy.Basic):
                        # Format SymPy expressions in LaTeX
                        results.append(f"{var_name} = $${latex(value)}$$")
                    elif isinstance(value, np.ndarray):
                        # Format NumPy arrays
                        results.append(f"{var_name} = {np.array2string(value, threshold=100)}")
                    else:
                        # Format other values
                        results.append(f"{var_name} = {str(value)}")
        
        # Combine output with variable state
        if results:
            output += "\nVariables:\n" + "\n".join(results)
        
        return True, output, local_namespace
    
    except Exception as e:
        return False, f"Error executing code: {str(e)}, {code}", None

class MatlabExecutor:
    """Handles MATLAB code execution and management with LaTeX output"""
    
    def __init__(self):
        self._engine = None
        self._temp_dir = None
    
    def __enter__(self):
        self.start_engine()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def start_engine(self):
        if self._engine is None:
            self._engine = matlab.engine.start_matlab()
            self._temp_dir = tempfile.mkdtemp()
            self._engine.addpath(self._temp_dir, nargout=0)
    
    def cleanup(self):
        if self._engine:
            try:
                self._engine.quit()
            finally:
                self._engine = None
        
        if self._temp_dir and os.path.exists(self._temp_dir):
            for file in Path(self._temp_dir).glob("*.*"):
                try:
                    file.unlink()
                except:
                    pass
            try:
                os.rmdir(self._temp_dir)
            except:
                pass
            self._temp_dir = None
    
    def matlab_to_latex(self, output: str) -> str:
        """Convert MATLAB output to LaTeX format"""
        # Split output into lines
        lines = output.strip().split('\n')
        latex_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Check if line contains equation-like content
            if any(char in line for char in "=-+*/^()[]{}"):
                # Remove 'ans =' if present
                line = re.sub(r'^ans\s*=\s*', '', line)
                
                # Convert MATLAB syntax to LaTeX
                # Replace basic operations
                line = line.replace('^', '^{').replace('*', '\\cdot ')
                
                # Add closing braces for exponents
                exp_count = line.count('^{')
                line = line + '}' * exp_count
                
                # Convert sqrt() to \sqrt{}
                line = re.sub(r'sqrt\((.*?)\)', r'\\sqrt{\1}', line)
                
                # Wrap in LaTeX equation delimiters
                latex_lines.append(f"$${line}$$")
            else:
                # For text output, just pass through
                latex_lines.append(line)
        
        return '\n'.join(latex_lines)

    def execute_matlab_code(self, code: str) -> Tuple[bool, str, Any]:
        """Execute MATLAB code and return LaTeX-formatted output"""
        if not self._engine:
            return False, "MATLAB engine not initialized", None
            
        try:
            # Create a temporary script file
            script_path = Path(self._temp_dir) / "temp_script.m"
            
            # Add LaTeX output formatting
            latex_code = [
                "try",
                "    diary('output.txt');",
                "    format compact;",  # Reduce whitespace in output
                "    syms_present = false;",
                "    try",
                "        syms x y z;  % Test if symbolic toolbox is available",
                "        syms_present = true;",
                "        latex_mode = true;",
                "    catch",
                "        latex_mode = false;",
                "    end",
                *code.split('\n'),  # Original code
                "    diary off;",
                "catch ME",
                "    diary off;",
                "    disp(['Error: ' ME.message]);",
                "end"
            ]
            
            # Write the code to file
            with open(script_path, 'w') as f:
                f.write('\n'.join(latex_code))
            
            # Execute the script
            self._engine.cd(self._temp_dir, nargout=0)
            self._engine.run('temp_script.m', nargout=0)
            
            # Read and process output
            output_path = Path(self._temp_dir) / "output.txt"
            output = ""
            if output_path.exists():
                with open(output_path, 'r') as f:
                    output = f.read()
                output_path.unlink()
            
            script_path.unlink()
            
            # Convert output to LaTeX format
            latex_output = self.matlab_to_latex(output)
            
            return True, latex_output, None
            
        except Exception as e:
            return False, f"Error executing MATLAB code: {str(e)}", None

def modify_process_with_claude(client, prompt, max_iterations=5):
    """Modified version of process_with_claude to handle MATLAB code generation with LaTeX output"""
    conversation = []
    iterations = 0
    
    with MatlabExecutor() as matlab_exec:
        while iterations < max_iterations:
            try:
                if not conversation:
                    messages = [{
                        "role": "user",
                        "content": f"""You are a mathematical problem solver who thinks step by step.
                        For each step:
                        1. Explain the step in words
                        2. Write MATLAB code to perform the calculation. 
                           IMPORTANT: 
                           - Do not write functions, only write direct MATLAB commands
                           - Use the symbolic math toolbox when dealing with algebraic expressions
                           - Use latex() function to display symbolic expressions in LaTeX format
                           
                           Example MATLAB code:
                           ```matlab
                           % Define symbolic variables
                           syms x
                           % Create expression
                           expr = x^2 + 2*x + 1;
                           % Solve equation
                           solution = solve(expr == 0, x);
                           % Display in LaTeX format
                           disp('The equation is:');
                           disp(latex(expr));
                           disp('The solutions are:');
                           disp(latex(solution));
                           ```
                        3. Present the result in a clear format
                        4. Determine if more steps are needed
                        
                        Current problem or step: {prompt}
                        
                        Important:
                        - Use symbolic math when possible
                        - Use latex() for mathematical expressions
                        - Define all variables clearly
                        - Add comments to explain the code
                        - Show intermediate calculations
                        - Use semicolons to suppress non-LaTeX output
                        
                        If you generate MATLAB code, wrap it in ```matlab``` markers.
                        If the problem is solved, end with "SOLUTION COMPLETE"."""
                    }]
                else:
                    messages = conversation + [{
                        "role": "user",
                        "content": f"""Previous results:
                        {prompt}
                        
                        Continue solving. Remember:
                        - Use symbolic math
                        - Use latex() for output
                        - Define variables clearly
                        - Show intermediate calculations"""
                    }]

                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    messages=messages,
                    temperature=0
                )
                
                content = response.content[0].text
                conversation.extend([messages[-1], {"role": "assistant", "content": content}])
                
                # Extract and execute MATLAB code blocks
                code_blocks = []
                lines = content.split('\n')
                in_code_block = False
                current_block = []
                
                for line in lines:
                    if line.strip().startswith('```matlab'):
                        in_code_block = True
                        continue
                    elif line.strip().startswith('```') and in_code_block:
                        in_code_block = False
                        code_blocks.append('\n'.join(current_block))
                        current_block = []
                        continue
                    elif in_code_block:
                        current_block.append(line)
                
                # Execute code blocks and format output
                code_results = []
                for code in code_blocks:
                    success, result, _ = matlab_exec.execute_matlab_code(code)
                    if success and result:
                        code_results.append(result)  # Result is already in LaTeX format
                    else:
                        code_results.append(f"Error: {result}")
                
                result = {
                    'explanation': content,
                    'code_results': code_results
                }
                
                if "SOLUTION COMPLETE" in content:
                    return result
                
                prompt = str(code_results)
                iterations += 1
                
            except Exception as e:
                return {'error': f"Error processing with Claude: {str(e)}"}
        
        return {'error': "Maximum iterations reached without solution"}

def process_with_claude(client, prompt, max_iterations=10):
    """Process mathematical problem with Claude, generating and executing code as needed"""
    conversation = []
    iterations = 0
    local_namespace = None  # Store variables between iterations
    
    while iterations < max_iterations:
        try:
            # Create system prompt and message
            if not conversation:
                messages = [{
                    "role": "user",
                    "content": f"""You are a expert mathematical problem solver who thinks step by step.
                    For each step:
                    1. Explain the step in words
                    2. If calculation is needed, write Python code to perform it. 
                       Available libraries: numpy (as np), sympy
                       For symbolic math, use sympy. For numerical calculations, use numpy.
                       Always format mathematical expressions in LaTeX using sympy.latex()
                       Example using sympy:
                       ```python
                       from sympy import symbols, solve, latex
                       x = symbols('x')
                       expr = x**2 + 2*x + 1
                       solution = solve(expr, x)
                       print(f"The solutions are: $${{latex(solution)}}$$")
                       ```
                       ```python
                       from sympy import symbols, solve, latex
                       x = symbols('x')
                       expr = x**2 + 2*x + 1
                       from sympy import symbols, integrate, cos, sin, latex
                       # Define variable
                       x = symbols('x')
                       # Define the integrand
                       integrand = 4*x*cos(2-3*x) # Note that we need to specify multiplication with *
                       # Solve the integral
                       result = integrate(integrand, x)
                       print(f"The integral is: $${{latex(result)}}$$")
                       ```
                    3. Present the result in LaTeX format when it's mathematical
                    4. Determine if more steps are needed
                    
                    Current problem or step: {prompt}
                    
                    Important:
                    - Define all variables you need in each code block
                    - Always wrap mathematical expressions in LaTeX $$ markers
                    - Use sympy.latex() to convert expressions to LaTeX
                    - If showing intermediate calculations, format them in LaTeX
                    - Show all logical mathematical steps (e.g., simplifications, substitutions)
                    - When using a particular mathematical method, explain the method in words
                    
                    If you generate Python code, wrap it in ```python``` markers.
                    If the problem is solved, end with "SOLUTION COMPLETE"."""
                }]
            else:
                messages = conversation + [{
                    "role": "user",
                    "content": f"""Previous results:
                    {prompt}
                    
                    Continue solving. Remember to:
                    1. Format all mathematical expressions in LaTeX
                    2. Define any variables you need, even if defined in previous steps
                    3. Show all intermediate calculations"""
                }]

            # Get Claude's response
            response = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=2000,
                messages=messages,
                temperature=0
            )
            
            content = response.content[0].text
            conversation.extend([messages[-1], {"role": "assistant", "content": content}])
            
            # Extract any Python code
            code_blocks = []
            lines = content.split('\n')
            in_code_block = False
            current_block = []
            
            for line in lines:
                if line.strip().startswith('```python'):
                    in_code_block = True
                    continue
                elif line.strip().startswith('```') and in_code_block:
                    in_code_block = False
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                    continue
                elif in_code_block:
                    current_block.append(line)
            
            # Execute any code blocks and capture results
            code_results = []
            for code in code_blocks:
                success, result, new_locals = execute_python_safely(code, local_namespace)
                if success:
                    code_results.append(f"Code execution result:\n{result}")
                    local_namespace = new_locals  # Update the namespace for next iteration
                else:
                    code_results.append(f"Error: {result}")
            
            # Create combined result
            result = {
                'explanation': content,
                'code_results': code_results
            }
            
            # Check if solution is complete
            if "SOLUTION COMPLETE" in content:
                return result
            
            # Update prompt with results for next iteration
            prompt = str(code_results)
            iterations += 1
            
        except Exception as e:
            print(traceback.format_exc())

            return {'error': f"Error processing with Claude: {str(e)}"}
    
    return {'error': "Maximum iterations reached without solution"}

def solve_with_wolfram(latex_code, wolfram_app_id):
    """Solve equation using Wolfram Alpha API"""
    if not latex_code.strip():
        st.error('No LaTeX code available.')
        return []
    
    try:
        tex = urllib.parse.quote(latex_code)
        api_url = f"http://api.wolframalpha.com/v2/query?appid={wolfram_app_id}&input={tex}&podstate=Step-by-step+solution&format=image&output=json"
        
        res = requests.get(api_url)
        data = res.json()

        print(data)
        
        image_urls = []
        if data['queryresult']['success'] and 'pods' in data['queryresult']:
            pods = data['queryresult']['pods']
            for pod in pods:
                subpods = pod.get('subpods', [])
                for subpod in subpods:
                    img = subpod.get('img')
                    if isinstance(img, dict):
                        img_src = img.get('src')
                        if img_src:
                            image_urls.append(img_src)
            return image_urls
        else:
            error_msg = data['queryresult'].get('error', {}).get('msg', 'Wolfram Alpha could not interpret the input.')
            st.error(error_msg)
            return []
    except Exception as e:
        st.error(f'An error occurred: {e}')
        return []

def main():
    st.set_page_config(page_title="LaTeX OCR and Solver", layout="wide")
    
    st.title("MathPix")
    
    # Initialize session state variables
    if 'latex_code' not in st.session_state:
        st.session_state.latex_code = ""
    if 'solution_images' not in st.session_state:
        st.session_state.solution_images = []
    
    # Load the OCR model
    model = load_model()
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("Settings")
        anthropic_api_key = st.text_input("Anthropic API Key", type="password")
        if anthropic_api_key:
            st.session_state['ANTHROPIC_API_KEY'] = anthropic_api_key
            
        wolfram_app_id = st.text_input("Wolfram Alpha App ID", type="password")
        
        st.markdown("""
        Get your API keys:
        - [Anthropic API](https://console.anthropic.com/)
        - [Wolfram Alpha API](https://developer.wolframalpha.com/portal/myapps/)
        """)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Image")
        upload_option = st.radio("Choose input method:", ["Upload Image", "Paste Image"])
        
        if upload_option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=False)
                
                if st.button("Convert to LaTeX"):
                    with st.spinner("Converting to LaTeX..."):
                        try:
                            st.session_state.latex_code = model(image)
                        except Exception as e:
                            st.error(f"Error during OCR: {e}")
        
        else:  # Paste Image
            st.write("Please copy an image to your clipboard and use the button below to paste it")
            if st.button("Paste from Clipboard"):
                st.warning("Note: Clipboard access is not available in Streamlit. Please use the upload option instead.")
    
    with col2:
        st.header("LaTeX Code")
        latex_code = st.text_area("Edit LaTeX code if needed:", value=st.session_state.latex_code, height=150)
        
        if latex_code:
            st.header("LaTeX Preview")
            try:
                st.latex(latex_code)
            except Exception as e:
                st.error(f"Error rendering LaTeX: {e}")
        
        solver_option = st.radio("Choose solver:", ["Claude (Python)", "Claude (MATLAB)", "Wolfram Alpha"])
        
        if solver_option == "Claude (Python)":
            if st.button("Solve with Claude (Python)"):
                client = create_claude_client()
                if client:
                    with st.spinner("Processing with Claude..."):
                        result = process_with_claude(client, f"Solve this mathematical problem: {latex_code}")
                        
                        st.header("Solution Steps")
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            st.markdown(result['explanation'])
                            if result['code_results']:
                                st.header("Calculations")
                                for code_result in result['code_results']:
                                    st.code(code_result)
        
        elif solver_option == "Claude (MATLAB)":
            if st.button("Solve with Claude (MATLAB)"):
                client = create_claude_client()
                if client:
                    with st.spinner("Processing with Claude using MATLAB..."):
                        try:
                            result = modify_process_with_claude(client, f"Solve this mathematical problem: {latex_code}")
                            
                            st.header("Solution Steps")
                            if 'error' in result:
                                st.error(result['error'])
                            else:
                                st.markdown(result['explanation'])
                                if result['code_results']:
                                    st.header("MATLAB Calculations")
                                    for code_result in result['code_results']:
                                        st.code(code_result)
                        except Exception as e:
                            st.error(f"Error during MATLAB execution: {str(e)}")
        
        else:  # Wolfram Alpha
            if st.button("Solve with Wolfram Alpha") and wolfram_app_id:
                with st.spinner("Solving with Wolfram Alpha..."):
                    image_urls = solve_with_wolfram(latex_code, wolfram_app_id)
                    if image_urls:
                        st.session_state.solution_images = image_urls
    
    # Solution section for Wolfram Alpha
    if solver_option == "Wolfram Alpha" and st.session_state.solution_images:
        st.header("Wolfram Alpha Solution")
        for url in st.session_state.solution_images:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    st.image(image, use_column_width=False)
            except Exception as e:
                st.error(f"Error loading solution image: {e}")

if __name__ == '__main__':
    main()