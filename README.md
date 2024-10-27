# MathPix: AI-Powered Mathematical Expression Solver

![Demo](assets/demo.png)

## Inspiration

Our inspiration for MathPix came from a common frustration in the academic and scientific community - the tedious process of converting handwritten or image-based mathematical expressions into solvable digital format. Students and researchers often struggle with:

- Taking photos of equations from textbooks or handwritten notes
- Manually transcribing mathematical expressions into computer-readable format
- Switching between different tools for OCR, equation solving, and visualization
- Getting step-by-step explanations that are both rigorous and understandable

We wanted to create a seamless, end-to-end solution that could take a photo of a mathematical expression and not just recognize it, but solve it with detailed explanations.

## What it does

MathPix is an integrated mathematical problem-solving platform that combines:

1. **Image-to-LaTeX Conversion**:
   - Accepts uploaded images containing mathematical expressions
   - Uses advanced OCR (Pix2Tex) to convert images into accurate LaTeX code
   - Provides real-time preview of the recognized expression
   - Allows manual editing of the LaTeX code for corrections

2. **Multi-Engine Problem Solving**:
   - Claude-powered step-by-step explanations with Python calculations
   - MATLAB integration for complex numerical computations
   - Wolfram Alpha integration for symbolic mathematics
   - Automatically formats mathematical output in LaTeX

3. **Interactive User Interface**:
   - Clean, intuitive Streamlit interface
   - Live LaTeX previews
   - Multiple solver options for different types of problems
   - Clear presentation of solutions and explanations

4. **Comprehensive Solution Output**:
   - Detailed step-by-step explanations
   - Rigorous mathematical proofs when applicable
   - Numerical results with appropriate precision
   - Visualizations and graphs where relevant

## How we built it

We constructed MathPix using a modern tech stack and multiple specialized components:

1. **Core Technologies**:
   - Python for the backend logic and integration
   - Streamlit for the web interface
   - LaTeX for mathematical typesetting

2. **Key Components**:
   - **OCR Engine**: Implemented using Pix2Tex, a specialized mathematical OCR model
   - **Computation Engines**:
     - Anthropic's Claude API for natural language processing and step-by-step solutions
     - MATLAB integration using Transplant for numerical computations
     - Wolfram Alpha API for symbolic mathematics
   - **Frontend**:
     - Streamlit components for UI elements
     - MathJax for LaTeX rendering
     - Custom file handling for image uploads

3. **Integration Layer**:
   - Custom wrapper classes for each computation engine
   - Unified error handling and output formatting
   - Asynchronous execution for responsive UI
   - Temporary file management for secure operation

4. **Development Process**:
   - Started with core OCR functionality
   - Added step-by-step solution generation using Claude
   - Integrated MATLAB for numerical computations
   - Added Wolfram Alpha for symbolic mathematics
   - Implemented unified LaTeX output formatting
   - Created clean, intuitive user interface
   - Added error handling and input validation
   - Optimized performance and resource usage

The project represents a careful balance of different technologies, each chosen for its strengths:

- Claude for natural language understanding and explanation generation
- MATLAB for efficient numerical computations
- Wolfram Alpha for symbolic mathematics
- Streamlit for rapid UI development
- LaTeX for professional mathematical typesetting

## Deployment

```sh
pip install requirements.txt
streamlit run app.py
```

## Contributors

- [Alex Feng](https://github.com/Alexander-Feng)
- [David Chang](https://github.com/dchang0524)

## Licence

Licensed under the MIT License.
