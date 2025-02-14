from setuptools import setup, find_packages

setup(
    name="underwater_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'fastapi>=0.68.0',
        'python-multipart>=0.0.5',
        'uvicorn>=0.15.0',
        'streamlit>=1.0.0',
        'Pillow>=8.0.0',
        'python-dotenv>=0.19.0',
        'plotly>=5.0.0',
    ],
)
