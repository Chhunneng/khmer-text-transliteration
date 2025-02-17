from setuptools import setup, find_packages

setup(
    name='khmer-english-transliteration',
    version='0.1.4',
    author='Chanchhunneng Chrea',
    author_email='chreachanchhunneng@gmail.com',
    description='A Python package for transliterating English text to Khmer script.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Chhunneng/khmer-text-transliteration',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow>=2.0',
        'numpy',
        'scikit-learn',
        'python-Levenshtein',
        'rapidfuzz',
        'gradio'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
) 