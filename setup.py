from setuptools import setup, find_packages

setup(
    name='signature-verification',
    version='2.0.0',
    description='A comprehensive signature verification system using machine learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='V.S. Sanjana',
    url='https://github.com/V-S-Sanjana/signature-verification',
    packages=find_packages(),
    install_requires=[
        'Flask==2.3.3',
        'opencv-python==4.8.1.78',
        'tensorflow==2.13.0',
        'numpy==1.24.3',
        'Pillow==10.0.1',
        'Werkzeug==2.3.7'
    ],
    entry_points={
        'console_scripts': [
            'signature-verify = app:main',
            'train-models = train_models:main'
        ]
    },
    python_requires='>=3.8',
    license=open('LICENSE').read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
