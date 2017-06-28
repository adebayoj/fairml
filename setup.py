from setuptools import setup


def convert_to_rst(filename):
    """
    Nice markdown to .rst hack. PyPI needs .rst.
    
    Uses pandoc to convert the README.md
    
    From https://coderwall.com/p/qawuyq/use-markdown-readme-s-in-python-modules
    """
    try:
        import pypandoc
        long_description = pypandoc.convert(filename, 'rst')
        long_description = long_description.replace("\r", "")  # YOU  NEED THIS LINE
    except (ImportError, OSError):
        print("Pandoc not found. Long_description conversion failure.")
        import io
        # pandoc is not installed, fallback to using raw contents
        with io.open(filename, encoding="utf-8") as f:
            long_description = f.read()

    return long_description

setup(
    name='fairml',
    version='0.1.1.5.rc08',
    description=("Module for measuring feature dependence"
                 " for black-box models"),
    long_description=convert_to_rst('README.md'),
    url='https://github.com/adebayoj/fairml',
    author='Julius Adebayo',
    author_email='julius.adebayo@gmail.com',
    license='MIT',
    packages=['fairml'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn>=0.18',
        'matplotlib>=1.5.3',
        'seaborn>=0.7.1',
        'pandas>=0.19.0'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
