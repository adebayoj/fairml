from setuptools import setup

setup(
    name='fairml',
    version='0.1.1.5',
    description=("Module for measuring feature dependence"
                 " for black-box models"),
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
