from setuptools import setup

setup(
    name='my_package',
    version='0.1',
    packages=['src'],
    install_requires=[
        'marshmallow',
        'unsloth',
        'setproctitle',
        'common_ml @ git+ssh://git@github.com/qluvio/common-ml.git#egg=common_ml',
    ]
)
