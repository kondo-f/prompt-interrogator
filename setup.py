from distutils.core import setup


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name='prompt-interrogator',
    version='0.0.1',
    packages=['prompt_interrogator'],
    url='https://github.com/kondo-f/prompt-interrogator',
    author='kondo',
    description='',
    install_requires=_requires_from_file('requirements.txt'),
    package_data={
        'prompt_interrogator': ['text/*.txt'],
    },
)
