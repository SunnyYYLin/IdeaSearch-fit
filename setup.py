from setuptools import setup
from setuptools import find_packages


setup(
    name = "IdeaSearch-fit",
    version = "0.0.2",
    packages = find_packages(),
    description = "Extension of IdeaSearch for data fitting",
    author = "parkcai",
    author_email = "sun_retailer@163.com",
    url = "https://github.com/IdeaSearch/IdeaSearch-fit",
    include_package_data = True,
    package_data = {
        "IdeaSearch_fit": ["locales/**/LC_MESSAGES/*.mo"],
    },
    python_requires = ">=3.8",
    install_requires = [
        "pywheels>=0.6.5",
    ],
)