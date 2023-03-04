from setuptools import find_packages, setup
import versioneer

# # Setup Package # #

setup(
    name="pyopf",
    author="Naeem Turner-Bandele",
    maintainer="Naeem Turner-Bandele",
    maintainer_email="naeem627@users.noreply.github.com",
    description="A package to perform AC Optimal Power Flow using a current-voltage formulation.",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    include_package_data=True,
    package_dir={'pyopf': 'pyopf'},
    python_requires='>=3.10',
    zip_safe=False
)
