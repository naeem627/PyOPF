from setuptools import find_packages, setup

import versioneer

# # Install Requirements # #
install_requires = ["numpy", "scipy", "networkx", "sympy", "pyomo", "termcolor"]

# # Setup Package # #

setup(
    name="pyopf",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Naeem Turner-Bandele",
    maintainer="Naeem Turner-Bandele",
    maintainer_email="naeem@naeem.engineer",
    description="A package to perform AC Optimal Power Flow using a current-voltage formulation.",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=install_requires,
    zip_safe=False
)
