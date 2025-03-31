from setuptools import setup, find_packages

setup(
    name="face_recog_lib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "insightface",
        "numpy"
    ],
    author="Orkun Acar",
    description="A face recognition library using insightface",
)
